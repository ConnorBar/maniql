  Opus Prompt

  You are reviewing a complete offline IQL (Implicit Q-Learning) pipeline for robot manipulation.
  The repo is `torch-maniql`. Below is the full source of every relevant file. Please do a holistic
  audit covering: correctness of the IQL algorithm implementation, data pipeline validity,
  architectural choices, regularization / overfitting risk, and anything that would cause training
  to fail silently or not learn well.

  ---

  ## Dataset context

  - 40 human demonstration episodes, concatenated into a single flat pickle (~9GB, ~wrist_state mode
    used in practice: 224×224×3 uint8 wrist camera + 7-dim joint state)
  - Actions are 7-dim continuous, clipped to [-1+ε, 1-ε]
  - Rewards are likely sparse (no normalization applied by default)
  - Each file = 1 episode; episodes are concatenated with `dones` flags marking episode boundaries
  - train_test_split is episode-level: 10% test (~4 episodes), 90% train (~36 episodes)
  - Batch size 128, 1M (testing with 50k) gradient steps → roughly 5,000–125,000 passes through the training data
    depending on episode length (severe overfitting risk territory for 36 episodes)

  Key hyperparameters: discount=0.99, tau=0.005, expectile=0.8, temperature=0.1,
  hidden_dims=[256,256], actor_lr=critic_lr=value_lr=3e-4, backbone=resnet18 (no R3M by default)

  ---

  ## Full source files

  ### `seed_data.py` — raw pkl → preprocessed dataset
"""Preprocessor: raw transition files -> IQL-ready pickle with raw observations.

Unlike the previous pipeline, R3M encoding is **not** applied here.  Images
are stored as uint8 (224x224x3) and the vision backbone runs during training
so it can be finetuned end-to-end.

Two pipeline modes
------------------
  wrist_state  : wrist (224x224x3 uint8)  + state (7)
  full         : wrist + tactile (224x224x3 uint8) + force (420 float32) + state

Usage
-----
    # wrist + state
    python seed_data.py \\
        --input-dir data --mode wrist_state \\
        --output data/preprocessed/raw_wrist_state.pkl

    # full multimodal
    python seed_data.py \\
        --input-dir data --mode full \\
        --output data/preprocessed/raw_full.pkl
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
from PIL import Image
from tqdm import tqdm

from obs_modality import get_split_keys, Modality

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess raw transitions into an IQL-ready pickle "
                    "(no R3M encoding -- images stored raw for in-training "
                    "backbone finetuning)."
    )
    p.add_argument("--input-dir", type=str,
                   default="data")
    p.add_argument("--input-glob", type=str, default="*_transitions.pkl")
    p.add_argument("--output", type=str,
                   default="data/preprocessed/all_transitions_raw.pkl")
    p.add_argument("--mode", type=str, default="wrist_state",
                   choices=["wrist_state", "full"],
                   help="Pipeline mode: 'wrist_state' or 'full'.")
    p.add_argument("--image-size", type=int, default=224,
                   help="Resize images to this square size (default 224).")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--chunk-size-files", type=int, default=3,
                   help="Flush a chunk every N input files.")
    p.add_argument("--chunk-dir", type=str, default="")
    p.add_argument("--keep-chunks", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Image helpers
# ---------------------------------------------------------------------------

def _to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """Ensure an image array is uint8 HWC."""
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.dtype != np.uint8:
        maxv = float(np.max(arr)) if arr.size else 0.0
        if maxv <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_uint8(img_hwc: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 image to ``(size, size, 3)`` via bilinear."""
    pil = Image.fromarray(img_hwc)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _squeeze_to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).squeeze()


# ---------------------------------------------------------------------------
#  Per-file extraction
# ---------------------------------------------------------------------------

TACTILE_RAW_SHAPE = (160, 120, 3)


def preprocess_file(
    file_path: Path,
    mode: str,
    image_size: int,
    assigned_seed: int,
) -> Dict[str, np.ndarray]:
    """Convert one raw transitions file into arrays keyed by modality."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    transitions = (data.get("transitions", data)
                   if isinstance(data, dict) else data)
    if not isinstance(transitions, list) or len(transitions) == 0:
        raise ValueError(f"Bad transitions in {file_path}")

    # Trim trailing duplicate-done tail
    first_done = None
    for idx, tr in enumerate(transitions):
        if bool(np.asarray(tr["done"]).reshape(-1)[0]):
            first_done = idx
            break
    if first_done is not None:
        transitions = transitions[:first_done + 1]

    split_keys = get_split_keys(mode)
    obs_buf: Dict[Modality, list] = {k: [] for k in split_keys}
    next_buf: Dict[Modality, list] = {k: [] for k in split_keys}
    actions, rewards, dones, success, timeouts = [], [], [], [], []
    seeds: list = []
    source_files: list = []

    for tr in transitions:
        # --- wrist (always present) ---
        w = _resize_uint8(_to_uint8_hwc(tr["obs"]["wrist"]), image_size)
        nw = _resize_uint8(_to_uint8_hwc(tr["next_obs"]["wrist"]), image_size)
        obs_buf["wrist"].append(w)
        next_buf["wrist"].append(nw)

        # --- state (always present) ---
        obs_buf["state"].append(
            _squeeze_to_float32(tr["obs"]["state"]).reshape(-1))
        next_buf["state"].append(
            _squeeze_to_float32(tr["next_obs"]["state"]).reshape(-1))

        if mode == "full":
            # --- tactile ---
            def _tact(obs_key):
                raw = _squeeze_to_float32(
                    tr[obs_key]["right_tactile_camera_taxim"]).reshape(-1)
                hwc = raw.reshape(TACTILE_RAW_SHAPE)
                return _resize_uint8(
                    np.clip(hwc * 255.0 if hwc.max() <= 1.5 else hwc,
                            0, 255).astype(np.uint8),
                    image_size,
                )
            obs_buf["tactile"].append(_tact("obs"))
            next_buf["tactile"].append(_tact("next_obs"))

            # --- force field (stored as raw float vector) ---
            obs_buf["force"].append(
                _squeeze_to_float32(
                    tr["obs"]["tactile_force_field_right"]).reshape(-1))
            next_buf["force"].append(
                _squeeze_to_float32(
                    tr["next_obs"]["tactile_force_field_right"]).reshape(-1))

        # --- scalars ---
        actions.append(_squeeze_to_float32(tr["action"]).reshape(-1))
        rewards.append(float(np.asarray(tr["reward"]).reshape(-1)[0]))
        dones.append(float(bool(np.asarray(tr["done"]).reshape(-1)[0])))
        success.append(float(bool(np.asarray(tr["success"]).reshape(-1)[0])))
        timeouts.append(float(bool(np.asarray(tr["timeout"]).reshape(-1)[0])))
        seeds.append(int(assigned_seed))
        source_files.append(file_path.name)

    result = {
        "obs": {k: np.stack(obs_buf[k]) for k in split_keys},
        "next_obs": {k: np.stack(next_buf[k]) for k in split_keys},
        "actions": np.stack(actions).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "success": np.asarray(success, dtype=np.float32),
        "timeouts": np.asarray(timeouts, dtype=np.float32),
        "seed": np.asarray(seeds, dtype=np.int32),
        "source_file": np.asarray(source_files, dtype=object),
    }
    return result


# ---------------------------------------------------------------------------
#  Chunking / merging  (same flow as before, adapted for new keys)
# ---------------------------------------------------------------------------

def flush_chunk(chunk_idx: int, chunk_dir: Path, file_records: list,
                buffers: dict, split_keys):
    if len(buffers["obs"][split_keys[0]]) == 0:
        return
    chunk_path = chunk_dir / f"chunk_{chunk_idx:05d}.pkl"

    def _cat(lst, dtype=None):
        arr = np.concatenate(lst, axis=0)
        return arr if dtype is None else arr.astype(dtype)

    payload = {
        "obs": {k: _cat(buffers["obs"][k]) for k in split_keys},
        "next_obs": {k: _cat(buffers["next_obs"][k]) for k in split_keys},
        "actions": _cat(buffers["actions"], np.float32),
        "rewards": _cat(buffers["rewards"], np.float32),
        "dones": _cat(buffers["dones"], np.float32),
        "success": _cat(buffers["success"], np.float32),
        "timeouts": _cat(buffers["timeouts"], np.float32),
        "seed": _cat(buffers["seed"], np.int32),
        "source_file": np.concatenate(buffers["source_file"], axis=0),
        "file_records": file_records,
    }
    with open(chunk_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _empty_buffers(split_keys):
    scalar_keys = ["actions", "rewards", "dones", "success", "timeouts",
                   "seed", "source_file"]
    return {
        "obs": {k: [] for k in split_keys},
        "next_obs": {k: [] for k in split_keys},
        **{k: [] for k in scalar_keys},
    }


def merge_chunks(chunk_dir: Path, output_path: Path, metadata: dict,
                 split_keys):
    chunk_files = sorted(chunk_dir.glob("chunk_*.pkl"))
    if not chunk_files:
        raise RuntimeError(f"No chunk files in {chunk_dir}")

    all_records: list = []
    total = 0
    for cf in chunk_files:
        with open(cf, "rb") as f:
            c = pickle.load(f)
        total += int(c["actions"].shape[0])
        all_records.extend(c["file_records"])

    # Merge arrays
    obs_arrs = {k: [] for k in split_keys}
    nobs_arrs = {k: [] for k in split_keys}
    scalar_arrs = {sk: [] for sk in
                   ["actions", "rewards", "dones", "success", "timeouts",
                    "seed"]}
    source_arrs: list = []

    for cf in tqdm(chunk_files, desc="Merging"):
        with open(cf, "rb") as f:
            c = pickle.load(f)
        for k in split_keys:
            obs_arrs[k].append(c["obs"][k])
            nobs_arrs[k].append(c["next_obs"][k])
        for sk in scalar_arrs:
            scalar_arrs[sk].append(c[sk])
        source_arrs.append(c["source_file"])

    # Build file_index
    file_index: list = []
    cursor = 0
    for rec in all_records:
        n = int(rec["num_transitions"])
        file_index.append({
            "source_file": rec["source_file"],
            "seed": int(rec["seed"]),
            "num_transitions": n,
            "start_idx": cursor,
            "end_idx_exclusive": cursor + n,
        })
        cursor += n

    final = {
        "metadata": metadata,
        "file_index": file_index,
        "obs": {k: np.concatenate(obs_arrs[k]) for k in split_keys},
        "next_obs": {k: np.concatenate(nobs_arrs[k]) for k in split_keys},
        **{sk: np.concatenate(scalar_arrs[sk]) for sk in scalar_arrs},
        "terminals": np.clip(
            np.concatenate(scalar_arrs["dones"])
            - np.concatenate(scalar_arrs["timeouts"]),
            0.0, 1.0).astype(np.float32),
        "source_file": np.concatenate(source_arrs),
    }
    with open(output_path, "wb") as f:
        pickle.dump(final, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mode = args.mode
    split_keys = get_split_keys(mode)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_dir.glob(args.input_glob), key=lambda p: p.name)
    if not all_files:
        raise FileNotFoundError(
            f"No files matching '{args.input_glob}' in {input_dir}")

    chunk_dir = (Path(args.chunk_dir) if args.chunk_dir
                 else output_path.parent / f"{output_path.stem}_chunks")
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] mode={mode}  keys={split_keys}  "
          f"image_size={args.image_size}  files={len(all_files)}")

    buffers = _empty_buffers(split_keys)
    chunk_idx = 0
    files_in_chunk = 0
    file_records: list = []
    total_transitions = 0

    for file_idx, fp in enumerate(tqdm(all_files, desc="Preprocessing")):
        assigned_seed = int(args.seed_start) + file_idx
        arrs = preprocess_file(fp, mode, args.image_size, assigned_seed)

        for k in split_keys:
            buffers["obs"][k].append(arrs["obs"][k])
            buffers["next_obs"][k].append(arrs["next_obs"][k])
        for sk in ["actions", "rewards", "dones", "success", "timeouts",
                    "seed", "source_file"]:
            buffers[sk].append(arrs[sk])

        n_here = int(arrs["obs"][split_keys[0]].shape[0])
        file_records.append({
            "source_file": fp.name,
            "seed": assigned_seed,
            "num_transitions": n_here,
        })
        total_transitions += n_here
        files_in_chunk += 1

        if files_in_chunk >= args.chunk_size_files:
            flush_chunk(chunk_idx, chunk_dir, file_records, buffers,
                        split_keys)
            chunk_idx += 1
            files_in_chunk = 0
            file_records = []
            buffers = _empty_buffers(split_keys)

    if files_in_chunk > 0:
        flush_chunk(chunk_idx, chunk_dir, file_records, buffers, split_keys)

    metadata = {
        "mode": mode,
        "split_keys": list(split_keys),
        "image_size": args.image_size,
        "input_dir": str(input_dir),
        "input_glob": args.input_glob,
        "num_files": len(all_files),
        "num_transitions": total_transitions,
        "seed_start": int(args.seed_start),
        "file_sort": "filename ascending",
    }
    merge_chunks(chunk_dir, output_path, metadata, split_keys)

    if not args.keep_chunks:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    print(f"[DONE] {output_path}  ({total_transitions:,} transitions, "
          f"mode={mode})")


if __name__ == "__main__":
    main()

  ### `manifeel_iql.py` — dataset loader + Batch interface

"""Dataset loader: preprocessed pickle -> IQL Batch interface.

Supports both pipeline modes:
  wrist_state : obs keys = ("wrist", "state")
  full        : obs keys = ("wrist", "tactile", "force", "state")

Image modalities (wrist, tactile) are kept as uint8 in memory and returned
as-is in batches.  The model handles uint8 -> float32 conversion on GPU
during the forward pass (via ``r3m_preprocess``).
"""

import collections
import pickle
import sys

import numpy as np

# Pickle files saved with NumPy 2.x reference numpy._core, which doesn't exist in 1.x
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)  # type: ignore[attr-defined]

from obs_modality import IMAGE_KEYS, get_split_keys

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


class ManiFeelDataset:
    """Loads a preprocessed pickle and exposes ``sample(batch_size) -> Batch``.

    Args:
        pkl_path: Path to the preprocessed pickle.
        clip_actions: Clip actions to ``[-1+eps, 1-eps]``.
        eps: Clipping epsilon.
    """

    def __init__(self, pkl_path: str, clip_actions: bool = True,
                 eps: float = 1e-5):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.metadata = data.get("metadata", {})
        self.file_index = data.get("file_index", [])

        mode = self.metadata.get("mode")
        if mode is None:
            raise ValueError(
                "Pickle is missing metadata['mode']. "
                "Re-run seed_data.py with the new pipeline."
            )
        self._mode = mode
        self._split_keys = get_split_keys(mode)

        for k in self._split_keys:
            if k not in data.get("obs", {}):
                raise ValueError(
                    f"Expected obs key {k!r} for mode={mode!r} but not found."
                )

        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32).ravel()
        dones = data["dones"].astype(np.float32).ravel()

        if clip_actions:
            lim = 1.0 - eps
            actions = np.clip(actions, -lim, lim)

        # Images stay as their native dtype (uint8); vectors become float32.
        self._obs = {}
        self._next_obs = {}
        for k in self._split_keys:
            if k in IMAGE_KEYS:
                self._obs[k] = data["obs"][k]
                self._next_obs[k] = data["next_obs"][k]
            else:
                self._obs[k] = data["obs"][k].astype(np.float32)
                self._next_obs[k] = data["next_obs"][k].astype(np.float32)

        self._indices = None
        self.size = len(self._obs[self._split_keys[0]])

        self.actions = actions
        self.rewards = rewards
        terminals = dones.astype(np.float32)
        self.masks = (1.0 - terminals).astype(np.float32)
        self.dones_float = dones.copy()
        self.terminals = terminals

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def split_keys(self):
        return self._split_keys

    def observation_example(self):
        """Single observation with leading batch dim 1 (for model init)."""
        if self._indices is None:
            return {k: self._obs[k][:1] for k in self._split_keys}
        i = int(self._indices[0])
        return {k: self._obs[k][i:i + 1] for k in self._split_keys}

    def _pack_obs(self, idx):
        if self._indices is not None:
            idx = self._indices[idx]
        return {k: self._obs[k][idx] for k in self._split_keys}

    def _pack_next(self, idx):
        if self._indices is not None:
            idx = self._indices[idx]
        return {k: self._next_obs[k][idx] for k in self._split_keys}

    # ------ construction helpers ------------------------------------------

    @classmethod
    def _from_dicts(cls, obs, next_obs, actions, rewards, masks,
                    dones_float, terminals, split_keys, mode,
                    metadata=None, indices=None):
        ds = object.__new__(cls)
        ds.metadata = metadata or {}
        ds.file_index = []
        ds._mode = mode
        ds._split_keys = split_keys
        ds._obs = {k: obs[k] for k in split_keys}
        ds._next_obs = {k: next_obs[k] for k in split_keys}
        ds.actions = actions
        ds.rewards = rewards
        ds.masks = masks
        ds.dones_float = dones_float
        ds.terminals = terminals
        ds._indices = indices
        ds.size = (int(len(indices)) if indices is not None
                   else len(obs[split_keys[0]]))
        return ds

    def train_test_split(self, test_ratio: float = 0.1, seed: int = 42):
        rng = np.random.RandomState(seed)

        # IMPORTANT: Use index-based split so we don't duplicate multi-GB tactile arrays.
        # We always split in the base dataset index space.
        if self._indices is not None:
            base_done = self.dones_float[self._indices]
            ep_ends = np.where(base_done == 1.0)[0]
            base_indices = self._indices
        else:
            ep_ends = np.where(self.dones_float == 1.0)[0]
            base_indices = None

        n_eps = len(ep_ends)
        n_test = max(1, int(n_eps * test_ratio))
        ep_order = rng.permutation(n_eps)
        test_ep_set = set(ep_order[:n_test].tolist())

        train_idx, test_idx = [], []
        ep_start = 0
        for ep_i, ep_end in enumerate(ep_ends):
            indices = np.arange(ep_start, ep_end + 1)
            (test_idx if ep_i in test_ep_set else train_idx).append(indices)
            ep_start = ep_end + 1

        local_size = (int(len(self._indices)) if self._indices is not None
                      else self.size)
        if ep_start < local_size:
            train_idx.append(np.arange(ep_start, local_size))

        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)

        if base_indices is not None:
            train_idx = base_indices[train_idx]
            test_idx = base_indices[test_idx]

        def _make(idxs):
            return ManiFeelDataset._from_dicts(
                self._obs, self._next_obs, self.actions, self.rewards,
                self.masks, self.dones_float, self.terminals,
                self._split_keys, self._mode, self.metadata,
                indices=idxs.astype(np.int64),
            )

        return _make(train_idx), _make(test_idx)

    # ------ IQL interface -------------------------------------------------

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self._pack_obs(idx),
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            masks=self.masks[idx],
            next_observations=self._pack_next(idx),
        )

    # ------ diagnostics ---------------------------------------------------

    def validate(self) -> bool:
        """Run sanity checks and print warnings.  Returns True if clean."""
        ok = True

        def _check(name, arr, batch=256):
            nonlocal ok
            a = np.asarray(arr)
            if a.dtype == np.uint8:
                return True
            n0 = int(a.shape[0]) if a.ndim > 0 else 1
            if a.ndim == 0:
                if np.isnan(a) or np.isinf(a):
                    print(f"[WARN] Bad value in {name}")
                    ok = False
                return ok
            for i in range(0, n0, batch):
                sl = slice(i, min(i + batch, n0))
                chunk = a[sl]
                if np.isnan(chunk).any():
                    print(f"[WARN] NaN detected in {name} (chunk {sl.start}:{sl.stop})")
                    ok = False
                    return False
                if np.isinf(chunk).any():
                    print(f"[WARN] Inf detected in {name} (chunk {sl.start}:{sl.stop})")
                    ok = False
                    return False
            return True

        for k in self._split_keys:
            _check(f"obs.{k}", self._obs[k])
            _check(f"next_obs.{k}", self._next_obs[k])
        _check("actions", self.actions)
        _check("rewards", self.rewards)

        n_eps = int(self.dones_float.sum())
        ep_lengths = []
        cur = 0
        for i in range(self.size):
            cur += 1
            if self.dones_float[i] == 1.0:
                ep_lengths.append(cur)
                cur = 0
        if cur > 0:
            ep_lengths.append(cur)
        el = np.array(ep_lengths)

        print(f"[INFO] {self.size:,} transitions, {n_eps} episodes, "
              f"ep_len: mean={el.mean():.0f} median={np.median(el):.0f} "
              f"min={el.min()} max={el.max()}")
        if ok:
            print("[OK] No data integrity issues found.")
        return ok

    def summary(self) -> str:
        n_eps = int(self.dones_float.sum())
        obs_parts = []
        for k in self._split_keys:
            shp = self._obs[k].shape[1:]
            dt = self._obs[k].dtype
            obs_parts.append(f"{k}{shp}[{dt}]")
        obs_line = "  obs: " + " + ".join(obs_parts)

        lines = [
            f"ManiFeelDataset [{self._mode}]: "
            f"{self.size:,} transitions, {n_eps} episodes",
            obs_line,
            f"  action dim:    {self.actions.shape[1]}",
            f"  reward range:  [{self.rewards.min():.4f}, "
            f"{self.rewards.max():.4f}]",
            f"  reward mean:   {self.rewards.mean():.4f}",
            f"  terminals:     {int(self.terminals.sum())} / {self.size}",
        ]
        return "\n".join(lines)

  ### `obs_modality.py`
"""Observation modality constants for the two pipeline modes.

Mode "wrist_state":
    wrist (raw image 224x224x3) + state (7-dim vector)

Mode "full":
    wrist (raw image 224x224x3) + tactile (raw image 224x224x3)
    + force (420-dim vector) + state (7-dim vector)
"""

from typing import Literal

WRIST_STATE_KEYS = ("wrist", "state")
FULL_KEYS = ("wrist", "tactile", "force", "state")

IMAGE_KEYS = frozenset({"wrist", "tactile"})

VALID_MODES = ("wrist_state", "full")

Modality = Literal["wrist", "tactile", "force", "state"]

def get_split_keys(mode: str):
    if mode == "wrist_state":
        return WRIST_STATE_KEYS
    if mode == "full":
        return FULL_KEYS
    raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}")
  ### `vision_backbone.py` — ResNetBackbone, force_to_image, r3m_preprocess_bhwc
"""PyTorch-native vision utilities + R3M-backed ResNet feature extractor.

This project trains IQL end-to-end in PyTorch. Image observations are stored
raw (uint8, HWC, [0..255]) in the dataset and are preprocessed on GPU here.

Key requirements:
- **No JAX/Flax** dependencies/imports.
- The backbone must remain **trainable** (no freezing) so gradients from IQL
  losses update the visual encoder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESNET_OUT_DIM: Dict[str, int] = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

FORCE_FIELD_DIM = 420
FORCE_GRID_SHAPE = (14, 10, 3)

# ---------------------------------------------------------------------------
#  Preprocessing helpers
# ---------------------------------------------------------------------------

def r3m_preprocess_bhwc(images: torch.Tensor) -> torch.Tensor:
    """Convert BHWC images to normalized BCHW for ImageNet-pretrained ResNets.

    - If input is uint8, assumes [0..255].
    - If input is float, assumes values are already in a reasonable scale for
      normalization (this repo's dataset uses uint8 for real images; float inputs
      are mainly used for synthetic modalities like force-fields).
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected (B,H,W,3) images, got {tuple(images.shape)}")
    x = images
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        if x.dtype != torch.float32:
            x = x.float()
    x = x.permute(0, 3, 1, 2).contiguous()  # BCHW
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def force_to_image(force: torch.Tensor) -> torch.Tensor:
    """(B,420) -> (B,224,224,3) float (unnormalized)."""
    if force.ndim != 2 or force.shape[-1] != FORCE_FIELD_DIM:
        raise ValueError(f"Expected (B,{FORCE_FIELD_DIM}) force, got {tuple(force.shape)}")
    b = force.shape[0]
    x = force.view(b, *FORCE_GRID_SHAPE)  # BHWC
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    x = (x - mean) / std
    # interpolate expects NCHW
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


@dataclass(frozen=True)
class R3MLoadResult:
    arch: str
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]


def _strip_prefix(key: str) -> str:
    for p in ("module.", "convnet."):
        if key.startswith(p):
            return key[len(p):]
    return key


def load_r3m_resnet_weights(backbone: nn.Module, checkpoint_path: str) -> R3MLoadResult:
    """Load R3M convnet weights into a torchvision ResNet backbone.

    Accepts checkpoints that are either:
    - raw state_dict (convnet.* keys), or
    - dict with 'r3m' key (common in this repo).
    """
    path = os.path.expanduser(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["r3m"] if isinstance(ckpt, dict) and "r3m" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    cleaned = {_strip_prefix(k): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    msg = backbone.load_state_dict(cleaned, strict=False)
    return R3MLoadResult(
        arch=backbone.__class__.__name__,
        missing_keys=tuple(getattr(msg, "missing_keys", [])),
        unexpected_keys=tuple(getattr(msg, "unexpected_keys", [])),
    )


class ResNetBackbone(nn.Module):
    """ResNet feature extractor with optional R3M initialization (trainable)."""

    def __init__(self, arch: str = "resnet18", r3m_checkpoint: str | None = None):
        super().__init__()
        if arch not in RESNET_OUT_DIM:
            raise ValueError(f"Unknown arch {arch!r}; expected one of {sorted(RESNET_OUT_DIM)}")
        if arch == "resnet18":
            net = torchvision.models.resnet18(weights=None)
        elif arch == "resnet34":
            net = torchvision.models.resnet34(weights=None)
        else:
            net = torchvision.models.resnet50(weights=None)
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = RESNET_OUT_DIM[arch]

        if r3m_checkpoint:
            res = load_r3m_resnet_weights(self.net, r3m_checkpoint)
            if res.unexpected_keys:
                # Most unexpected keys are language heads; safe to ignore.
                pass

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        # torchvision ResNet expects normalized BCHW float
        return self.net(x_bchw)

  ### `multimodal_nets.py` — MultiModalEncoder, QNet, ValueNet, DiagGaussianPolicy, IQLLearner

"""Multi-modal networks + PyTorch IQL learner (end-to-end trainable R3M).

This file replaces the prior JAX/Flax implementation with pure PyTorch.
The dataset (`manifeel_iql.ManiFeelDataset`) returns a Batch where
`observations` and `next_observations` are dicts containing:

- wrist_state: {"wrist": uint8(B,224,224,3), "state": float32(B,7)}
- full: {"wrist": uint8, "tactile": uint8, "force": float32(B,420), "state": float32}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from vision_backbone import (
    RESNET_OUT_DIM,
    ResNetBackbone,
    force_to_image,
    r3m_preprocess_bhwc,
)

STATE_DIM = 7
FORCE_DIM = 420


def encoded_obs_dim(arch: str, mode: str) -> int:
    feat = RESNET_OUT_DIM[arch]
    if mode == "wrist_state":
        return feat + STATE_DIM
    return feat * 3 + STATE_DIM


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int, *, activate_final: bool = False):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-1], out_dim))
        if activate_final:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalEncoder(nn.Module):
    def __init__(self, arch: str, mode: str, r3m_checkpoint: Optional[str] = None):
        super().__init__()
        self.arch = arch
        self.mode = mode

        self.wrist_backbone = ResNetBackbone(arch=arch, r3m_checkpoint=r3m_checkpoint)
        if mode == "full":
            self.tactile_backbone = ResNetBackbone(arch=arch, r3m_checkpoint=r3m_checkpoint)
            self.force_backbone = ResNetBackbone(arch=arch, r3m_checkpoint=r3m_checkpoint)
        else:
            self.tactile_backbone = None
            self.force_backbone = None

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        wrist = r3m_preprocess_bhwc(obs["wrist"])
        wrist_feat = self.wrist_backbone(wrist)
        parts = [wrist_feat]

        if self.mode == "full":
            tactile = r3m_preprocess_bhwc(obs["tactile"])
            tact_feat = self.tactile_backbone(tactile)  # type: ignore[operator]
            parts.append(tact_feat)

            force_img = force_to_image(obs["force"])
            force_norm = r3m_preprocess_bhwc(force_img)
            force_feat = self.force_backbone(force_norm)  # type: ignore[operator]
            parts.append(force_feat)

        state = obs["state"].float()
        parts.append(state)
        return torch.cat(parts, dim=-1)


class ValueNet(nn.Module):
    def __init__(self, arch: str, mode: str, hidden_dims: Sequence[int], r3m_checkpoint: Optional[str]):
        super().__init__()
        self.encoder = MultiModalEncoder(arch, mode, r3m_checkpoint=r3m_checkpoint)
        self.head = MLP(encoded_obs_dim(arch, mode), hidden_dims, 1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(obs)
        v = self.head(h)
        return v.squeeze(-1)


class QNet(nn.Module):
    def __init__(self, arch: str, mode: str, hidden_dims: Sequence[int], action_dim: int, r3m_checkpoint: Optional[str]):
        super().__init__()
        self.encoder = MultiModalEncoder(arch, mode, r3m_checkpoint=r3m_checkpoint)
        self.head = MLP(encoded_obs_dim(arch, mode) + action_dim, hidden_dims, 1)

    def forward(self, obs: Dict[str, torch.Tensor], act: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        x = torch.cat([h, act], dim=-1)
        q = self.head(x)
        return q.squeeze(-1)


class DoubleQNet(nn.Module):
    def __init__(self, arch: str, mode: str, hidden_dims: Sequence[int], action_dim: int, r3m_checkpoint: Optional[str]):
        super().__init__()
        self.q1 = QNet(arch, mode, hidden_dims, action_dim, r3m_checkpoint=r3m_checkpoint)
        self.q2 = QNet(arch, mode, hidden_dims, action_dim, r3m_checkpoint=r3m_checkpoint)

    def forward(self, obs: Dict[str, torch.Tensor], act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, act), self.q2(obs, act)


class DiagGaussianPolicy(nn.Module):
    def __init__(
        self,
        arch: str,
        mode: str,
        hidden_dims: Sequence[int],
        action_dim: int,
        r3m_checkpoint: Optional[str],
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.encoder = MultiModalEncoder(arch, mode, r3m_checkpoint=r3m_checkpoint)
        self.trunk = MLP(encoded_obs_dim(arch, mode), hidden_dims, hidden_dims[-1], activate_final=True)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def _dist(self, obs: Dict[str, torch.Tensor], temperature: float = 1.0) -> Normal:
        h = self.encoder(obs)
        z = self.trunk(h)
        mean = torch.tanh(self.mean(z))
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) * float(temperature)
        return Normal(mean, std)

    def log_prob(self, obs: Dict[str, torch.Tensor], act: torch.Tensor) -> torch.Tensor:
        dist = self._dist(obs, temperature=1.0)
        # Normal is factorized; sum over action dims
        return dist.log_prob(act).sum(dim=-1)

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        dist = self._dist(obs, temperature=1.0)
        a = dist.mean if deterministic else dist.sample()
        return a.clamp(-1.0, 1.0)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * (diff ** 2)


@dataclass
class IQLInfo:
    actor_loss: float
    critic_loss: float
    value_loss: float
    q_mean: float
    v_mean: float
    adv_mean: float
    backbone_grad_norm: float


class IQLLearner:
    def __init__(
        self,
        *,
        device: torch.device,
        obs_example: Dict[str, np.ndarray],
        action_dim: int,
        arch: str,
        mode: str,
        r3m_checkpoint: Optional[str],
        hidden_dims: Sequence[int],
        actor_lr: float,
        critic_lr: float,
        value_lr: float,
        discount: float,
        tau: float,
        expectile: float,
        temperature: float,
    ):
        self.device = device
        self.arch = arch
        self.mode = mode
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature

        self.actor = DiagGaussianPolicy(
            arch=arch, mode=mode, hidden_dims=hidden_dims, action_dim=action_dim, r3m_checkpoint=r3m_checkpoint
        ).to(device)
        self.critic = DoubleQNet(
            arch=arch, mode=mode, hidden_dims=hidden_dims, action_dim=action_dim, r3m_checkpoint=r3m_checkpoint
        ).to(device)
        self.value = ValueNet(arch=arch, mode=mode, hidden_dims=hidden_dims, r3m_checkpoint=r3m_checkpoint).to(device)
        self.target_critic = DoubleQNet(
            arch=arch, mode=mode, hidden_dims=hidden_dims, action_dim=action_dim, r3m_checkpoint=r3m_checkpoint
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)

    def _to_torch_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            t = torch.as_tensor(v, device=self.device)
            out[k] = t
        return out

    def update(self, batch) -> IQLInfo:
        obs = self._to_torch_obs(batch.observations)
        next_obs = self._to_torch_obs(batch.next_observations)
        act = torch.as_tensor(batch.actions, device=self.device, dtype=torch.float32)
        rew = torch.as_tensor(batch.rewards, device=self.device, dtype=torch.float32)
        msk = torch.as_tensor(batch.masks, device=self.device, dtype=torch.float32)

        # --- value update ---
        with torch.no_grad():
            tq1, tq2 = self.target_critic(obs, act)
            tq = torch.minimum(tq1, tq2)
        v = self.value(obs)
        v_loss = expectile_loss(tq - v, self.expectile).mean()
        self.value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.value_opt.step()

        # --- actor update (AWR-style) ---
        with torch.no_grad():
            v_detached = self.value(obs)
            tq1, tq2 = self.target_critic(obs, act)
            tq = torch.minimum(tq1, tq2)
            adv = tq - v_detached
            weights = torch.exp(adv * self.temperature).clamp(max=100.0)
        logp = self.actor.log_prob(obs, act)
        a_loss = -(weights * logp).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        self.actor_opt.step()

        # --- critic update ---
        with torch.no_grad():
            next_v = self.value(next_obs)
            target_q = rew + self.discount * msk * next_v
        q1, q2 = self.critic(obs, act)
        c_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        g = self.critic.q1.encoder.wrist_backbone.net.conv1.weight.grad
        grad_norm = float(g.norm().detach().cpu()) if g is not None else 0.0
        self.critic_opt.step()

        # --- target critic EMA ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)

        return IQLInfo(
            actor_loss=float(a_loss.detach().cpu()),
            critic_loss=float(c_loss.detach().cpu()),
            value_loss=float(v_loss.detach().cpu()),
            q_mean=float(tq.mean().detach().cpu()),
            v_mean=float(v.mean().detach().cpu()),
            adv_mean=float(adv.mean().detach().cpu()),
            backbone_grad_norm=grad_norm,
        )

    @torch.no_grad()
    def compute_losses(self, batch) -> Dict[str, float]:
        obs = self._to_torch_obs(batch.observations)
        next_obs = self._to_torch_obs(batch.next_observations)
        act = torch.as_tensor(batch.actions, device=self.device, dtype=torch.float32)
        rew = torch.as_tensor(batch.rewards, device=self.device, dtype=torch.float32)
        msk = torch.as_tensor(batch.masks, device=self.device, dtype=torch.float32)

        tq1, tq2 = self.target_critic(obs, act)
        tq = torch.minimum(tq1, tq2)
        v = self.value(obs)
        v_loss = expectile_loss(tq - v, self.expectile).mean()

        next_v = self.value(next_obs)
        target_q = rew + self.discount * msk * next_v
        q1, q2 = self.critic(obs, act)
        c_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        adv = tq - v
        weights = torch.exp(adv * self.temperature).clamp(max=100.0)
        logp = self.actor.log_prob(obs, act)
        a_loss = -(weights * logp).mean()

        return {
            "actor_loss": float(a_loss.detach().cpu()),
            "critic_loss": float(c_loss.detach().cpu()),
            "value_loss": float(v_loss.detach().cpu()),
            "q": float(tq.mean().detach().cpu()),
            "v": float(v.mean().detach().cpu()),
            "adv": float(adv.mean().detach().cpu()),
        }

    @torch.no_grad()
    def sample_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        tobs = self._to_torch_obs(obs)
        a = self.actor.act(tobs, deterministic=deterministic)
        return a.detach().cpu().numpy()

  ### `train_iql.py` — training loop, normalize_rewards, train_test_split usage

"""Offline IQL training on ManiFeel datasets (pure PyTorch, end-to-end vision finetuning)."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
import tqdm

from manifeel_iql import ManiFeelDataset
from multimodal_nets import IQLLearner
from log_utils import init_wandb, setup_logging, wandb_log, write_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_rewards(dataset: ManiFeelDataset) -> None:
    episode_returns = []
    cur = 0.0
    for i in range(dataset.size):
        cur += float(dataset.rewards[i])
        if dataset.dones_float[i] == 1.0:
            episode_returns.append(cur)
            cur = 0.0
    if len(episode_returns) < 2:
        print("[WARN] <2 trajectories, skipping reward normalisation.")
        return
    ret_range = max(episode_returns) - min(episode_returns)
    if ret_range < 1e-8:
        print("[WARN] Flat returns, skipping normalisation.")
        return
    dataset.rewards /= ret_range
    dataset.rewards *= 1000.0
    print(f"[INFO] Rewards normalised (range {ret_range:.4f} -> 1000).")


@torch.no_grad()
def eval_on_dataset(agent: IQLLearner, dataset: ManiFeelDataset, batch_size: int, n_batches: int = 10) -> Dict[str, float]:
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    value_losses: list[float] = []
    q_means: list[float] = []
    v_means: list[float] = []
    adv_means: list[float] = []

    for _ in range(n_batches):
        b = dataset.sample(batch_size)
        info = agent.compute_losses(b)
        actor_losses.append(info["actor_loss"])
        critic_losses.append(info["critic_loss"])
        value_losses.append(info["value_loss"])
        q_means.append(info["q"])
        v_means.append(info["v"])
        adv_means.append(info["adv"])
    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "value_loss": float(np.mean(value_losses)),
        "q": float(np.mean(q_means)),
        "v": float(np.mean(v_means)),
        "adv": float(np.mean(adv_means)),
    }


def write_training_meta(save_dir: str, *, mode: str, arch: str, action_dim: int, hidden_dims, dataset_path: str, obs_example: dict, seed: int, batch_size: int, max_steps: int) -> str:
    obs_shapes = {k: list(v.shape) for k, v in obs_example.items()}
    obs_dtypes = {k: str(v.dtype) for k, v in obs_example.items()}
    meta = {
        "backend": "pytorch",
        "mode": mode,
        "arch": arch,
        "action_dim": int(action_dim),
        "hidden_dims": [int(h) for h in hidden_dims],
        "dataset_path": dataset_path,
        "obs_shapes": obs_shapes,
        "obs_dtypes": obs_dtypes,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
    }
    path = os.path.join(save_dir, "training_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def save_checkpoint(agent: IQLLearner, save_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(save_dir, f"checkpoint_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "value": agent.value.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "value_opt": agent.value_opt.state_dict(),
    }
    torch.save(payload, os.path.join(ckpt_dir, "checkpoint.pt"))
    with open(os.path.join(ckpt_dir, "DONE"), "w") as f:
        f.write(str(step))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--save_dir", default="./runs/manifeel_iql/")
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--r3m_checkpoint", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=1_000_000)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--eval_interval", type=int, default=2000)
    p.add_argument("--save_interval", type=int, default=50_000)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--tqdm", action="store_true", default=True)
    p.add_argument("--normalize_rewards", action="store_true", default=False)
    p.add_argument("--clip_actions", action="store_true", default=True)
    p.add_argument("--validate", action="store_true", default=True)
    # IQL hyperparams
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--value_lr", type=float, default=3e-4)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--expectile", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=0.1)
    # Logging / W&B
    p.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", default="torch-maniql")
    p.add_argument("--wandb_entity", default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb_name", default="", help="W&B run name (optional).")
    p.add_argument("--wandb_group", default="", help="W&B group (optional).")
    p.add_argument("--wandb_tags", nargs="*", default=[], help="W&B tags (optional).")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(args.save_dir, level=args.log_level)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    full_ds = ManiFeelDataset(args.dataset_path, clip_actions=args.clip_actions)
    mode = full_ds.mode
    arch = args.backbone
    r3m_ckpt = args.r3m_checkpoint or None
    logger.info("mode=%s backbone=%s r3m_checkpoint=%s", mode, arch, r3m_ckpt or "(none)")

    train_ds, test_ds = full_ds.train_test_split(test_ratio=args.test_ratio, seed=args.seed)
    logger.info("=== Train ===\n%s", train_ds.summary())
    logger.info("=== Test ===\n%s", test_ds.summary())
    if args.validate:
        logger.info("--- Train validation ---")
        train_ds.validate()
        logger.info("--- Test validation ---")
        test_ds.validate()

    if args.normalize_rewards:
        normalize_rewards(train_ds)
        normalize_rewards(test_ds)

    obs_example = train_ds.observation_example()
    action_dim = int(train_ds.actions.shape[-1])

    agent = IQLLearner(
        device=device,
        obs_example=obs_example,
        action_dim=action_dim,
        arch=arch,
        mode=mode,
        r3m_checkpoint=r3m_ckpt,
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
    )

    meta_path = write_training_meta(
        args.save_dir,
        mode=mode,
        arch=arch,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        dataset_path=os.path.abspath(args.dataset_path),
        obs_example=obs_example,
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )
    wandb = init_wandb(
        enabled=bool(args.wandb) and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        group=args.wandb_group or None,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        save_dir=args.save_dir,
        config=vars(args),
    )
    if wandb is not None:
        try:
            wandb.save(meta_path)
        except Exception:
            pass

    metrics_path = os.path.join(args.save_dir, "metrics", "metrics.jsonl")

    logger.info("[START] Training IQL for %s steps (batch=%s)", f"{args.max_steps:,}", args.batch_size)
    it = range(1, args.max_steps + 1)
    if args.tqdm:
        it = tqdm.tqdm(it, smoothing=0.1)

    for step in it:
        batch = train_ds.sample(args.batch_size)
        info = agent.update(batch)

        if step % args.log_interval == 0:
            train_metrics = {
                "train/actor_loss": float(info.actor_loss),
                "train/critic_loss": float(info.critic_loss),
                "train/value_loss": float(info.value_loss),
                "train/q": float(info.q_mean),
                "train/v": float(info.v_mean),
                "train/adv": float(info.adv_mean),
                "train/backbone_grad_norm": float(info.backbone_grad_norm),
            }
            wandb_log(wandb, train_metrics, step=step)
            write_jsonl(metrics_path, {"step": int(step), **train_metrics})
            logger.info(
                "step=%d actor=%.6f critic=%.6f value=%.6f q=%.4f v=%.4f adv=%.4f grad=%.4f",
                step,
                train_metrics["train/actor_loss"],
                train_metrics["train/critic_loss"],
                train_metrics["train/value_loss"],
                train_metrics["train/q"],
                train_metrics["train/v"],
                train_metrics["train/adv"],
                train_metrics["train/backbone_grad_norm"],
            )

        if step % args.eval_interval == 0:
            # Note: current eval uses agent.update() for metric computation; keep eval_interval large.
            test_info = eval_on_dataset(agent, test_ds, args.batch_size, n_batches=5)
            test_metrics = {f"test/{k}": float(v) for k, v in test_info.items()}
            wandb_log(wandb, test_metrics, step=step)
            write_jsonl(metrics_path, {"step": int(step), **test_metrics})
            logger.info(
                "eval step=%d actor=%.6f critic=%.6f value=%.6f q=%.4f v=%.4f adv=%.4f",
                step,
                test_metrics["test/actor_loss"],
                test_metrics["test/critic_loss"],
                test_metrics["test/value_loss"],
                test_metrics["test/q"],
                test_metrics["test/v"],
                test_metrics["test/adv"],
            )

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(agent, args.save_dir, step)

    save_checkpoint(agent, args.save_dir, args.max_steps)
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    logger.info("[DONE] Checkpoints in %s", args.save_dir)


if __name__ == "__main__":
    main()

  ---

  ## Questions I want you to answer

  **1. IQL algorithm correctness**
  - Is the update order (value → actor → critic → target EMA) correct per the IQL paper?
  - Is the expectile loss correctly implemented? Is it being minimized in the right direction?
  - The actor uses `weights = exp(adv * temperature).clamp(max=100)` — is the clamping
    value sensible? What does it do to the gradient when adv is large?
  - The actor's log_prob uses a Normal(tanh(mean), std) and computes log_prob(act) directly
    — is this correct for IQL's AWR objective, or should there be a tanh change-of-variables
    correction? Would this cause numerical issues near ±1?
  - The `log_std` is a global parameter (not input-conditioned). What are the implications?

  **2. Terminal/timeout masking**
  - `manifeel_iql.py` sets `masks = 1 - dones`. The pickle from `seed_data.py` stores both
    `dones` (includes timeouts) and a separate `terminals` field (dones minus timeouts, clipped).
    The dataset loader ignores `terminals` and uses `dones` directly for masks. If there are
    timeout episodes in the data (robot hits time limit, not a true failure), they get mask=0
    (treated as terminal). Is this a correctness issue for Bellman updates? How bad is it?

  **3. Architecture: separate encoders for actor/critic/value**
  - Each of the three networks (actor, critic, value) has its own `MultiModalEncoder`, meaning
    3 separate ResNet18s with no parameter sharing. What are the implications for: memory,
    training stability, consistency of the advantage estimate Q - V, and representation collapse?
  - In vision-based offline RL literature, is shared vs separate encoder the standard?

  **4. Dataset size vs. model capacity — overfitting**
  - ~36 training episodes, unknown length, batch_size=128, 1M steps. What are the expected
    overfitting dynamics? Is there any dropout, weight decay, or data augmentation?
  - The MLP heads have no regularization. The ResNets are fine-tuned end-to-end. What
    regularization would you recommend?
  - Is the train/test split meaningful here? (Train and test are from the same demonstration
    distribution; the "test" eval computes IQL losses, not rollout returns.)

  **5. Reward structure**
  - `normalize_rewards` is off by default. With sparse rewards, what happens to the value
    estimates and the advantage? Will `exp(adv * 0.1)` collapse or explode?
  - The normalization formula is `rewards /= (max_return - min_return); rewards *= 1000`.
    Is scaling to 1000 a good choice? What effect does it have on value targets and the
    temperature hyperparameter?

  **6. Sampling and data pipeline correctness**
  - `sample(batch_size)` does uniform random sampling over all transitions (no prioritized
    replay, no trajectory-level sampling). Is this OK for IQL?
  - Episode boundaries: when sampling a terminal step, `next_obs` at that index is the reset
    observation of the NEXT episode (or garbage). The mask is 0 so `next_v` is multiplied out.
    Is this handled correctly end-to-end?
  - The `preprocess_file` trims to the first `done` signal. Are there any edge cases where
    this loses valid data?

  **7. Force field encoding (full mode)**
  - The force field (420-dim → reshape to 14×10×3 → bilinear upsample to 224×224 → ImageNet
    normalize → ResNet18) seems architecturally odd. Is there a better way to encode a 420-dim
    vector that doesn't waste a full ResNet?
  - The `force_to_image` normalizes per-batch (not per-sample). Could this cause issues in
    a batch where force readings are all near zero?

  **8. Training loop and optimization**
  - No gradient clipping anywhere. With end-to-end ResNet fine-tuning and sparse rewards,
    what could happen?
  - No LR scheduling (fixed 3e-4 for 1M steps). Recommendation?
  - The `eval_on_dataset` function computes IQL losses on the test set, not environment
    returns. Is this a useful evaluation signal? What does decreasing test critic loss
    actually tell you?
  - The backbone gradient norm is tracked only for `critic.q1.encoder.wrist_backbone.net.conv1.weight`.
    Is this a good proxy for overall training health?

  **9. Overall prognosis: will this train well?**
  - Given 40 episodes, sparse rewards, 3 separate ResNets, no regularization, and 1M steps —
    what is your honest assessment of the learning dynamics? Will the policy improve beyond
    behavioral cloning?
  - What are the top 3 changes you would make first to maximize the chance of actually
    learning a useful policy?
  - Are there any outright bugs that would prevent learning (not just inefficiency)?

