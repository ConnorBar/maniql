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
    p.add_argument("--normalize-actions", action="store_true", default=True,
                   help="Normalize actions to [-1, 1] per dimension using "
                        "max absolute value from the data.")
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


def normalize_actions(actions: np.ndarray) -> np.ndarray:
    """Normalize each action dimension to [-1, 1] by its max absolute value.

    Dimensions that are already within [-1, 1] are left untouched.
    Dimensions that are constant (std=0) are left as-is.
    """
    n_dims = actions.shape[1]
    scales = np.ones(n_dims, dtype=np.float32)
    for d in range(n_dims):
        max_abs = np.max(np.abs(actions[:, d]))
        if max_abs > 1.0:
            scales[d] = max_abs
    print(f"[INFO] Action normalization scales: {scales.tolist()}")
    for d in range(n_dims):
        if scales[d] > 1.0:
            print(f"  dim {d}: range [{actions[:, d].min():.4f}, "
                  f"{actions[:, d].max():.4f}] -> "
                  f"[{actions[:, d].min() / scales[d]:.4f}, "
                  f"{actions[:, d].max() / scales[d]:.4f}]")
    return actions / scales[None, :]


def merge_chunks(chunk_dir: Path, output_path: Path, metadata: dict,
                 split_keys, normalize_acts: bool = False):
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

    all_actions = np.concatenate(scalar_arrs["actions"])
    if normalize_acts:
        all_actions = normalize_actions(all_actions)
        metadata["action_normalization"] = "per_dim_max_abs"

    final = {
        "metadata": metadata,
        "file_index": file_index,
        "obs": {k: np.concatenate(obs_arrs[k]) for k in split_keys},
        "next_obs": {k: np.concatenate(nobs_arrs[k]) for k in split_keys},
        "actions": all_actions.astype(np.float32),
        **{sk: np.concatenate(scalar_arrs[sk]) for sk in scalar_arrs
           if sk != "actions"},
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
    merge_chunks(chunk_dir, output_path, metadata, split_keys,
                 normalize_acts=args.normalize_actions)

    if not args.keep_chunks:
        shutil.rmtree(chunk_dir, ignore_errors=True)

    print(f"[DONE] {output_path}  ({total_transitions:,} transitions, "
          f"mode={mode})")


if __name__ == "__main__":
    main()
