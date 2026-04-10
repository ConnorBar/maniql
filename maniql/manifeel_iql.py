"""Custom dataset loader: ManiFeel preprocessed pickle -> IQL Dataset interface.

Expects split-storage pickles where ``obs`` and ``next_obs`` are dicts with
keys ``pass`` (wrist+state), ``tact`` (tactile image volume), ``forcefield``.
"""

import collections
import pickle

import numpy as np

from obs_modality import SPLIT_KEYS

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


def _coerce_nested_split(data: dict) -> dict:
    """Inline legacy top-level *_{passthrough,tactile,...} into ``obs`` / ``next_obs`` dicts."""
    if isinstance(data.get("obs"), dict) and SPLIT_KEYS[0] in data["obs"]:
        return data
    if "obs_passthrough" not in data:
        return data
    k0, k1, k2 = SPLIT_KEYS
    out = {**data}
    out["obs"] = {
        k0: data["obs_passthrough"],
        k1: data["obs_tactile"],
        k2: data["obs_forcefield"],
    }
    out["next_obs"] = {
        k0: data["next_obs_passthrough"],
        k1: data["next_obs_tactile"],
        k2: data["next_obs_forcefield"],
    }
    return out


class ManiFeelDataset:
    """Loads a ManiFeel preprocessed pickle (split-storage) and exposes the
    ``sample(batch_size) -> Batch`` interface that IQL expects.

    Args:
        pkl_path: Path to the preprocessed pickle file.
        clip_actions: Clip actions to [-1+eps, 1-eps].
        eps: Clipping epsilon.
    """

    def __init__(
        self,
        pkl_path: str,
        clip_actions: bool = True,
        eps: float = 1e-5,
    ):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        data = _coerce_nested_split(data)
        self.metadata = data.get("metadata", {})
        self.file_index = data.get("file_index", [])

        if not (isinstance(data.get("obs"), dict) and SPLIT_KEYS[0] in data["obs"]):
            raise ValueError(
                "Expected split-storage pickle with obs dict keys "
                f"{SPLIT_KEYS}, got flat obs array or missing keys.  "
                "Re-run seed_data.py with all four features."
            )

        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32).ravel()
        dones = data["dones"].astype(np.float32).ravel()

        if clip_actions:
            lim = 1.0 - eps
            actions = np.clip(actions, -lim, lim)

        self._obs = {k: data["obs"][k].astype(np.float32) for k in SPLIT_KEYS}
        self._next_obs = {k: data["next_obs"][k].astype(np.float32) for k in SPLIT_KEYS}
        self._indices = None  # optional indirection for train/test splits (avoid big copies)
        self.size = len(self._obs[SPLIT_KEYS[0]])

        self.actions = actions
        self.rewards = rewards
        terminals = dones.astype(np.float32)
        self.masks = (1.0 - terminals).astype(np.float32)
        self.dones_float = dones.copy()
        self.terminals = terminals

    def observation_example(self):
        """One transition as a pytree with leading batch dim 1 (for model init)."""
        if self._indices is None:
            return {k: self._obs[k][:1] for k in SPLIT_KEYS}
        base_i = int(self._indices[0])
        return {k: self._obs[k][base_i:base_i + 1] for k in SPLIT_KEYS}

    def _pack_obs_batch(self, idx: np.ndarray):
        if self._indices is not None:
            idx = self._indices[idx]
        return {k: self._obs[k][idx] for k in SPLIT_KEYS}

    def _pack_next_batch(self, idx: np.ndarray):
        if self._indices is not None:
            idx = self._indices[idx]
        return {k: self._next_obs[k][idx] for k in SPLIT_KEYS}

    @classmethod
    def _from_split_dicts(
        cls,
        obs: dict,
        next_obs: dict,
        actions,
        rewards,
        masks,
        dones_float,
        terminals,
        metadata=None,
        indices=None,
    ):
        ds = object.__new__(cls)
        ds.metadata = metadata or {}
        ds.file_index = []
        ds._obs = {k: obs[k] for k in SPLIT_KEYS}
        ds._next_obs = {k: next_obs[k] for k in SPLIT_KEYS}
        ds.actions = actions
        ds.rewards = rewards
        ds.masks = masks
        ds.dones_float = dones_float
        ds.terminals = terminals
        ds._indices = indices
        ds.size = int(len(indices)) if indices is not None else len(ds._obs[SPLIT_KEYS[0]])
        return ds

    def train_test_split(self, test_ratio: float = 0.1, seed: int = 42):
        rng = np.random.RandomState(seed)

        # IMPORTANT: Use index-based split so we don't duplicate multi-GB tactile arrays.
        # We always split in the base dataset index space.
        if self._indices is not None:
            base_done = self.dones_float[self._indices]
            ep_ends_local = np.where(base_done == 1.0)[0]
            base_indices = self._indices
        else:
            ep_ends_local = np.where(self.dones_float == 1.0)[0]
            base_indices = None

        ep_ends = ep_ends_local
        n_eps = len(ep_ends)
        n_test = max(1, int(n_eps * test_ratio))
        ep_order = rng.permutation(n_eps)
        test_ep_set = set(ep_order[:n_test].tolist())

        train_idx, test_idx = [], []
        ep_start = 0
        for ep_i, ep_end in enumerate(ep_ends):
            indices = np.arange(ep_start, ep_end + 1)
            if ep_i in test_ep_set:
                test_idx.append(indices)
            else:
                train_idx.append(indices)
            ep_start = ep_end + 1

        if ep_start < (len(ep_ends_local) and (ep_ends_local[-1] + 1) or 0):
            # unreachable, but keep structure
            pass
        # If the last episode doesn't end with done=True, include tail as train.
        local_size = int(len(self._indices)) if self._indices is not None else int(self.size)
        if ep_start < local_size:
            train_idx.append(np.arange(ep_start, local_size))

        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)

        if base_indices is not None:
            train_base = base_indices[train_idx]
            test_base = base_indices[test_idx]
        else:
            train_base = train_idx
            test_base = test_idx

        def _make_indexed(indices_base: np.ndarray):
            return ManiFeelDataset._from_split_dicts(
                self._obs,
                self._next_obs,
                self.actions,
                self.rewards,
                self.masks,
                self.dones_float,
                self.terminals,
                self.metadata,
                indices=indices_base.astype(np.int64, copy=False),
            )

        return _make_indexed(train_base), _make_indexed(test_base)

    # ----- IQL interface ---------------------------------------------------

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self._pack_obs_batch(idx),
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            masks=self.masks[idx],
            next_observations=self._pack_next_batch(idx),
        )

    # ----- diagnostics -----------------------------------------------------

    def validate(self) -> bool:
        """Run sanity checks and print warnings.  Returns True if clean."""
        def _finite_check(name: str, arr: np.ndarray, batch: int = 256) -> bool:
            """Check NaN/Inf without allocating full-size masks.

            Large modalities (e.g. tactile volumes) can be multi-GB; calling
            np.isnan(arr) materializes a full boolean array and may OOM.
            We instead scan along the first dimension in chunks.
            """
            nonlocal ok
            a = np.asarray(arr)
            n0 = int(a.shape[0]) if a.ndim > 0 else 1
            if a.ndim == 0:
                if np.isnan(a):
                    print(f"[WARN] NaN detected in {name}")
                    ok = False
                if np.isinf(a):
                    print(f"[WARN] Inf detected in {name}")
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

        ok = True
        to_check = (
            [(f"obs.{k}", self._obs[k]) for k in SPLIT_KEYS]
            + [(f"next_obs.{k}", self._next_obs[k]) for k in SPLIT_KEYS]
            + [("actions", self.actions), ("rewards", self.rewards)]
        )

        for name, arr in to_check:
            _finite_check(name, arr)

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
        ep_lens = np.array(ep_lengths)

        print(
            f"[INFO] {self.size:,} transitions, {n_eps} episodes, "
            f"ep_len: mean={ep_lens.mean():.0f} median={np.median(ep_lens):.0f} "
            f"min={ep_lens.min()} max={ep_lens.max()}"
        )
        if ok:
            print("[OK] No data integrity issues found.")
        return ok

    def summary(self) -> str:
        n_eps = int(self.dones_float.sum())
        k0, k1, k2 = SPLIT_KEYS
        ptp_d = int(self._obs[k0].shape[1])
        tac_shp = self._obs[k1].shape[1:]
        ff_d = int(self._obs[k2].shape[1])
        obs_line = (
            f"  obs (split):   {k0}({ptp_d}) + {k1}{tac_shp} + {k2}({ff_d})"
        )

        lines = [
            f"ManiFeelDataset: {self.size:,} transitions, {n_eps} episodes",
            obs_line,
            f"  action dim:    {self.actions.shape[1]}",
            f"  reward range:  [{self.rewards.min():.4f}, {self.rewards.max():.4f}]",
            f"  reward mean:   {self.rewards.mean():.4f}",
            f"  terminals:     {int(self.terminals.sum())} / {self.size}",
            f"  successes:     {int(self.dones_float.sum())} / {n_eps} episodes",
            f"  masks mean:    {self.masks.mean():.6f}",
        ]
        if self.metadata:
            lines.append(f"  wrist_encoder: {self.metadata.get('wrist_encoder', '?')}")
        return "\n".join(lines)
