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
