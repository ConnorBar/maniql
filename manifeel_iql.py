"""Custom dataset loader: ManiFeel preprocessed pickle -> IQL Dataset interface."""

import collections
import pickle

import numpy as np

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


class ManiFeelDataset:
    """Loads a ManiFeel preprocessed pickle and exposes the same interface
    that the IQL Learner expects (``sample(batch_size) -> Batch``).

    Assumes preprocessing has already cleaned up the sticky done flag
    (one done=True per episode at the end).  ``done`` = task success.

    Args:
        pkl_path: Path to the preprocessed pickle file.
        use_features: Optional list of feature names to select from the
            observation vector (e.g. ``["wrist", "state"]``).  Uses the
            ``obs_layout`` metadata to slice the correct columns.  If
            ``None``, uses the full observation as-is.
        clip_actions: Clip actions to [-1+eps, 1-eps].
        eps: Clipping epsilon.
    """

    def __init__(
        self,
        pkl_path: str,
        use_features: list = None,
        clip_actions: bool = True,
        eps: float = 1e-5,
    ):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.metadata = data.get("metadata", {})
        self.file_index = data.get("file_index", [])

        obs = data["obs"].astype(np.float32)
        next_obs = data["next_obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32).ravel()
        dones = data["dones"].astype(np.float32).ravel()

        if clip_actions:
            lim = 1.0 - eps
            actions = np.clip(actions, -lim, lim)

        obs, next_obs = self._select_features(obs, next_obs, use_features)

        terminals = dones.astype(np.float32)
        masks = (1.0 - terminals).astype(np.float32)

        self.observations = obs
        self.next_observations = next_obs
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones.copy()
        self.terminals = terminals
        self.size = len(obs)

    @classmethod
    def _from_arrays(cls, observations, next_observations, actions, rewards,
                     masks, dones_float, terminals, metadata=None):
        """Construct a dataset directly from arrays (used by train_test_split)."""
        ds = object.__new__(cls)
        ds.observations = observations
        ds.next_observations = next_observations
        ds.actions = actions
        ds.rewards = rewards
        ds.masks = masks
        ds.dones_float = dones_float
        ds.terminals = terminals
        ds.size = len(observations)
        ds.metadata = metadata or {}
        ds.file_index = []
        ds._active_layout = metadata.get("obs_layout", []) if metadata else []
        return ds

    # ----- train / test split ----------------------------------------------

    def train_test_split(self, test_ratio: float = 0.1, seed: int = 42):
        """Split into train/test at the episode level.

        Returns:
            (train_dataset, test_dataset)
        """
        rng = np.random.RandomState(seed)

        ep_ends = np.where(self.dones_float == 1.0)[0]
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

        if ep_start < self.size:
            train_idx.append(np.arange(ep_start, self.size))

        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)

        def _slice(idx):
            return ManiFeelDataset._from_arrays(
                self.observations[idx],
                self.next_observations[idx],
                self.actions[idx],
                self.rewards[idx],
                self.masks[idx],
                self.dones_float[idx],
                self.terminals[idx],
                self.metadata,
            )

        return _slice(train_idx), _slice(test_idx)

    # ----- feature selection -----------------------------------------------

    def _select_features(self, obs, next_obs, use_features):
        """Slice observation columns to keep only the requested features."""
        if use_features is None:
            self._active_layout = self.metadata.get("obs_layout", [])
            return obs, next_obs

        layout = self.metadata.get("obs_layout", [])
        if not layout:
            raise ValueError(
                "use_features requires obs_layout in metadata, but the "
                "preprocessed file has none.  Re-run seed_data.py or pass "
                "use_features=None to use the full observation."
            )

        layout_by_name = {entry["name"]: entry for entry in layout}
        _aliases = {
            "wrist": "wrist",
            "tactile": "right_tactile_camera_taxim",
            "forcefield": "tactile_force_field_right",
            "state": "state",
        }

        col_indices = []
        active_layout = []
        for feat in use_features:
            key = _aliases.get(feat, feat)
            if key not in layout_by_name:
                available = [e["name"] for e in layout]
                raise ValueError(
                    f"Feature '{feat}' (mapped to '{key}') not found in "
                    f"obs_layout. Available: {available}"
                )
            entry = layout_by_name[key]
            col_indices.extend(range(entry["start"], entry["end"]))
            active_layout.append(entry)

        col_indices = np.array(col_indices, dtype=np.intp)
        self._active_layout = active_layout
        return obs[:, col_indices], next_obs[:, col_indices]

    # ----- IQL interface ---------------------------------------------------

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            masks=self.masks[idx],
            next_observations=self.next_observations[idx],
        )

    # ----- diagnostics -----------------------------------------------------

    def validate(self) -> bool:
        """Run sanity checks and print warnings.  Returns True if clean."""
        ok = True

        for name, arr in [
            ("observations", self.observations),
            ("next_observations", self.next_observations),
            ("actions", self.actions),
            ("rewards", self.rewards),
        ]:
            if np.any(np.isnan(arr)):
                print(f"[WARN] NaN detected in {name}")
                ok = False
            if np.any(np.isinf(arr)):
                print(f"[WARN] Inf detected in {name}")
                ok = False

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
        lines = [
            f"ManiFeelDataset: {self.size:,} transitions, {n_eps} episodes",
            f"  obs dim:       {self.observations.shape[1]}",
            f"  action dim:    {self.actions.shape[1]}",
            f"  reward range:  [{self.rewards.min():.4f}, {self.rewards.max():.4f}]",
            f"  reward mean:   {self.rewards.mean():.4f}",
            f"  terminals:     {int(self.terminals.sum())} / {self.size}",
            f"  successes:     {int(self.dones_float.sum())} / {n_eps} episodes",
            f"  masks mean:    {self.masks.mean():.6f}",
        ]
        if self.metadata:
            lines.append(f"  wrist_encoder: {self.metadata.get('wrist_encoder', '?')}")
            full_layout = self.metadata.get("obs_layout", [])
            if full_layout:
                full_parts = [f"{e['name']}({e['dim']})" for e in full_layout]
                lines.append(f"  file layout:   {' + '.join(full_parts)}")
            if hasattr(self, '_active_layout') and self._active_layout:
                active_parts = [f"{e['name']}({e['dim']})" for e in self._active_layout]
                lines.append(f"  active feats:  {' + '.join(active_parts)}")
        return "\n".join(lines)
