# Holistic IQL Pipeline Review — Prompt for Claude Opus

You are doing a deep technical audit of a complete offline IQL (Implicit Q-Learning) pipeline
for robot manipulation. The repo is `torch-maniql`. Below is the full source of every relevant
file, followed by concrete dataset statistics, known issues already discovered, and specific
questions. Please give an honest, thorough assessment — including things that would cause silent
failure, not just obvious bugs.

---

## Project context

- **Task**: Franka robot manipulation (TacSL / ManiFeel lab setup). The specific task is the
  **Bulb task** (`TacSLTaskBulb`, `numActions: 7`), a precision insertion/screw task.
- **Goal**: Train an offline IQL policy from 40 human teleoperation demonstrations, then roll
  out in IsaacGym simulation at the lab.
- **Pipeline**:
  1. `seed_data.py` — raw `*_transitions.pkl` → preprocessed pickle (images as uint8, no R3M)
  2. `train_iql.py` — offline IQL with end-to-end R3M ResNet18 fine-tuning
  3. `rollout_watch_isaac.py` — loads actor checkpoint, runs in IsaacGym (lab only)
- **Mode in use**: `wrist_state` (wrist camera 224×224×3 uint8 + 7-dim joint state)

---

## Actual dataset statistics

Measured directly from `data/preprocessed/raw_wrist_state.pkl`:

```
Total transitions : 31,797
Episodes          : 40
Episode lengths   : mean=794.9  median=790  min=556  max=1224
Reward stats      : mean=-0.0206  std=0.0316  min=-0.1997  max=-0.0039
                    100% nonzero — dense shaped reward (negative distance-to-goal)
Episode returns   : mean=-16.38  std=2.94  min=-23.17  max=-11.47
Success flag      : 0/40 episodes — see Known Issue #1 below
Timeouts          : 0/40 episodes

Actions shape     : (31797, 7)
  dim 0: mean= 0.023  std=0.180  min=-1.299  max= 1.500
  dim 1: mean= 0.004  std=0.108  min=-1.071  max= 1.080
  dim 2: mean=-0.060  std=0.172  min=-1.500  max= 1.376
  dim 3: mean= 0.000  std=0.000  min= 0.000  max= 0.000  ← ALWAYS ZERO
  dim 4: mean= 0.000  std=0.000  min= 0.000  max= 0.000  ← ALWAYS ZERO
  dim 5: mean= 0.016  std=2.739  min=-8.250  max= 8.250  ← WRONG SCALE (see Known Issue #2)
  dim 6: mean= 0.024  std=0.011  min= 0.000  max= 0.035  ← likely gripper in meters

State shape       : (31797, 7)  range [-0.92, 1.00]
Wrist shape       : (31797, 224, 224, 3)  dtype=uint8
Train/test split  : ~36 episodes train / ~4 episodes test (episode-level, 10% test ratio)
```

---

## Known issues already discovered (do not re-derive, just factor into your analysis)

### Known Issue #1 — success flag is always zero
`success=0` for all 31,797 transitions across all 40 episodes. This is a data collection bug:
the teleoperation recording system never set `success=True`. It is **not** a transform bug in
`seed_data.py`. The episodes ARE successful demonstrations — the dense negative-distance reward
converges toward 0 by end of episode (e.g., reward goes from -0.12 at step 0 to -0.004 at the
terminal step), confirming the robot reached the goal. IQL should still be able to learn from
the dense reward. The `success` field in the dataset is effectively dead weight.

### Known Issue #2 — action space mismatch (CRITICAL)
The IsaacGym sim expects **normalized [-1, 1] actions**. This is confirmed by:
- Code comment in `tacsl_task_insertion.py:202`: `# values = [-1, 1]`
- The sim scales actions internally via `_apply_actions_as_ctrl_targets(do_scale=True)`:
  - `pos_action_scale: [0.01, 0.01, 0.01]` → dims 0-2 become ≤ 1 cm/step
  - `rot_action_scale: [0.05, 0.05, 0.05]` → dims 3-5 become ≤ 0.05 rad/step
- `rollout_watch_isaac.py` passes `actor.act()` output directly to `env.step()` with no
  de-normalization, confirming the policy output goes straight into the sim unscaled.

The **raw demo actions were recorded in physical/raw units**, not normalized:
- Dim 5 is ±8.25 (likely raw wrist rotation from the SpaceMouse/teleop), not in [-1, 1].
  To normalize: `8.25 / rot_action_scale = 8.25 / 0.05 = 165` — 165× over the expected range.
- Dim 6 is [0, 0.035] — likely gripper finger width in meters, also not normalized.
- Dims 3 & 4 are always zero — plausible if roll/pitch axes weren't used in the task.
- Dims 0–2 are roughly in [-1, 1] (range ~[-1.5, 1.5]) — possibly correct or close.

**Effect of `clip_actions=True`** (default): clips dim 5 from ±8.25 to ±(1-ε). This means every
training sample for dim 5 is ≈±1 (fully saturated), destroying all magnitude variation in the
rotation action. The policy is trained on corrupted action labels for its most dynamic dimension.

**Required fix in `seed_data.py`**: divide raw actions by their corresponding action scale before
storing, then clip to [-1, 1]:
```python
action_scales = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, ???])  # 7th dim TBD
actions = np.clip(raw_actions / action_scales, -1.0, 1.0)
```
The scale for dim 6 (gripper) needs lab confirmation.

### Known Issue #3 — trailing done tail in raw files
Raw pkl files have multiple consecutive `done=1` transitions at the end (e.g., last 5 steps all
`done=1`). `seed_data.py` correctly trims to the first done. This is working as intended.

---

## Hyperparameters in use

```
discount     = 0.99
tau          = 0.005      (target critic EMA)
expectile    = 0.8
temperature  = 0.1
hidden_dims  = [256, 256]
batch_size   = 128
max_steps    = 1_000_000
actor_lr     = critic_lr = value_lr = 3e-4
backbone     = resnet18   (no R3M checkpoint by default)
clip_actions = True       (default — see Known Issue #2)
normalize_rewards = False (default)
test_ratio   = 0.1
```

---

## Full source files

### `obs_modality.py`
```python
"""Observation modality constants for the two pipeline modes."""

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
```

### `vision_backbone.py`
```python
"""PyTorch-native vision utilities + R3M-backed ResNet feature extractor."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RESNET_OUT_DIM: Dict[str, int] = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

FORCE_FIELD_DIM = 420
FORCE_GRID_SHAPE = (14, 10, 3)


def r3m_preprocess_bhwc(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected (B,H,W,3) images, got {tuple(images.shape)}")
    x = images
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        if x.dtype != torch.float32:
            x = x.float()
    x = x.permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def force_to_image(force: torch.Tensor) -> torch.Tensor:
    """(B,420) -> (B,224,224,3) float (unnormalized)."""
    if force.ndim != 2 or force.shape[-1] != FORCE_FIELD_DIM:
        raise ValueError(f"Expected (B,{FORCE_FIELD_DIM}) force, got {tuple(force.shape)}")
    b = force.shape[0]
    x = force.view(b, *FORCE_GRID_SHAPE)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    x = (x - mean) / std
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
    def __init__(self, arch: str = "resnet18", r3m_checkpoint: str | None = None):
        super().__init__()
        if arch not in RESNET_OUT_DIM:
            raise ValueError(f"Unknown arch {arch!r}")
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

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        return self.net(x_bchw)
```

### `multimodal_nets.py`
```python
"""Multi-modal networks + PyTorch IQL learner (end-to-end trainable R3M)."""

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
            tact_feat = self.tactile_backbone(tactile)
            parts.append(tact_feat)
            force_img = force_to_image(obs["force"])
            force_norm = r3m_preprocess_bhwc(force_img)
            force_feat = self.force_backbone(force_norm)
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
        return {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}

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
            "actor_loss": float(a_loss), "critic_loss": float(c_loss),
            "value_loss": float(v_loss), "q": float(tq.mean()),
            "v": float(v.mean()), "adv": float(adv.mean()),
        }

    @torch.no_grad()
    def sample_actions(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        tobs = self._to_torch_obs(obs)
        return self.actor.act(tobs, deterministic=deterministic).detach().cpu().numpy()
```

### `manifeel_iql.py` (dataset loader)
```python
"""Dataset loader: preprocessed pickle -> IQL Batch interface."""

import collections
import pickle
import sys

import numpy as np

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)

from obs_modality import IMAGE_KEYS, get_split_keys

Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "masks", "next_observations"]
)


class ManiFeelDataset:
    def __init__(self, pkl_path: str, clip_actions: bool = True, eps: float = 1e-5):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.metadata = data.get("metadata", {})
        self.file_index = data.get("file_index", [])

        mode = self.metadata.get("mode")
        if mode is None:
            raise ValueError("Pickle is missing metadata['mode'].")
        self._mode = mode
        self._split_keys = get_split_keys(mode)

        for k in self._split_keys:
            if k not in data.get("obs", {}):
                raise ValueError(f"Expected obs key {k!r} for mode={mode!r} but not found.")

        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32).ravel()
        dones = data["dones"].astype(np.float32).ravel()

        if clip_actions:
            lim = 1.0 - eps
            actions = np.clip(actions, -lim, lim)

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
        ds.size = (int(len(indices)) if indices is not None else len(obs[split_keys[0]]))
        return ds

    def train_test_split(self, test_ratio: float = 0.1, seed: int = 42):
        rng = np.random.RandomState(seed)
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

        local_size = (int(len(self._indices)) if self._indices is not None else self.size)
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

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self._pack_obs(idx),
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            masks=self.masks[idx],
            next_observations=self._pack_next(idx),
        )

    def validate(self) -> bool:
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
                    print(f"[WARN] NaN in {name}")
                    ok = False
                    return False
                if np.isinf(chunk).any():
                    print(f"[WARN] Inf in {name}")
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
            f"ManiFeelDataset [{self._mode}]: {self.size:,} transitions, {n_eps} episodes",
            obs_line,
            f"  action dim:    {self.actions.shape[1]}",
            f"  reward range:  [{self.rewards.min():.4f}, {self.rewards.max():.4f}]",
            f"  reward mean:   {self.rewards.mean():.4f}",
            f"  terminals:     {int(self.terminals.sum())} / {self.size}",
        ]
        return "\n".join(lines)
```

### `train_iql.py`
```python
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
def eval_on_dataset(agent, dataset, batch_size, n_batches=10):
    actor_losses, critic_losses, value_losses = [], [], []
    q_means, v_means, adv_means = [], [], []
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(args.save_dir, level=args.log_level)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ds = ManiFeelDataset(args.dataset_path, clip_actions=args.clip_actions)
    mode = full_ds.mode

    train_ds, test_ds = full_ds.train_test_split(test_ratio=args.test_ratio, seed=args.seed)
    if args.validate:
        train_ds.validate()
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
        arch=args.backbone,
        mode=mode,
        r3m_checkpoint=args.r3m_checkpoint or None,
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
    )

    for step in range(1, args.max_steps + 1):
        batch = train_ds.sample(args.batch_size)
        info = agent.update(batch)
        # (logging and checkpointing omitted for brevity)
```

---

## Specific questions for Opus

### 1. Action space mismatch — severity and fix
Given Known Issue #2: training proceeds with `clip_actions=True`, so dim 5 (raw ±8.25, clipped
to ±1) and dims 3&4 (always 0) are used as action targets. The policy uses a Gaussian with
`tanh(mean)` output bounded to (-1, 1).
- How severely does the clipped/wrong dim 5 corrupt IQL training? Will the policy learn *anything*
  useful for that dimension, or will it converge to a degenerate solution (always predict ±1)?
- Does training on always-zero dims 3&4 hurt, or does the policy just learn to output ~0 for
  those (which happens to be the correct behavior at rollout)?
- Proposed fix: divide raw actions by action_scales in `seed_data.py` before storing. Any
  concerns or edge cases with that approach?

### 2. IQL algorithm correctness
- Is the update order (value → actor → critic → target EMA) correct per the IQL paper
  (Kostrikov et al. 2021)? The paper's pseudocode has a specific ordering.
- The actor update re-runs `self.value(obs)` inside a `torch.no_grad()` block *after* the
  value network was already updated this step. The value used for advantage = the post-update
  value, not the pre-update. Is this a meaningful inconsistency?
- The value loss is `expectile_loss(tq - v, expectile=0.8).mean()`. The expectile loss puts
  weight `τ=0.8` on positive residuals (tq > v) and `1-τ=0.2` on negative. Is this the right
  direction — does this correctly push V toward the 80th percentile of Q?
- AWR weights: `exp(adv * 0.1).clamp(max=100)`. With rewards in [-0.20, -0.004] and
  discount=0.99, what will the typical scale of `adv` be? Will most weights saturate at 100,
  or collapse near 1? Is temperature=0.1 a sensible default for this reward scale?

### 3. Separate encoders — no parameter sharing
Each of the three networks (actor, critic, value) has its own `MultiModalEncoder` (its own
ResNet18). No parameter sharing anywhere.
- The advantage estimate is `Q(s,a) - V(s)` where Q and V are encoded by *different* ResNets.
  Does this inconsistency in the latent space undermine the advantage signal?
- In vision-based offline RL (e.g., DrQ, TD3+BC with encoders), what is the standard practice?
  Is shared-encoder between at least value and critic the norm?
- Memory/compute: 3 ResNet18s = ~33M params × 3 = ~99M params just for encoders. Any
  practical concern for training stability (optimizer momentum accumulation, learning dynamics)?

### 4. Masking — dones vs terminals
`manifeel_iql.py` computes `masks = 1 - dones`. The preprocessed pickle also has a `terminals`
field defined as `clip(dones - timeouts, 0, 1)`. In this dataset, timeouts=0 for all episodes,
so `dones == terminals` and the distinction doesn't matter here. However:
- Is `masks = 1 - dones` (rather than `1 - terminals`) architecturally correct for the general
  case? How does treating timeouts as terminal affect Bellman backups?

### 5. Overfitting and regularization
- ~28,660 training transitions (36 eps × ~797 steps), batch=128, 1M gradient steps ≈ 3,500
  epochs through the training data. What are the expected overfitting dynamics for both the
  ResNet encoder and the MLP heads?
- The MLP has no dropout. The ResNet has no weight decay. Is L2 regularization via Adam's
  weight decay the right lever here, or something else?
- Would random crop / color jitter augmentation on the wrist images be worth adding? Any
  concern that augmentation breaks the temporal coherence that IQL relies on?
- The test evaluation computes IQL losses, not rollout returns. Is decreasing test critic loss
  a meaningful overfitting signal? What would you actually watch to detect overfitting?

### 6. Reward scale and normalization
- With rewards in [-0.20, -0.004] and episode returns in [-23, -11], the default of
  `normalize_rewards=False` means value targets will be in roughly [-20, -5] range.
- The `normalize_rewards` function scales to range 1000 (i.e., `ret / (max - min) * 1000`).
  Given the return std is only 2.9, this results in very high-magnitude targets. Is 1000 a
  sensible scale factor? What scale would you recommend?
- Without normalization, will the advantage `tq - v` be well-behaved with temperature=0.1?
  Specifically: if |adv| is routinely > 10, then `exp(10 * 0.1) = e^1 ≈ 2.7` which is fine.
  But if |adv| is routinely > 46, weights hit the clamp of 100. Is that likely here?

### 7. Gradient clipping and learning rate
- No gradient clipping anywhere. With end-to-end ResNet fine-tuning and 3 separate Adam
  optimizers, what can go wrong?
- Fixed lr=3e-4 for 1M steps. Recommendation for this dataset size and architecture?
- The backbone gradient norm is tracked only for `critic.q1.encoder.wrist_backbone.net.conv1.weight`.
  Is this a useful proxy, and what values indicate healthy vs. pathological training?

### 8. Actor policy design
- The policy uses `Normal(tanh(mean), exp(log_std))` and evaluates `log_prob(act)` directly
  on clipped actions in [-1, 1]. There is no tanh change-of-variables correction (as in SAC).
  For IQL/AWR this is intentional (no reparameterization needed) — but is there a numerical
  issue when `act ≈ ±1` and `tanh(mean) ≈ ±1`? The Gaussian can assign reasonable probability
  to values near ±1, but can the gradient flow cleanly?
- `log_std` is a global parameter (not input-conditioned). For a manipulation task with very
  different phases (approach vs. fine insertion), is a state-independent std a significant
  limitation?

### 9. Force field encoding (full mode, for reference)
- Force field: 420-dim → reshape (14,10,3) → `force_to_image` normalizes per-batch-item → 
  bilinear upsample to (224,224,3) → ImageNet normalize → ResNet18. The normalization in
  `force_to_image` is per-batch-item (mean/std over all spatial positions and channels for that
  item), then `r3m_preprocess_bhwc` applies ImageNet statistics on top.
  - Is this double normalization (per-item std → then ImageNet mean/std subtract) coherent?
  - Is running a full ResNet18 on a 420-number signal (upsampled to 224×224) computationally
    and representationally sensible? What alternative architecture would you suggest?

### 10. Overall prognosis
Given everything above — 40 episodes, dense negative-distance reward, action mismatch on dim 5,
no regularization, 3 separate ResNets, 3,500 effective epochs — give your honest assessment:
- Will this pipeline learn anything useful at all (beyond memorizing the training demos)?
- What is the **single most important fix** before running another training run?
- List the top 5 changes in priority order for maximizing the chance of a deployable policy.
- Are there any outright bugs (not just inefficiencies) that would cause training to produce a
  numerically incorrect result or silently wrong behavior?
