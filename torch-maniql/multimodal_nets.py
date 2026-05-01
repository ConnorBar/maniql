"""Multi-modal networks + PyTorch IQL learner (end-to-end trainable R3M).

This file replaces the prior JAX/Flax implementation with pure PyTorch.
The dataset (`manifeel_iql.ManiFeelDataset`) returns a Batch where
`observations` and `next_observations` are dicts containing:

- wrist_state: {"wrist": uint8(B,224,224,3), "state": float32(B,7)}
- full: {"wrist": uint8, "tactile": uint8, "force": float32(B,420), "state": float32}

Architecture: a single shared MultiModalEncoder is used across actor, critic,
and value networks.  Only MLP heads are separate.  The target critic uses an
EMA copy of the encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from obs_modality import IMAGE_KEYS
from vision_backbone import (
    RESNET_OUT_DIM,
    ResNetBackbone,
    force_to_image,
    r3m_preprocess_bhwc,
    random_shift_aug,
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


# ---------------------------------------------------------------------------
#  Head-only networks (no encoder — receive pre-encoded features)
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.head = MLP(obs_dim, hidden_dims, 1)

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        return self.head(encoded_obs).squeeze(-1)


class QHead(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int], action_dim: int):
        super().__init__()
        self.head = MLP(obs_dim + action_dim, hidden_dims, 1)

    def forward(self, encoded_obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([encoded_obs, act], dim=-1)).squeeze(-1)


class DoubleQHead(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: Sequence[int], action_dim: int):
        super().__init__()
        self.q1 = QHead(obs_dim, hidden_dims, action_dim)
        self.q2 = QHead(obs_dim, hidden_dims, action_dim)

    def forward(self, encoded_obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(encoded_obs, act), self.q2(encoded_obs, act)


class PolicyHead(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int],
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.trunk = MLP(obs_dim, hidden_dims, hidden_dims[-1], activate_final=True)
        self.mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def _dist(self, encoded_obs: torch.Tensor, temperature: float = 1.0) -> Normal:
        z = self.trunk(encoded_obs)
        mean = torch.tanh(self.mean(z))
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) * float(temperature)
        return Normal(mean, std)

    def log_prob(self, encoded_obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        dist = self._dist(encoded_obs, temperature=1.0)
        return dist.log_prob(act).sum(dim=-1)

    @torch.no_grad()
    def act(self, encoded_obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        dist = self._dist(encoded_obs, temperature=1.0)
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
        augment: bool = False,
        aug_pad: int = 4,
        backbone_lr: float = 1e-5,
        max_steps: int = 1_000_000,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
    ):
        self.device = device
        self.arch = arch
        self.mode = mode
        self.discount = discount
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.augment = augment
        self.aug_pad = aug_pad
        self.max_grad_norm = max_grad_norm

        obs_dim = encoded_obs_dim(arch, mode)

        self.encoder = MultiModalEncoder(arch, mode, r3m_checkpoint=r3m_checkpoint).to(device)
        self.target_encoder = MultiModalEncoder(arch, mode, r3m_checkpoint=r3m_checkpoint).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.actor = PolicyHead(
            obs_dim=obs_dim, hidden_dims=hidden_dims, action_dim=action_dim,
        ).to(device)
        self.critic = DoubleQHead(
            obs_dim=obs_dim, hidden_dims=hidden_dims, action_dim=action_dim,
        ).to(device)
        self.value = ValueHead(obs_dim=obs_dim, hidden_dims=hidden_dims).to(device)
        self.target_critic = DoubleQHead(
            obs_dim=obs_dim, hidden_dims=hidden_dims, action_dim=action_dim,
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=backbone_lr, weight_decay=1e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-4)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr, weight_decay=1e-4)

        def _warmup_cosine(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))

        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, _warmup_cosine)
            for opt in [self.encoder_opt, self.actor_opt, self.critic_opt, self.value_opt]
        ]

    def _to_torch_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            t = torch.as_tensor(v, device=self.device)
            out[k] = t
        return out

    def _augment_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.augment:
            return obs
        out = {}
        for k, v in obs.items():
            if k in IMAGE_KEYS:
                out[k] = random_shift_aug(v, pad=self.aug_pad)
            else:
                out[k] = v
        return out

    def update(self, batch) -> IQLInfo:
        obs = self._augment_obs(self._to_torch_obs(batch.observations))
        next_obs = self._augment_obs(self._to_torch_obs(batch.next_observations))
        act = torch.as_tensor(batch.actions, device=self.device, dtype=torch.float32)
        rew = torch.as_tensor(batch.rewards, device=self.device, dtype=torch.float32)
        msk = torch.as_tensor(batch.masks, device=self.device, dtype=torch.float32)

        # Zero all gradients up front
        self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.value_opt.zero_grad(set_to_none=True)

        # Encode current obs (with gradients for value + critic)
        encoded_obs = self.encoder(obs)

        # Target Q and next-state encoding (no gradients)
        with torch.no_grad():
            encoded_next = self.encoder(next_obs)
            target_encoded = self.target_encoder(obs)
            tq1, tq2 = self.target_critic(target_encoded, act)
            tq = torch.minimum(tq1, tq2)

        # --- value loss ---
        v = self.value(encoded_obs)
        v_loss = expectile_loss(tq - v, self.expectile).mean()

        # --- critic loss ---
        with torch.no_grad():
            next_v = self.value(encoded_next)
            target_q = rew + self.discount * msk * next_v
        q1, q2 = self.critic(encoded_obs, act)
        c_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # Backward value + critic together (both flow through shared encoder)
        (v_loss + c_loss).backward()

        # Track backbone grad norm before clipping
        g = self.encoder.wrist_backbone.net.conv1.weight.grad
        grad_norm = float(g.norm().detach().cpu()) if g is not None else 0.0

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.encoder_opt.step()
        self.value_opt.step()
        self.critic_opt.step()

        # --- actor loss (AWR-style, detached from encoder) ---
        with torch.no_grad():
            adv = tq - v.detach()
            weights = torch.exp(adv * self.temperature).clamp(max=100.0)
        logp = self.actor.log_prob(encoded_obs.detach(), act)
        a_loss = -(weights * logp).mean()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # --- LR scheduling ---
        for sched in self.schedulers:
            sched.step()

        # --- target EMA ---
        with torch.no_grad():
            for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(p.data, alpha=self.tau)
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

        encoded_obs = self.encoder(obs)
        encoded_next = self.encoder(next_obs)
        target_encoded = self.target_encoder(obs)

        tq1, tq2 = self.target_critic(target_encoded, act)
        tq = torch.minimum(tq1, tq2)
        v = self.value(encoded_obs)
        v_loss = expectile_loss(tq - v, self.expectile).mean()

        next_v = self.value(encoded_next)
        target_q = rew + self.discount * msk * next_v
        q1, q2 = self.critic(encoded_obs, act)
        c_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        adv = tq - v
        weights = torch.exp(adv * self.temperature).clamp(max=100.0)
        logp = self.actor.log_prob(encoded_obs, act)
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
        encoded = self.encoder(tobs)
        a = self.actor.act(encoded, deterministic=deterministic)
        return a.detach().cpu().numpy()
