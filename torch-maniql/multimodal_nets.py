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
