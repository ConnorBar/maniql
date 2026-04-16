"""Multi-modal IQL networks with R3M ResNet backbone finetuning.

Architecture (end-to-end, finetuned during IQL training):
  wrist  (224,224,3 uint8)  -> R3M ResNet  -> feature_dim
  tactile (224,224,3 uint8) -> R3M ResNet  -> feature_dim   (full mode only)
  force  (420 flat)         -> R3M ResNet  -> feature_dim   (full mode only)
  state  (7)                -> passthrough
  Combined -> standard IQL MLP heads (value, critic, actor)

Each IQL head (value, double-critic, actor) has its own encoder copy so
gradients from each loss flow through a dedicated encoder.  The ResNet
backbone is initialised from R3M pretrained weights and finetuned via IQL.

Pipeline modes:
  "wrist_state" : wrist + state
  "full"        : wrist + tactile + force + state

The backbone is model-agnostic -- swap ``arch`` to change from resnet18
to resnet34/50 or replace ``FlaxResNet`` with any module that maps
``(B, 224, 224, 3) -> (B, D)``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "implicit_q_learning"))

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from common import MLP, Batch, InfoDict, Model, PRNGKey, Params, default_init
from critic import update_q, update_v
from actor import update as awr_update_actor
import policy as iql_policy
from vision_backbone import (
    FlaxResNet, RESNET_OUT_DIM,
    r3m_preprocess, force_to_image,
    load_r3m_to_flax, load_r3m_checkpoint, inject_r3m_weights,
)

tfd = tfp.distributions
tfb = tfp.bijectors

# ---------------------------------------------------------------------------
#  Observation constants
# ---------------------------------------------------------------------------
STATE_DIM = 7
FORCE_DIM = 420


def encoded_obs_dim(arch: str, mode: str) -> int:
    """Total feature vector width after encoding all modalities."""
    feat = RESNET_OUT_DIM[arch]
    if mode == "wrist_state":
        return feat + STATE_DIM
    # full mode: wrist + tactile + force (all through ResNet) + state
    return feat * 3 + STATE_DIM


# ---------------------------------------------------------------------------
#  Multi-modal observation encoder
# ---------------------------------------------------------------------------

class MultiModalEncoder(nn.Module):
    """Encodes raw observations into a flat feature vector.

    Each image-like modality gets its own ``FlaxResNet`` (separate params,
    independent gradient flow).  State is passed through as-is.

    Expects a dict observation with keys matching the pipeline mode:
      wrist_state : ``{"wrist": (B,224,224,3), "state": (B,7)}``
      full        : ``{"wrist": …, "tactile": …, "force": (B,420), "state": …}``
    """
    arch: str = "resnet18"
    mode: str = "wrist_state"

    @nn.compact
    def __call__(self, observations: dict) -> jnp.ndarray:
        wrist = r3m_preprocess(observations["wrist"])
        wrist_feat = FlaxResNet(self.arch, name="wrist_backbone")(wrist)

        parts = [wrist_feat]

        if self.mode == "full":
            tact = r3m_preprocess(observations["tactile"])
            tact_feat = FlaxResNet(self.arch, name="tactile_backbone")(tact)
            parts.append(tact_feat)

            force_img = force_to_image(observations["force"])
            force_feat = FlaxResNet(self.arch, name="force_backbone")(force_img)
            parts.append(force_feat)

        parts.append(observations["state"].astype(jnp.float32))
        return jnp.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
#  IQL network heads
# ---------------------------------------------------------------------------

class MMValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    arch: str = "resnet18"
    mode: str = "wrist_state"

    @nn.compact
    def __call__(self, observations) -> jnp.ndarray:
        encoded = MultiModalEncoder(self.arch, self.mode,
                                    name="encoder")(observations)
        critic = MLP((*self.hidden_dims, 1))(encoded)
        return jnp.squeeze(critic, -1)


class MMCritic(nn.Module):
    hidden_dims: Sequence[int]
    arch: str = "resnet18"
    mode: str = "wrist_state"
    activations: callable = nn.relu

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray) -> jnp.ndarray:
        encoded = MultiModalEncoder(self.arch, self.mode,
                                    name="encoder")(observations)
        inputs = jnp.concatenate([encoded, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class MMDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    arch: str = "resnet18"
    mode: str = "wrist_state"
    activations: callable = nn.relu

    @nn.compact
    def __call__(self, observations,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        c1 = MMCritic(self.hidden_dims, self.arch, self.mode,
                      activations=self.activations)(observations, actions)
        c2 = MMCritic(self.hidden_dims, self.arch, self.mode,
                      activations=self.activations)(observations, actions)
        return c1, c2


class MMNormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    arch: str = "resnet18"
    mode: str = "wrist_state"
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self, observations,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        encoded = MultiModalEncoder(self.arch, self.mode,
                                    name="encoder")(observations)

        outputs = MLP(self.hidden_dims, activate_final=True,
                      dropout_rate=self.dropout_rate)(encoded,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.log_std_scale))(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros,
                                  (self.action_dim,))

        log_std_min = self.log_std_min or -10.0
        log_std_max = self.log_std_max or 2.0
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(
                distribution=base_dist, bijector=tfb.Tanh())
        return base_dist


# ---------------------------------------------------------------------------
#  JIT-compiled update step (from implicit_q_learning/learner.py)
# ---------------------------------------------------------------------------

def _mm_target_update(critic: Model, target_critic: Model,
                      tau: float) -> Model:
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau),
        critic.params, target_critic.params)
    return target_critic.replace(params=new_target_params)


@jax.jit
def _mm_update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(
        key, actor, target_critic, new_value, batch, temperature)
    new_critic, critic_info = update_q(critic, new_value, batch, discount)
    new_target_critic = _mm_target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info, **value_info, **actor_info,
    }


# ---------------------------------------------------------------------------
#  Multi-modal Learner
# ---------------------------------------------------------------------------

class MultiModalLearner:
    """IQL Learner with per-head ResNet observation encoders.

    On construction the vision backbones are initialised from R3M pretrained
    weights (if ``r3m_checkpoint`` is provided).  During training the
    backbone is finetuned end-to-end via IQL loss gradients.

    Exposes the same ``update(batch)`` / ``sample_actions(obs)`` API as
    ``learner.Learner``.
    """

    def __init__(
        self,
        seed: int,
        observations,
        actions: jnp.ndarray,
        arch: str = "resnet18",
        mode: str = "wrist_state",
        r3m_checkpoint: Optional[str] = None,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
    ):
        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]

        # --- load R3M pretrained backbone params (once, requires torch) ---
        r3m_params = None
        if r3m_checkpoint is not None:
            print(f"[INFO] Loading R3M weights from {r3m_checkpoint} "
                  f"(arch={arch})")
            pt_sd = load_r3m_checkpoint(r3m_checkpoint)
            r3m_params = load_r3m_to_flax(pt_sd, arch=arch)

        # --- actor ---
        actor_def = MMNormalTanhPolicy(
            hidden_dims, action_dim, arch=arch, mode=mode,
            log_std_scale=1e-3, log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=False,
        )
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations], tx=optimiser)

        # --- critic ---
        critic_def = MMDoubleCritic(hidden_dims, arch=arch, mode=mode)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        # --- value ---
        value_def = MMValueCritic(hidden_dims, arch=arch, mode=mode)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        # --- target critic (no optimiser, soft-updated) ---
        target_critic = Model.create(critic_def,
                                     inputs=[critic_key, observations, actions])

        # --- inject R3M pretrained weights into all backbones ---
        if r3m_params is not None:
            actor = actor.replace(
                params=inject_r3m_weights(actor.params, r3m_params))
            critic = critic.replace(
                params=inject_r3m_weights(critic.params, r3m_params))
            value = value.replace(
                params=inject_r3m_weights(value.params, r3m_params))
            target_critic = target_critic.replace(
                params=inject_r3m_weights(target_critic.params, r3m_params))
            print("[INFO] R3M backbone weights injected into all heads.")

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self, observations,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = iql_policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params,
            observations, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        (new_rng, new_actor, new_critic, new_value,
         new_target_critic, info) = _mm_update_jit(
            self.rng, self.actor, self.critic, self.value,
            self.target_critic, batch, self.discount, self.tau,
            self.expectile, self.temperature)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic
        return info
