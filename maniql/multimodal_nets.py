"""Multi-modal IQL networks: CNN for tactile, MLP for forcefield, trained end-to-end.

Architecture:
  R3M wrist (2048)       -> passthrough
  tactile (160,120,3)    -> CNN encoder -> 256
  forcefield (420 flat)  -> MLP encoder -> 128
  state (7)              -> passthrough
  Combined (2439-dim)    -> standard IQL MLP heads (value, critic, actor)

Each IQL head (value, double-critic, actor) has its own encoder copy so
gradients from each loss flow through a dedicated encoder.  This is the
standard per-head approach and avoids gradient interference.
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

tfd = tfp.distributions
tfb = tfp.bijectors

# ---------------------------------------------------------------------------
#  Observation layout (matches seed_data.py output order)
# ---------------------------------------------------------------------------
WRIST_DIM = 2048
TACTILE_FLAT_DIM = 57_600          # 160 * 120 * 3
TACTILE_SHAPE = (160, 120, 3)
FORCEFIELD_FLAT_DIM = 420          # 10 * 14 * 3
STATE_DIM = 7

TACTILE_ENC_DIM = 256
FORCEFIELD_ENC_DIM = 128

FULL_OBS_DIM = WRIST_DIM + TACTILE_FLAT_DIM + FORCEFIELD_FLAT_DIM + STATE_DIM
ENCODED_DIM = WRIST_DIM + TACTILE_ENC_DIM + FORCEFIELD_ENC_DIM + STATE_DIM


# ---------------------------------------------------------------------------
#  Encoder modules
# ---------------------------------------------------------------------------

class TactileEncoder(nn.Module):
    """4-layer CNN with global average pooling for tactile images."""
    out_dim: int = TACTILE_ENC_DIM

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, 160, 120, 3)
        x = nn.Conv(32, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(128, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(256, (3, 3), strides=(2, 2))(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))          # global avg pool -> (B, 256)
        x = nn.Dense(self.out_dim)(x)
        x = nn.relu(x)
        return x


class ForceFieldEncoder(nn.Module):
    """Two-layer MLP encoder for flattened force field grid."""
    out_dim: int = FORCEFIELD_ENC_DIM
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, 420)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        x = nn.relu(x)
        return x


class ObsEncoder(nn.Module):
    """Encodes tactile + forcefield and concatenates with passthrough (wrist+state).

    * **Split dict** (``metadata["modality_storage"] == "split"``): keys
      ``pass`` (passthrough), ``tact`` ``(B,H,W,C)``, ``forcefield`` ``(B,420)``.
    * **Flat vector** (legacy concat pickle): layout
      ``[wrist | tactile | forcefield | state]``.
    """
    @nn.compact
    def __call__(self, observations) -> jnp.ndarray:
        # Split-format batches use a dict; legacy concat uses (B, full_dim).
        if getattr(observations, "ndim", None) == 2:
            return self._encode_flat(observations)
        passthrough = observations["pass"]
        tactile = observations["tact"]
        ff_flat = observations["forcefield"]
        tac_feat = TactileEncoder()(tactile)
        ff_feat = ForceFieldEncoder()(ff_flat)
        return jnp.concatenate([passthrough, tac_feat, ff_feat], axis=-1)

    def _encode_flat(self, obs_flat: jnp.ndarray) -> jnp.ndarray:
        idx = 0
        wrist = obs_flat[:, idx : idx + WRIST_DIM]
        idx += WRIST_DIM

        tac_flat = obs_flat[:, idx : idx + TACTILE_FLAT_DIM]
        idx += TACTILE_FLAT_DIM
        tactile = tac_flat.reshape(-1, *TACTILE_SHAPE)

        ff_flat = obs_flat[:, idx : idx + FORCEFIELD_FLAT_DIM]
        idx += FORCEFIELD_FLAT_DIM

        state = obs_flat[:, idx : idx + STATE_DIM]

        tac_feat = TactileEncoder()(tactile)
        ff_feat = ForceFieldEncoder()(ff_flat)

        return jnp.concatenate([wrist, tac_feat, ff_feat, state], axis=-1)


# ---------------------------------------------------------------------------
#  Multi-modal IQL network heads
# ---------------------------------------------------------------------------

class MMValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations) -> jnp.ndarray:
        encoded = ObsEncoder()(observations)
        critic = MLP((*self.hidden_dims, 1))(encoded)
        return jnp.squeeze(critic, -1)


class MMCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: callable = nn.relu

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray) -> jnp.ndarray:
        encoded = ObsEncoder()(observations)
        inputs = jnp.concatenate([encoded, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class MMDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: callable = nn.relu

    @nn.compact
    def __call__(self, observations, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        c1 = MMCritic(self.hidden_dims, activations=self.activations)(
            observations, actions)
        c2 = MMCritic(self.hidden_dims, activations=self.activations)(
            observations, actions)
        return c1, c2


class MMNormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
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
        encoded = ObsEncoder()(observations)

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
#  Multi-modal Learner
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


class MultiModalLearner:
    """IQL Learner with per-head CNN/MLP observation encoders.

    Drop-in replacement for ``learner.Learner`` -- exposes the same
    ``update(batch)`` and ``sample_actions(obs)`` API.
    """

    def __init__(
        self,
        seed: int,
        observations,
        actions: jnp.ndarray,
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

        # --- actor ---
        actor_def = MMNormalTanhPolicy(
            hidden_dims, action_dim,
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
        critic_def = MMDoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        # --- value ---
        value_def = MMValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        # --- target critic (no optimiser, soft-updated) ---
        target_critic = Model.create(critic_def,
                                     inputs=[critic_key, observations, actions])

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
