"""Offline IQL training on ManiFeel preprocessed datasets.

Usage (unimodal, wrist+state only):
    python train_iql.py \
        --dataset_path data/preprocessed/all_transitions_r3m_wrist_state.pkl \
        --save_dir runs/manifeel_iql_wrist_state \
        --max_steps 200000

Usage (multi-modal, all modalities with CNN/MLP encoders):
    python train_iql.py \
        --dataset_path data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl \
        --multimodal \
        --save_dir runs/manifeel_iql_multimodal \
        --max_steps 200000

    If you see CUDA_ERROR_OUT_OF_MEMORY, lower VRAM use with e.g.
    ``--batch_size 128`` or ``64`` (default 256 is aggressive for 3 CNN encoders).
"""

import os
import sys

# JAX defaults to grabbing most GPU VRAM up front; on WSL / smaller GPUs that
# often causes CUDA OOM during peaks (multimodal IQL uses large tactile batches).
# Override with XLA_PYTHON_CLIENT_PREALLOCATE=true if you prefer the default.
# this is the main reason that fixed the OOM issue
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# gets access to critic and learner modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "implicit_q_learning"))

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from critic import loss as expectile_loss
from learner import Learner
from manifeel_iql import ManiFeelDataset
from multimodal_nets import MultiModalLearner, FULL_OBS_DIM # TODO: the obs dim likely needs to be hyperparameterized

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", "", "Path to ManiFeel preprocessed pickle.")
flags.DEFINE_string("save_dir", "./runs/manifeel_iql/", "Tensorboard + checkpoint dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 2000, "Test-set evaluation interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of gradient steps.")
flags.DEFINE_integer("save_interval", 50000, "Checkpoint save interval (0 = final only).")
flags.DEFINE_float("test_ratio", 0.1, "Fraction of episodes held out for test.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("normalize_rewards", False, "Normalize rewards by trajectory return range.")
flags.DEFINE_boolean("clip_actions", True, "Clip actions to [-1+eps, 1-eps].")
flags.DEFINE_boolean("validate", True, "Run data sanity checks before training.")
flags.DEFINE_string(
    "use_features", None,
    "Comma-separated feature subset to use from obs_layout, e.g. 'wrist,state'. "
    "If unset, uses the full observation vector.  Ignored when --multimodal."
)
flags.DEFINE_boolean(
    "multimodal", False,
    "Use multi-modal IQL with CNN (tactile) and MLP (forcefield) encoders "
    "trained end-to-end.  Requires a dataset preprocessed with all four "
    "features (wrist, tactile, forcefield, state)."
)

config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "implicit_q_learning", "configs", "mujoco_config.py"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


# ---- eval losses (read-only, no gradient updates) -------------------------

@jax.jit
def _eval_losses(actor, critic, value, target_critic, batch,
                 discount, expectile, temperature, rng):
    """Compute IQL losses on a batch without updating any parameters."""
    # Q from target critic, V from value net
    q1, q2 = target_critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)

    # Value loss (expectile regression)
    value_loss = expectile_loss(q - v, expectile).mean()

    # Critic loss
    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v
    q1_pred, q2_pred = critic(batch.observations, batch.actions)
    critic_loss = ((q1_pred - target_q) ** 2 + (q2_pred - target_q) ** 2).mean()

    # Actor loss
    dist = actor.apply_fn.apply({"params": actor.params}, batch.observations)
    log_probs = dist.log_prob(batch.actions)
    exp_a = jnp.minimum(jnp.exp((q - v) * temperature), 100.0)
    actor_loss = -(exp_a * log_probs).mean()

    return {
        "critic_loss": critic_loss,
        "value_loss": value_loss,
        "actor_loss": actor_loss,
        "v": v.mean(),
        "q1": q1_pred.mean(),
        "q2": q2_pred.mean(),
        "adv": (q - v).mean(),
    }


def eval_on_dataset(agent, dataset, batch_size, n_batches=10):
    """Average eval losses over multiple batches from dataset."""
    accum = None
    for _ in range(n_batches):
        batch = dataset.sample(batch_size)
        info = _eval_losses(
            agent.actor, agent.critic, agent.value, agent.target_critic,
            batch, agent.discount, agent.expectile, agent.temperature,
            agent.rng,
        )
        info = jax.device_get(info)
        if accum is None:
            accum = {k: float(np.asarray(v)) for k, v in info.items()}
        else:
            for k, v in info.items():
                accum[k] += float(np.asarray(v))
    return {k: v / n_batches for k, v in accum.items()}


# ---- reward normalization -------------------------------------------------

def normalize_rewards(dataset: ManiFeelDataset):
    """Normalize rewards by range of per-trajectory returns, scaled to 1000."""
    episode_returns = []
    cur_return = 0.0
    for i in range(dataset.size):
        cur_return += float(dataset.rewards[i])
        if dataset.dones_float[i] == 1.0:
            episode_returns.append(cur_return)
            cur_return = 0.0

    if len(episode_returns) < 2:
        print("[WARN] <2 trajectories found, skipping reward normalization.")
        return

    ret_range = max(episode_returns) - min(episode_returns)
    if ret_range < 1e-8:
        print("[WARN] All trajectories have nearly identical returns, skipping normalization.")
        return

    dataset.rewards /= ret_range
    dataset.rewards *= 1000.0
    print(f"[INFO] Rewards normalized: range {ret_range:.4f} -> scaled to 1000.")


def save_checkpoint(agent, save_dir: str, step: int):
    ckpt_dir = os.path.join(save_dir, f"checkpoint_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    agent.actor.save(os.path.join(ckpt_dir, "actor.flax"))
    agent.critic.save(os.path.join(ckpt_dir, "critic.flax"))
    agent.value.save(os.path.join(ckpt_dir, "value.flax"))


def main(_):
    if not FLAGS.dataset_path:
        raise ValueError("--dataset_path is required.")

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed)), write_to_disk=True
    )

    np.random.seed(FLAGS.seed)

    use_features = None
    if FLAGS.multimodal:
        use_features = None  # need full obs for the multi-modal encoders
    elif FLAGS.use_features:
        use_features = [f.strip() for f in FLAGS.use_features.split(",") if f.strip()]

    full_dataset = ManiFeelDataset(
        FLAGS.dataset_path,
        use_features=use_features,
        clip_actions=FLAGS.clip_actions,
    )

    if FLAGS.multimodal:
        if full_dataset.modality_storage == "split":
            pass
        elif (
            full_dataset.observations is not None
            and full_dataset.observations.shape[1] == FULL_OBS_DIM
        ):
            pass  # legacy single-vector concat format
        else:
            raise ValueError(
                "--multimodal needs either modality-split data (re-run seed_data with "
                "wrist,tactile,forcefield,state) or a flat obs of dim "
                f"{FULL_OBS_DIM}."
            )

    # Episode-level train/test split
    train_ds, test_ds = full_dataset.train_test_split(
        test_ratio=FLAGS.test_ratio, seed=FLAGS.seed
    )
    print("=== Train set ===")
    print(train_ds.summary())
    print(f"\n=== Test set ===")
    print(test_ds.summary())
    print()

    if FLAGS.validate:
        print("--- Train validation ---")
        train_ds.validate()
        print("--- Test validation ---")
        test_ds.validate()
        print()

    if FLAGS.normalize_rewards:
        normalize_rewards(train_ds)
        normalize_rewards(test_ds)

    kwargs = dict(FLAGS.config)
    LearnerCls = MultiModalLearner if FLAGS.multimodal else Learner
    agent = LearnerCls(
        FLAGS.seed,
        train_ds.observation_example(),
        train_ds.actions[:1],
        max_steps=FLAGS.max_steps,
        **kwargs,
    )

    mode_str = "multi-modal (CNN+MLP encoders)" if FLAGS.multimodal else "unimodal"
    print(f"[START] Training IQL ({mode_str}) for {FLAGS.max_steps:,} steps "
          f"(batch={FLAGS.batch_size}, train={train_ds.size:,}, test={test_ds.size:,})")

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = train_ds.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            # One host transfer avoids repeated GPU syncs; scalars must be plain
            # floats for TensorBoard (float(jax.Array) can fail under VRAM pressure).
            info_host = jax.device_get(update_info)
            for k, v in info_host.items():
                v = np.asarray(v)
                if v.ndim == 0:
                    summary_writer.add_scalar(f"train/{k}", float(v), i)
                else:
                    summary_writer.add_histogram(f"train/{k}", v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            test_info = eval_on_dataset(agent, test_ds, FLAGS.batch_size)
            train_info = eval_on_dataset(agent, train_ds, FLAGS.batch_size)
            for k, v in test_info.items():
                summary_writer.add_scalar(f"test/{k}", v, i)
            for k, v in train_info.items():
                summary_writer.add_scalar(f"train_eval/{k}", v, i)
            summary_writer.flush()

        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_checkpoint(agent, FLAGS.save_dir, i)

    save_checkpoint(agent, FLAGS.save_dir, FLAGS.max_steps)
    summary_writer.close()
    print(f"\n[DONE] Training complete.  Checkpoints saved to {FLAGS.save_dir}")


if __name__ == "__main__":
    app.run(main)
