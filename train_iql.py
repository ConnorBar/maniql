"""Offline IQL training on ManiFeel preprocessed datasets.

Usage (wrist + state only):
    python train_iql.py \
        --dataset_path data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl \
        --use_features wrist,state \
        --save_dir runs/manifeel_iql_wrist_state \
        --max_steps 500000

Usage (full observation):
    python train_iql.py \
        --dataset_path data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl \
        --save_dir runs/manifeel_iql_full \
        --max_steps 500000
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "implicit_q_learning"))

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from learner import Learner
from manifeel_iql import ManiFeelDataset

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", "", "Path to ManiFeel preprocessed pickle.")
flags.DEFINE_string("save_dir", "./runs/manifeel_iql/", "Tensorboard + checkpoint dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of gradient steps.")
flags.DEFINE_integer("save_interval", 50000, "Checkpoint save interval (0 = final only).")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("normalize_rewards", False, "Normalize rewards by trajectory return range.")
flags.DEFINE_boolean("clip_actions", True, "Clip actions to [-1+eps, 1-eps].")
flags.DEFINE_boolean("validate", True, "Run data sanity checks before training.")
flags.DEFINE_string(
    "use_features", None,
    "Comma-separated feature subset to use from obs_layout, e.g. 'wrist,state'. "
    "If unset, uses the full observation vector."
)


config_flags.DEFINE_config_file(
    "config",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "implicit_q_learning", "configs", "mujoco_config.py"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


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


def save_checkpoint(agent: Learner, save_dir: str, step: int):
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
    if FLAGS.use_features:
        use_features = [f.strip() for f in FLAGS.use_features.split(",") if f.strip()]

    dataset = ManiFeelDataset(
        FLAGS.dataset_path,
        use_features=use_features,
        clip_actions=FLAGS.clip_actions,
    )
    print(dataset.summary())
    if FLAGS.validate:
        dataset.validate()

    if FLAGS.normalize_rewards:
        normalize_rewards(dataset)

    kwargs = dict(FLAGS.config)
    agent = Learner(
        FLAGS.seed,
        dataset.observations[:1],   # sample obs  (1, obs_dim)
        dataset.actions[:1],        # sample act  (1, act_dim)
        max_steps=FLAGS.max_steps,
        **kwargs,
    )

    print(f"\n[START] Training IQL for {FLAGS.max_steps:,} steps (batch_size={FLAGS.batch_size})")
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f"training/{k}", float(v), i)
                else:
                    summary_writer.add_histogram(f"training/{k}", v, i)
            summary_writer.flush()

        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_checkpoint(agent, FLAGS.save_dir, i)

    save_checkpoint(agent, FLAGS.save_dir, FLAGS.max_steps)
    summary_writer.close()
    print(f"\n[DONE] Training complete.  Checkpoints saved to {FLAGS.save_dir}")


if __name__ == "__main__":
    app.run(main)
