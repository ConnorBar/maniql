"""Async rollout watcher: load JAX IQL checkpoints and evaluate in IsaacGym.

Run this in the **Isaac/Torch** env (GPU torch + torchvision + isaacgym +
isaacgymenvs). It is decoupled from training (train_iql.py), so training can
live in a JAX-only env without cuDNN version fights.

Responsibilities
----------------
1. Poll ``--save_dir`` for new ``checkpoint_<step>/`` directories that contain
   a ``DONE`` sentinel file (written by ``train_iql.save_checkpoint``).
2. Rebuild the policy (MultiModalLearner) from ``training_meta.json`` so we
   can load ``actor.flax`` params.
3. Instantiate the IsaacGym task and run ``--eval_episodes`` episodes per
   checkpoint, computing ``success_rate`` and ``avg_return``.
4. Log metrics to the same TensorBoard run (``<save_dir>/tb/<seed>``) under
   ``evaluation/*`` with the training step as x-axis.

Important caveats
-----------------
* Your offline dataset and the Isaac task must agree on observation semantics
  (keys, shapes, dtypes, value ranges). The adapter ``isaac_obs_to_policy_obs``
  below is a starting point — edit it for your task.
* ``isaacgymenvs.make`` needs the task name + cfg overrides; use ``--task``.
* JAX inference and Torch-backed Isaac coexist in this script, but in the
  rollout env only. Keep this env separate from the training env.

Example
-------
    python maniql/rollout_watch_isaac.py \\
        --save_dir runs/iql_ws_r3m18 \\
        --task TacSLTaskInsertion \\
        --eval_episodes 10 \\
        --num_envs 1 \\
        --poll_interval 30
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from absl import app, flags

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "implicit_q_learning"))

FLAGS = flags.FLAGS

flags.DEFINE_string("save_dir", "", "Training save_dir (must contain training_meta.json).")
flags.DEFINE_string("task", "", "IsaacGymEnvs task name, e.g. TacSLTaskInsertion.")
flags.DEFINE_integer("num_envs", 1, "Parallel Isaac envs (kept small for eval).")
flags.DEFINE_integer("eval_episodes", 10, "Episodes per checkpoint.")
flags.DEFINE_integer("max_episode_steps", 1000, "Hard cap per episode.")
flags.DEFINE_integer("poll_interval", 30, "Seconds between checkpoint scans.")
flags.DEFINE_string("sim_device", "cuda:0", "Isaac sim_device.")
flags.DEFINE_string("rl_device", "cuda:0", "Isaac rl_device.")
flags.DEFINE_integer("graphics_device_id", 0, "Isaac graphics device id.")
flags.DEFINE_boolean("headless", True, "Run Isaac headless (no viewer).")
flags.DEFINE_integer("seed", -1, "Rollout seed (-1 = reuse training seed).")
flags.DEFINE_boolean("once", False, "Evaluate whatever's available now and exit.")


# ---------------------------------------------------------------------------
#  Checkpoint polling
# ---------------------------------------------------------------------------

def _list_ready_checkpoints(save_dir: str) -> List[Tuple[int, str]]:
    """Return [(step, path), ...] for ``checkpoint_<n>`` dirs that have DONE."""
    if not os.path.isdir(save_dir):
        return []
    out: List[Tuple[int, str]] = []
    for name in os.listdir(save_dir):
        if not name.startswith("checkpoint_"):
            continue
        suffix = name[len("checkpoint_"):]
        if not suffix.isdigit():
            continue
        path = os.path.join(save_dir, name)
        if not os.path.isfile(os.path.join(path, "DONE")):
            continue
        out.append((int(suffix), path))
    out.sort(key=lambda kv: kv[0])
    return out


# ---------------------------------------------------------------------------
#  Policy reconstruction (JAX)
# ---------------------------------------------------------------------------

def _make_obs_example(meta: dict) -> Dict[str, np.ndarray]:
    """Rebuild a dummy observation dict with the same shapes/dtypes as training."""
    out = {}
    for k, shape in meta["obs_shapes"].items():
        dtype = np.dtype(meta["obs_dtypes"][k])
        out[k] = np.zeros(shape, dtype=dtype)
    return out


def _load_policy(meta: dict, checkpoint_dir: str):
    """Instantiate MultiModalLearner and load actor params from checkpoint_dir."""
    from multimodal_nets import MultiModalLearner

    obs_example = _make_obs_example(meta)
    actions_example = np.zeros((1, meta["action_dim"]), dtype=np.float32)
    agent = MultiModalLearner(
        seed=int(meta.get("seed", 0)),
        observations=obs_example,
        actions=actions_example,
        arch=meta["arch"],
        mode=meta["mode"],
        r3m_checkpoint=None,
        hidden_dims=tuple(meta["hidden_dims"]),
        max_steps=meta.get("max_steps", 1),
    )
    actor_path = os.path.join(checkpoint_dir, "actor.flax")
    if not os.path.isfile(actor_path):
        raise FileNotFoundError(actor_path)
    agent.actor = agent.actor.load(actor_path)
    return agent


# ---------------------------------------------------------------------------
#  Isaac env + obs adapter
# ---------------------------------------------------------------------------

def _make_isaac_env():
    """Instantiate the IsaacGymEnvs task. Edit if your registration differs."""
    import isaacgymenvs  # noqa: F401  (uses side-effect imports + .make)

    env = isaacgymenvs.make(
        seed=FLAGS.seed if FLAGS.seed >= 0 else 0,
        task=FLAGS.task,
        num_envs=FLAGS.num_envs,
        sim_device=FLAGS.sim_device,
        rl_device=FLAGS.rl_device,
        graphics_device_id=FLAGS.graphics_device_id,
        headless=FLAGS.headless,
    )
    return env


def isaac_obs_to_policy_obs(raw, meta: dict) -> Dict[str, np.ndarray]:
    """Convert Isaac observation into the dict the JAX policy expects.

    IsaacGymEnvs tasks return a dict (or tensor) of torch tensors. You MUST
    adapt this to match the keys/shapes declared in ``meta['obs_shapes']``.
    The default below handles two common shapes:

    * raw is a torch tensor → treat as flat 'state' vector.
    * raw is a dict → pick out keys that match the training obs keys.
    """
    import torch  # local import; only needed in the rollout env

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    want_keys = list(meta["obs_shapes"].keys())

    if isinstance(raw, dict):
        out = {}
        for k in want_keys:
            if k not in raw:
                raise KeyError(
                    f"Isaac obs missing key {k!r}; got {list(raw.keys())}. "
                    f"Edit isaac_obs_to_policy_obs() for this task.")
            arr = _to_np(raw[k])
            if arr.ndim == len(meta["obs_shapes"][k]) - 1:
                arr = arr[None, ...]
            out[k] = arr.astype(np.dtype(meta["obs_dtypes"][k]), copy=False)
        return out

    arr = _to_np(raw)
    if len(want_keys) == 1:
        k = want_keys[0]
        if arr.ndim == len(meta["obs_shapes"][k]) - 1:
            arr = arr[None, ...]
        return {k: arr.astype(np.dtype(meta["obs_dtypes"][k]), copy=False)}

    raise TypeError(
        "Isaac obs is not a dict and multiple obs keys are expected. "
        "Edit isaac_obs_to_policy_obs() for this task.")


# ---------------------------------------------------------------------------
#  Rollout
# ---------------------------------------------------------------------------

def _run_episodes(agent, env, meta: dict, n_episodes: int,
                  max_steps: int) -> dict:
    """Run ``n_episodes`` sequential episodes. Returns aggregate stats."""
    import torch

    returns: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []

    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict) and "obs" in obs and len(obs) == 1:
            obs = obs["obs"]

        ep_ret = 0.0
        ep_len = 0
        ep_success = 0.0
        for _ in range(max_steps):
            pobs = isaac_obs_to_policy_obs(obs, meta)
            action = agent.sample_actions(pobs, temperature=0.0)
            action = np.asarray(action).reshape(-1).astype(np.float32)

            act_t = torch.from_numpy(
                np.tile(action[None, :], (FLAGS.num_envs, 1))
            ).to(FLAGS.rl_device)
            step_out = env.step(act_t)

            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done_t = terminated | truncated
            else:
                obs, reward, done_t, info = step_out

            if isinstance(obs, dict) and "obs" in obs and len(obs) == 1:
                obs = obs["obs"]

            r = float(np.asarray(reward).reshape(-1)[0])
            ep_ret += r
            ep_len += 1

            if isinstance(info, dict):
                s = info.get("successes", info.get("success", None))
                if s is not None:
                    ep_success = float(np.asarray(s).reshape(-1)[0])

            d = bool(np.asarray(done_t).reshape(-1)[0])
            if d:
                break

        returns.append(ep_ret)
        successes.append(ep_success)
        lengths.append(ep_len)

    return {
        "return_mean": float(np.mean(returns)) if returns else float("nan"),
        "return_std": float(np.std(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": len(returns),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def _read_meta(save_dir: str) -> dict:
    path = os.path.join(save_dir, "training_meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Start training first so train_iql.py can write it.")
    with open(path) as f:
        return json.load(f)


def _await_meta(save_dir: str, poll_interval: int) -> dict:
    last = 0.0
    while True:
        try:
            return _read_meta(save_dir)
        except FileNotFoundError:
            now = time.time()
            if now - last > 30:
                print(f"[WAIT] training_meta.json not present in {save_dir}")
                last = now
            time.sleep(poll_interval)


def main(_):
    if not FLAGS.save_dir:
        raise SystemExit("--save_dir is required.")
    if not FLAGS.task:
        raise SystemExit("--task is required (e.g. TacSLTaskInsertion).")

    meta = _await_meta(FLAGS.save_dir, FLAGS.poll_interval)
    print(f"[INFO] Loaded training_meta.json: mode={meta['mode']} "
          f"arch={meta['arch']} action_dim={meta['action_dim']}")

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, "tb",
                     str(FLAGS.seed if FLAGS.seed >= 0 else meta.get("seed", 0))),
        write_to_disk=True,
    )

    env = _make_isaac_env()

    seen: set = set()
    try:
        while True:
            ready = _list_ready_checkpoints(FLAGS.save_dir)
            new = [(s, p) for (s, p) in ready if s not in seen]
            if not new:
                if FLAGS.once:
                    break
                time.sleep(FLAGS.poll_interval)
                continue

            for step, path in new:
                print(f"[EVAL] step={step} checkpoint={path}")
                try:
                    agent = _load_policy(meta, path)
                except Exception as e:
                    print(f"[WARN] Failed to load checkpoint {path}: {e}")
                    seen.add(step)
                    continue

                try:
                    stats = _run_episodes(
                        agent, env, meta,
                        n_episodes=FLAGS.eval_episodes,
                        max_steps=FLAGS.max_episode_steps,
                    )
                except Exception as e:
                    print(f"[WARN] Rollout failed at step {step}: {e}")
                    seen.add(step)
                    continue

                print(f"[EVAL] step={step} "
                      f"success={stats['success_rate']:.3f} "
                      f"return={stats['return_mean']:.3f}±{stats['return_std']:.3f} "
                      f"len={stats['episode_length_mean']:.1f} "
                      f"n={stats['n_episodes']}")

                for k, v in stats.items():
                    writer.add_scalar(f"evaluation/{k}", float(v), step)
                writer.flush()
                seen.add(step)

            if FLAGS.once:
                break
    finally:
        writer.close()


if __name__ == "__main__":
    app.run(main)
