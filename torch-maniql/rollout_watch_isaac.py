"""Async rollout watcher: load PyTorch IQL checkpoints and evaluate in IsaacGym.

Eval-only: rollouts do not feed back into training. This script also saves
trajectories to disk (one file per checkpoint) for later visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from multimodal_nets import DiagGaussianPolicy


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True, help="Training save_dir (must contain training_meta.json).")
    p.add_argument("--task", required=True, help="IsaacGymEnvs task name.")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--max_episode_steps", type=int, default=1000)
    p.add_argument("--poll_interval", type=int, default=30)
    p.add_argument("--sim_device", default="cuda:0")
    p.add_argument("--rl_device", default="cuda:0")
    p.add_argument("--graphics_device_id", type=int, default=0)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=-1, help="Rollout seed (-1 = reuse training seed).")
    p.add_argument("--once", action="store_true", default=False)
    p.add_argument("--traj_dir", default="", help="Where to write .npz trajectories (default: <save_dir>/trajectories).")
    return p.parse_args()


ARGS = _parse_args()


def _list_ready_checkpoints(save_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(save_dir):
        return []
    out: List[Tuple[int, str]] = []
    for name in os.listdir(save_dir):
        if not name.startswith("checkpoint_"):
            continue
        suffix = name[len("checkpoint_") :]
        if not suffix.isdigit():
            continue
        path = os.path.join(save_dir, name)
        if not os.path.isfile(os.path.join(path, "DONE")):
            continue
        out.append((int(suffix), path))
    out.sort(key=lambda kv: kv[0])
    return out


def _read_meta(save_dir: str) -> dict:
    path = os.path.join(save_dir, "training_meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
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


def _make_isaac_env():
    import isaacgymenvs  # noqa: F401

    env = isaacgymenvs.make(
        seed=int(ARGS.seed) if ARGS.seed >= 0 else 0,
        task=ARGS.task,
        num_envs=ARGS.num_envs,
        sim_device=ARGS.sim_device,
        rl_device=ARGS.rl_device,
        graphics_device_id=ARGS.graphics_device_id,
        headless=ARGS.headless,
    )
    return env


def isaac_obs_to_policy_obs(raw, meta: dict) -> Dict[str, np.ndarray]:
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
                    f"Edit isaac_obs_to_policy_obs() for this task."
                )
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

    raise TypeError("Isaac obs is not a dict and multiple obs keys are expected.")


def _load_actor(meta: dict, checkpoint_dir: str, device: torch.device) -> DiagGaussianPolicy:
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    payload = torch.load(ckpt_path, map_location="cpu")
    actor_sd = payload["actor"]
    actor = DiagGaussianPolicy(
        arch=meta["arch"],
        mode=meta["mode"],
        hidden_dims=tuple(meta["hidden_dims"]),
        action_dim=int(meta["action_dim"]),
        r3m_checkpoint=None,
    ).to(device)
    actor.load_state_dict(actor_sd, strict=True)
    actor.eval()
    return actor


def _run_episodes(
    actor: DiagGaussianPolicy,
    env,
    meta: dict,
    n_episodes: int,
    max_steps: int,
    *,
    traj_dir: str,
    step: int,
    device: torch.device,
) -> dict:
    returns: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []

    os.makedirs(traj_dir, exist_ok=True)
    traj_path = os.path.join(traj_dir, f"traj_step{step}.npz")
    all_eps = []

    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict) and "obs" in obs and len(obs) == 1:
            obs = obs["obs"]

        ep = {"obs": [], "action": [], "reward": [], "done": []}
        ep_ret = 0.0
        ep_len = 0
        ep_success = 0.0

        for _ in range(max_steps):
            pobs = isaac_obs_to_policy_obs(obs, meta)
            tobs = {k: torch.as_tensor(v, device=device) for k, v in pobs.items()}
            with torch.no_grad():
                action = actor.act(tobs, deterministic=True).detach().cpu().numpy().reshape(-1).astype(np.float32)

            act_t = torch.from_numpy(np.tile(action[None, :], (ARGS.num_envs, 1))).to(ARGS.rl_device)
            step_out = env.step(act_t)

            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done_t = terminated | truncated
            else:
                obs, reward, done_t, info = step_out

            if isinstance(obs, dict) and "obs" in obs and len(obs) == 1:
                obs = obs["obs"]

            r = float(np.asarray(reward).reshape(-1)[0])
            d = bool(np.asarray(done_t).reshape(-1)[0])

            ep["obs"].append(pobs)
            ep["action"].append(action.copy())
            ep["reward"].append(r)
            ep["done"].append(d)

            ep_ret += r
            ep_len += 1

            if isinstance(info, dict):
                s = info.get("successes", info.get("success", None))
                if s is not None:
                    ep_success = float(np.asarray(s).reshape(-1)[0])

            if d:
                break

        returns.append(ep_ret)
        successes.append(ep_success)
        lengths.append(ep_len)
        all_eps.append(ep)

    np.savez_compressed(traj_path, episodes=np.array(all_eps, dtype=object))

    return {
        "return_mean": float(np.mean(returns)) if returns else float("nan"),
        "return_std": float(np.std(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": len(returns),
        "trajectory_file": traj_path,
    }


def main() -> None:
    meta = _await_meta(ARGS.save_dir, ARGS.poll_interval)
    if meta.get("backend") != "pytorch":
        raise SystemExit(f"training_meta.json backend={meta.get('backend')} not supported; expected 'pytorch'")

    print(f"[INFO] Loaded training_meta.json: mode={meta['mode']} arch={meta['arch']} action_dim={meta['action_dim']}")

    writer = SummaryWriter(os.path.join(ARGS.save_dir, "tb", str(ARGS.seed if ARGS.seed >= 0 else meta.get("seed", 0))))
    env = _make_isaac_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traj_dir = ARGS.traj_dir or os.path.join(ARGS.save_dir, "trajectories")

    seen: set[int] = set()
    try:
        while True:
            ready = _list_ready_checkpoints(ARGS.save_dir)
            new = [(s, p) for (s, p) in ready if s not in seen]
            if not new:
                if ARGS.once:
                    break
                time.sleep(ARGS.poll_interval)
                continue

            for step, path in new:
                print(f"[EVAL] step={step} checkpoint={path}")
                try:
                    actor = _load_actor(meta, path, device=device)
                except Exception as e:
                    print(f"[WARN] Failed to load checkpoint {path}: {e}")
                    seen.add(step)
                    continue

                try:
                    stats = _run_episodes(
                        actor,
                        env,
                        meta,
                        n_episodes=ARGS.eval_episodes,
                        max_steps=ARGS.max_episode_steps,
                        traj_dir=traj_dir,
                        step=step,
                        device=device,
                    )
                except Exception as e:
                    print(f"[WARN] Rollout failed at step {step}: {e}")
                    seen.add(step)
                    continue

                print(
                    f"[EVAL] step={step} success={stats['success_rate']:.3f} "
                    f"return={stats['return_mean']:.3f}±{stats['return_std']:.3f} "
                    f"len={stats['episode_length_mean']:.1f} n={stats['n_episodes']} "
                    f"traj={stats['trajectory_file']}"
                )

                for k, v in stats.items():
                    if isinstance(v, (int, float, np.floating)):
                        writer.add_scalar(f"evaluation/{k}", float(v), step)
                writer.flush()
                seen.add(step)

            if ARGS.once:
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
