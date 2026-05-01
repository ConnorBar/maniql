"""Async rollout watcher: load PyTorch IQL checkpoints and evaluate in IsaacGym.

Eval-only: rollouts do not feed back into training. This script also saves
trajectories to disk (one file per checkpoint) for later visualization.

Video recording:
    --record_video saves wrist camera frames as MP4 for each episode.
    Works headless (no display needed). Requires imageio[ffmpeg]:
        pip install imageio[ffmpeg]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Isaac Gym MUST be imported before torch. We do the isaacgym import here at
# the top level so that it initialises before any torch import.  The actual
# env creation still happens lazily in _make_isaac_env().
import isaacgym  # noqa: F401

import torch

from multimodal_nets import MultiModalEncoder, PolicyHead, encoded_obs_dim
from log_utils import init_wandb, setup_logging, wandb_log, write_jsonl


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
    p.add_argument("--train_cfg", default="", help="Hydra train config name override (e.g. TacSLTaskInsertionPPO_LSTM_dict_AAC).")
    p.add_argument("--traj_dir", default="", help="Where to write .npz trajectories (default: <save_dir>/trajectories).")
    p.add_argument("--record_video", action="store_true", default=False, help="Save wrist camera frames as MP4 videos.")
    p.add_argument("--video_fps", type=int, default=30, help="FPS for saved videos.")
    p.add_argument("--video_episodes", type=int, default=3, help="How many episodes to record per checkpoint (saves disk).")
    # Logging / W&B
    p.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", default="torch-maniql")
    p.add_argument("--wandb_entity", default="", help="W&B entity/team (optional).")
    p.add_argument("--wandb_name", default="", help="W&B run name (optional).")
    p.add_argument("--wandb_group", default="", help="W&B group (optional).")
    p.add_argument("--wandb_tags", nargs="*", default=[], help="W&B tags (optional).")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


ARGS = None  # populated in main()


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
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import isaacgymenvs  # noqa: F401
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict

    cfg_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), "cfg")

    # Find any available train config — we don't use it, but hydra requires one.
    train_cfg = ARGS.train_cfg
    if not train_cfg:
        train_dir = os.path.join(cfg_dir, "train")
        avail = [f.replace(".yaml", "") for f in os.listdir(train_dir) if f.endswith(".yaml")]
        train_cfg = avail[0] if avail else "AntPPO"

    overrides = [
        f"task={ARGS.task}",
        f"train={train_cfg}",
        f"num_envs={ARGS.num_envs}",
        f"seed={int(ARGS.seed) if ARGS.seed >= 0 else 0}",
        f"sim_device={ARGS.sim_device}",
        f"rl_device={ARGS.rl_device}",
        f"graphics_device_id={ARGS.graphics_device_id}",
        f"headless={ARGS.headless}",
    ]

    with initialize_config_dir(config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=overrides)

        task_cfg = omegaconf_to_dict(cfg.task)

        task_name = task_cfg["name"]
        if task_name not in isaacgym_task_map:
            for registered in isaacgym_task_map:
                if task_name.lower() in registered.lower() or registered.lower() in task_name.lower():
                    print(f"[INFO] Task '{task_name}' not in task_map, using '{registered}'")
                    task_name = registered
                    break
            else:
                raise KeyError(
                    f"Task '{task_name}' not found in isaacgym_task_map. "
                    f"Available: {sorted(isaacgym_task_map.keys())}"
                )

        env = isaacgym_task_map[task_name](
            cfg=task_cfg,
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=True,
            force_render=True,
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


def _load_actor(meta: dict, checkpoint_dir: str, device: torch.device) -> Tuple[MultiModalEncoder, PolicyHead]:
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    payload = torch.load(ckpt_path, map_location="cpu")
    arch = meta["arch"]
    mode = meta["mode"]
    obs_dim = encoded_obs_dim(arch, mode)

    encoder = MultiModalEncoder(arch=arch, mode=mode, r3m_checkpoint=None).to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()

    actor = PolicyHead(
        obs_dim=obs_dim,
        hidden_dims=tuple(meta["hidden_dims"]),
        action_dim=int(meta["action_dim"]),
    ).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    return encoder, actor


def _save_video(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    """Write a list of uint8 HWC frames to an MP4 file."""
    try:
        import imageio.v2 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            print(f"[WARN] imageio not installed, skipping video save: {path}")
            return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = iio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()


def _extract_wrist_frame(pobs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Pull a single HWC uint8 wrist frame from the policy obs dict."""
    if "wrist" not in pobs:
        return None
    img = pobs["wrist"]
    if img.ndim == 4:
        img = img[0]
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _run_episodes(
    encoder: MultiModalEncoder,
    actor: PolicyHead,
    env,
    meta: dict,
    n_episodes: int,
    max_steps: int,
    *,
    traj_dir: str,
    step: int,
    device: torch.device,
    record_video: bool = False,
    video_fps: int = 30,
    video_episodes: int = 3,
) -> dict:
    returns: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []
    video_paths: List[str] = []

    os.makedirs(traj_dir, exist_ok=True)
    traj_path = os.path.join(traj_dir, f"traj_step{step}.npz")
    video_dir = os.path.join(traj_dir, "videos")
    all_eps = []

    for ep_idx in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict) and "obs" in obs and len(obs) == 1:
            obs = obs["obs"]

        ep = {"obs": [], "action": [], "reward": [], "done": []}
        ep_ret = 0.0
        ep_len = 0
        ep_success = 0.0
        should_record = record_video and ep_idx < video_episodes
        frames: List[np.ndarray] = []

        for _ in range(max_steps):
            pobs = isaac_obs_to_policy_obs(obs, meta)
            tobs = {k: torch.as_tensor(v, device=device) for k, v in pobs.items()}

            if should_record:
                frame = _extract_wrist_frame(pobs)
                if frame is not None:
                    frames.append(frame)

            with torch.no_grad():
                encoded = encoder(tobs)
                action = actor.act(encoded, deterministic=True).detach().cpu().numpy().reshape(-1).astype(np.float32)

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

        if should_record and frames:
            vpath = os.path.join(video_dir, f"step{step}_ep{ep_idx}.mp4")
            _save_video(frames, vpath, fps=video_fps)
            video_paths.append(vpath)

    np.savez_compressed(traj_path, episodes=np.array(all_eps, dtype=object))

    result = {
        "return_mean": float(np.mean(returns)) if returns else float("nan"),
        "return_std": float(np.std(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "n_episodes": len(returns),
        "trajectory_file": traj_path,
    }
    if video_paths:
        result["video_paths"] = video_paths
    return result


def main() -> None:
    global ARGS
    ARGS = _parse_args()
    logger = setup_logging(ARGS.save_dir, name="torch-maniql.rollout", level=ARGS.log_level)
    meta = _await_meta(ARGS.save_dir, ARGS.poll_interval)
    if meta.get("backend") != "pytorch":
        raise SystemExit(f"training_meta.json backend={meta.get('backend')} not supported; expected 'pytorch'")

    logger.info("Loaded training_meta.json: mode=%s arch=%s action_dim=%s", meta["mode"], meta["arch"], meta["action_dim"])

    wandb = init_wandb(
        enabled=bool(ARGS.wandb) and ARGS.wandb_mode != "disabled",
        project=ARGS.wandb_project,
        entity=ARGS.wandb_entity or None,
        name=ARGS.wandb_name or None,
        group=ARGS.wandb_group or None,
        tags=ARGS.wandb_tags or None,
        mode=ARGS.wandb_mode,
        save_dir=ARGS.save_dir,
        config={**vars(ARGS), **{"training_meta": meta}},
    )
    metrics_path = os.path.join(ARGS.save_dir, "metrics", "eval_metrics.jsonl")

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
                logger.info("[EVAL] step=%d checkpoint=%s", step, path)
                try:
                    encoder, actor = _load_actor(meta, path, device=device)
                except Exception as e:
                    logger.warning("Failed to load checkpoint %s: %s", path, e)
                    seen.add(step)
                    continue

                try:
                    stats = _run_episodes(
                        encoder,
                        actor,
                        env,
                        meta,
                        n_episodes=ARGS.eval_episodes,
                        max_steps=ARGS.max_episode_steps,
                        traj_dir=traj_dir,
                        step=step,
                        device=device,
                        record_video=ARGS.record_video,
                        video_fps=ARGS.video_fps,
                        video_episodes=ARGS.video_episodes,
                    )
                except Exception as e:
                    logger.warning("Rollout failed at step %d: %s", step, e)
                    seen.add(step)
                    continue

                logger.info(
                    "[EVAL] step=%d success=%.3f return=%.3f±%.3f len=%.1f n=%d traj=%s",
                    step,
                    float(stats["success_rate"]),
                    float(stats["return_mean"]),
                    float(stats["return_std"]),
                    float(stats["episode_length_mean"]),
                    int(stats["n_episodes"]),
                    stats["trajectory_file"],
                )
                if "video_paths" in stats:
                    logger.info("[EVAL] Videos saved: %s", ", ".join(stats["video_paths"]))

                eval_metrics = {f"evaluation/{k}": v for k, v in stats.items() if isinstance(v, (int, float, np.floating))}
                wandb_log(wandb, eval_metrics, step=step)
                write_jsonl(metrics_path, {"step": int(step), **{k: float(v) for k, v in eval_metrics.items()}})
                seen.add(step)

            if ARGS.once:
                break
    finally:
        if wandb is not None:
            try:
                wandb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
