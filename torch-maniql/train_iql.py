"""Offline IQL training on ManiFeel datasets (pure PyTorch, end-to-end vision finetuning)."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from manifeel_iql import ManiFeelDataset
from multimodal_nets import IQLLearner


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
def eval_on_dataset(agent: IQLLearner, dataset: ManiFeelDataset, batch_size: int, n_batches: int = 10) -> Dict[str, float]:
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    value_losses: list[float] = []
    q_means: list[float] = []
    v_means: list[float] = []
    adv_means: list[float] = []
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


def write_training_meta(save_dir: str, *, mode: str, arch: str, action_dim: int, hidden_dims, dataset_path: str, obs_example: dict, seed: int, batch_size: int, max_steps: int) -> str:
    obs_shapes = {k: list(v.shape) for k, v in obs_example.items()}
    obs_dtypes = {k: str(v.dtype) for k, v in obs_example.items()}
    meta = {
        "backend": "pytorch",
        "mode": mode,
        "arch": arch,
        "action_dim": int(action_dim),
        "hidden_dims": [int(h) for h in hidden_dims],
        "dataset_path": dataset_path,
        "obs_shapes": obs_shapes,
        "obs_dtypes": obs_dtypes,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
    }
    path = os.path.join(save_dir, "training_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def save_checkpoint(agent: IQLLearner, save_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(save_dir, f"checkpoint_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "value": agent.value.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "actor_opt": agent.actor_opt.state_dict(),
        "critic_opt": agent.critic_opt.state_dict(),
        "value_opt": agent.value_opt.state_dict(),
    }
    torch.save(payload, os.path.join(ckpt_dir, "checkpoint.pt"))
    with open(os.path.join(ckpt_dir, "DONE"), "w") as f:
        f.write(str(step))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--save_dir", default="./runs/manifeel_iql/")
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--r3m_checkpoint", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=1_000_000)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--eval_interval", type=int, default=2000)
    p.add_argument("--save_interval", type=int, default=50_000)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--tqdm", action="store_true", default=True)
    p.add_argument("--normalize_rewards", action="store_true", default=False)
    p.add_argument("--clip_actions", action="store_true", default=True)
    p.add_argument("--validate", action="store_true", default=True)
    # IQL hyperparams
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--value_lr", type=float, default=3e-4)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--expectile", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    full_ds = ManiFeelDataset(args.dataset_path, clip_actions=args.clip_actions)
    mode = full_ds.mode
    arch = args.backbone
    r3m_ckpt = args.r3m_checkpoint or None
    print(f"[INFO] mode={mode} backbone={arch} r3m_checkpoint={r3m_ckpt or '(none)'}")

    train_ds, test_ds = full_ds.train_test_split(test_ratio=args.test_ratio, seed=args.seed)
    print("=== Train ===")
    print(train_ds.summary())
    print("=== Test ===")
    print(test_ds.summary())
    if args.validate:
        print("--- Train validation ---")
        train_ds.validate()
        print("--- Test validation ---")
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
        arch=arch,
        mode=mode,
        r3m_checkpoint=r3m_ckpt,
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        temperature=args.temperature,
    )

    write_training_meta(
        args.save_dir,
        mode=mode,
        arch=arch,
        action_dim=action_dim,
        hidden_dims=tuple(args.hidden_dims),
        dataset_path=os.path.abspath(args.dataset_path),
        obs_example=obs_example,
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    writer = SummaryWriter(os.path.join(args.save_dir, "tb", str(args.seed)))

    print(f"[START] Training IQL for {args.max_steps:,} steps (batch={args.batch_size})")
    it = range(1, args.max_steps + 1)
    if args.tqdm:
        it = tqdm.tqdm(it, smoothing=0.1)

    for step in it:
        batch = train_ds.sample(args.batch_size)
        info = agent.update(batch)

        if step % args.log_interval == 0:
            writer.add_scalar("train/actor_loss", info.actor_loss, step)
            writer.add_scalar("train/critic_loss", info.critic_loss, step)
            writer.add_scalar("train/value_loss", info.value_loss, step)
            writer.add_scalar("train/q", info.q_mean, step)
            writer.add_scalar("train/v", info.v_mean, step)
            writer.add_scalar("train/adv", info.adv_mean, step)
            writer.add_scalar("train/backbone_grad_norm", info.backbone_grad_norm, step)
            writer.flush()

        if step % args.eval_interval == 0:
            # Note: current eval uses agent.update() for metric computation; keep eval_interval large.
            test_info = eval_on_dataset(agent, test_ds, args.batch_size, n_batches=5)
            for k, v in test_info.items():
                writer.add_scalar(f"test/{k}", v, step)
            writer.flush()

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(agent, args.save_dir, step)

    save_checkpoint(agent, args.save_dir, args.max_steps)
    writer.close()
    print(f"[DONE] Checkpoints in {args.save_dir}")


if __name__ == "__main__":
    main()
