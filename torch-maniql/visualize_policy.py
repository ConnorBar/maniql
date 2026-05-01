"""Offline policy visualizer: compare trained policy actions vs demo actions.

Loads a checkpoint and the preprocessed dataset, runs the policy on demo
observations, and renders a video showing the wrist camera image alongside
predicted vs actual action bar charts.  Does NOT require IsaacGym.

Usage:
    python visualize_policy.py \
        --checkpoint_dir runs/manifeel_iql/checkpoint_50000 \
        --dataset_path data/preprocessed/raw_wrist_state.pkl \
        --output policy_vs_demo.mp4

    # Visualize a specific episode (0-indexed):
    python visualize_policy.py \
        --checkpoint_dir runs/manifeel_iql/checkpoint_50000 \
        --dataset_path data/preprocessed/raw_wrist_state.pkl \
        --episode 3 --output ep3.mp4

    # Grid of episodes (first frame from each):
    python visualize_policy.py \
        --checkpoint_dir runs/manifeel_iql/checkpoint_50000 \
        --dataset_path data/preprocessed/raw_wrist_state.pkl \
        --grid --output policy_grid.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# NumPy 2.x compat
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)

import torch
from PIL import Image, ImageDraw, ImageFont

from manifeel_iql import ManiFeelDataset
from multimodal_nets import MultiModalEncoder, PolicyHead, encoded_obs_dim


ACTION_LABELS = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]
ACTION_COLORS = [
    (255, 99, 132),
    (54, 162, 235),
    (255, 206, 86),
    (75, 192, 192),
    (153, 102, 255),
    (255, 159, 64),
    (200, 200, 200),
]


def parse_args():
    p = argparse.ArgumentParser(description="Offline policy visualizer")
    p.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint_N directory")
    p.add_argument("--dataset_path", required=True, help="Path to preprocessed .pkl")
    p.add_argument("--output", "-o", default="policy_vs_demo.mp4", help="Output video or image path")
    p.add_argument("--episode", type=int, default=0, help="Which episode to visualize (0-indexed)")
    p.add_argument("--fps", type=int, default=15, help="Video FPS")
    p.add_argument("--max_frames", type=int, default=None, help="Limit frames rendered")
    p.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    p.add_argument("--grid", action="store_true", default=False,
                   help="Render a grid of first-frame comparisons across episodes")
    p.add_argument("--grid_cols", type=int, default=5, help="Columns in grid mode")
    p.add_argument("--device", default="", help="Device (default: cuda if available)")
    p.add_argument("--backbone", default="", help="Override backbone arch (default: from training_meta)")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=None, help="Override hidden dims")
    return p.parse_args()


def load_checkpoint(checkpoint_dir: str, device: torch.device,
                    backbone: str = "", hidden_dims=None):
    meta_path = os.path.join(os.path.dirname(checkpoint_dir), "training_meta.json")
    if not os.path.isfile(meta_path):
        parent = os.path.dirname(os.path.dirname(checkpoint_dir))
        meta_path = os.path.join(parent, "training_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Cannot find training_meta.json near {checkpoint_dir}. "
            f"Expected at {os.path.dirname(checkpoint_dir)}/training_meta.json"
        )

    with open(meta_path) as f:
        meta = json.load(f)

    arch = backbone or meta["arch"]
    mode = meta["mode"]
    action_dim = int(meta["action_dim"])
    hdims = tuple(hidden_dims or meta["hidden_dims"])

    ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    payload = torch.load(ckpt_path, map_location="cpu")

    obs_dim = encoded_obs_dim(arch, mode)
    encoder = MultiModalEncoder(arch=arch, mode=mode, r3m_checkpoint=None).to(device)
    encoder.load_state_dict(payload["encoder"], strict=True)
    encoder.eval()

    actor = PolicyHead(obs_dim=obs_dim, hidden_dims=hdims, action_dim=action_dim).to(device)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()

    step = payload.get("step", -1)
    return encoder, actor, meta, step


def get_episode_indices(dataset: ManiFeelDataset):
    episodes = []
    start = 0
    for i in range(dataset.size):
        if dataset.dones_float[i] == 1.0:
            episodes.append((start, i + 1))
            start = i + 1
    if start < dataset.size:
        episodes.append((start, dataset.size))
    return episodes


def run_policy_on_episode(encoder, actor, dataset, ep_start, ep_end,
                          device, deterministic=True):
    pred_actions = []
    demo_actions = []
    wrist_images = []

    for idx in range(ep_start, ep_end):
        obs = {k: dataset._obs[k][idx:idx+1] for k in dataset._split_keys}
        tobs = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}

        with torch.no_grad():
            encoded = encoder(tobs)
            action = actor.act(encoded, deterministic=deterministic)
            pred = action.detach().cpu().numpy().reshape(-1)

        pred_actions.append(pred)
        demo_actions.append(dataset.actions[idx])
        wrist_images.append(dataset._obs["wrist"][idx])

    return np.array(pred_actions), np.array(demo_actions), wrist_images


def _get_font(size=12):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
        except Exception:
            return ImageFont.load_default()


def draw_action_bars(pred: np.ndarray, demo: np.ndarray, width: int, height: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (25, 25, 30))
    draw = ImageDraw.Draw(img)
    font = _get_font(11)
    small_font = _get_font(9)

    n_dims = len(pred)
    bar_area_top = 22
    bar_area_bot = height - 8
    bar_area_h = bar_area_bot - bar_area_top
    group_w = width / n_dims
    bar_w = max(4, int(group_w * 0.30))
    gap = max(2, int(group_w * 0.06))

    draw.text((4, 2), "Predicted vs Demo Actions", font=font, fill=(200, 200, 200))

    mid_y = bar_area_top + bar_area_h // 2
    draw.line([(0, mid_y), (width, mid_y)], fill=(80, 80, 80), width=1)

    for d in range(n_dims):
        cx = int((d + 0.5) * group_w)

        def val_to_y(v):
            clamped = max(-1.0, min(1.0, float(v)))
            return int(mid_y - clamped * (bar_area_h // 2 - 4))

        # Demo bar (left, dimmer)
        dy = val_to_y(demo[d])
        x0 = cx - bar_w - gap // 2
        top_y, bot_y = min(dy, mid_y), max(dy, mid_y)
        color_dim = tuple(max(0, c - 100) for c in ACTION_COLORS[d % len(ACTION_COLORS)])
        draw.rectangle([(x0, top_y), (x0 + bar_w, bot_y)], fill=color_dim)

        # Predicted bar (right, bright)
        py = val_to_y(pred[d])
        x1 = cx + gap // 2
        top_y, bot_y = min(py, mid_y), max(py, mid_y)
        draw.rectangle([(x1, top_y), (x1 + bar_w, bot_y)], fill=ACTION_COLORS[d % len(ACTION_COLORS)])

        label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f"d{d}"
        tw = draw.textlength(label, font=small_font)
        draw.text((cx - tw / 2, bar_area_bot - 2), label, font=small_font, fill=(160, 160, 160))

    # Legend
    draw.rectangle([(width - 145, 2), (width - 135, 12)], fill=(100, 100, 100))
    draw.text((width - 132, 1), "demo", font=small_font, fill=(160, 160, 160))
    draw.rectangle([(width - 80, 2), (width - 70, 12)], fill=(255, 159, 64))
    draw.text((width - 67, 1), "pred", font=small_font, fill=(160, 160, 160))

    return np.array(img)


def draw_timeline(step: int, total: int, pred_actions: np.ndarray,
                  demo_actions: np.ndarray, rewards: np.ndarray,
                  width: int, height: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = _get_font(10)

    row_h = height // 3
    n = total

    def t_to_x(t):
        return int(t / max(1, n - 1) * (width - 1)) if n > 1 else width // 2

    # Row 1: action MSE over time
    if len(pred_actions) > 0:
        mse = np.mean((pred_actions[:len(demo_actions)] - demo_actions) ** 2, axis=1)
        mse_max = max(float(mse.max()), 1e-6)
        pts = []
        for t in range(len(mse)):
            x = t_to_x(t)
            y = int(row_h - 4 - (mse[t] / mse_max) * (row_h - 12))
            pts.append((x, max(8, y)))
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 100, 100), width=2)
        draw.text((4, 1), f"Action MSE (max={mse_max:.4f})", font=font, fill=(200, 200, 200))

    # Row 2: reward
    r_min, r_max = float(rewards.min()), float(rewards.max())
    if abs(r_max - r_min) < 1e-8:
        r_max = r_min + 1.0
    pts = []
    for t in range(n):
        x = t_to_x(t)
        alpha = (rewards[t] - r_min) / (r_max - r_min)
        y = int(row_h * 2 - 4 - alpha * (row_h - 12))
        pts.append((x, max(row_h + 8, y)))
    if len(pts) >= 2:
        draw.line(pts, fill=(80, 200, 120), width=2)
    draw.text((4, row_h + 1), "Reward", font=font, fill=(200, 200, 200))

    # Row 3: per-dim action comparison (predicted = solid, demo = dashed)
    for d in range(min(7, pred_actions.shape[1] if len(pred_actions) > 0 else 0)):
        color = ACTION_COLORS[d % len(ACTION_COLORS)]
        pts_pred, pts_demo = [], []
        for t in range(min(len(pred_actions), n)):
            x = t_to_x(t)
            def v_to_y(v):
                return int(row_h * 3 - 4 - (float(v) + 1.0) / 2.0 * (row_h - 12))
            pts_pred.append((x, max(row_h * 2 + 8, v_to_y(pred_actions[t, d]))))
            pts_demo.append((x, max(row_h * 2 + 8, v_to_y(demo_actions[t, d]))))
        if len(pts_pred) >= 2:
            draw.line(pts_pred, fill=color, width=1)
        dim_color = tuple(max(0, c - 120) for c in color)
        if len(pts_demo) >= 2:
            draw.line(pts_demo, fill=dim_color, width=1)
    draw.text((4, row_h * 2 + 1), "Actions (bright=pred, dim=demo)", font=font, fill=(200, 200, 200))

    # Cursor
    cx = t_to_x(step)
    draw.line([(cx, 0), (cx, height)], fill=(255, 255, 0), width=2)

    # Step counter
    draw.text((width - 120, 1), f"step {step+1}/{n}", font=font, fill=(220, 220, 220))

    return np.array(img)


def render_episode_video(encoder, actor, dataset, ep_idx, episodes,
                         device, output, fps, max_frames, deterministic):
    ep_start, ep_end = episodes[ep_idx]
    ep_len = ep_end - ep_start
    print(f"Episode {ep_idx}: transitions [{ep_start}, {ep_end}), length={ep_len}")

    print("Running policy inference...")
    pred_actions, demo_actions, wrist_images = run_policy_on_episode(
        encoder, actor, dataset, ep_start, ep_end, device, deterministic)

    rewards = dataset.rewards[ep_start:ep_end]
    mse = np.mean((pred_actions - demo_actions) ** 2, axis=1)
    print(f"Action MSE: mean={mse.mean():.6f}, max={mse.max():.6f}, min={mse.min():.6f}")

    n_frames = min(ep_len, max_frames) if max_frames else ep_len
    frame_w = 224 + 320
    bar_h = 224
    timeline_h = 192

    import imageio.v2 as imageio
    writer = imageio.get_writer(output, fps=fps)
    print(f"Rendering {n_frames} frames to {output} at {fps} fps...")

    try:
        for i in range(n_frames):
            wrist = wrist_images[i]
            if wrist.dtype != np.uint8:
                wrist = (wrist * 255).clip(0, 255).astype(np.uint8)
            wrist_pil = Image.fromarray(wrist).resize((224, 224))
            wrist_np = np.array(wrist_pil)

            bars = draw_action_bars(pred_actions[i], demo_actions[i], 320, bar_h)
            top = np.hstack([wrist_np, bars])
            timeline = draw_timeline(i, ep_len, pred_actions[:i+1], demo_actions[:i+1],
                                     rewards, top.shape[1], timeline_h)
            frame = np.vstack([top, timeline])
            writer.append_data(frame)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_frames} frames...")
    finally:
        writer.close()

    print(f"Done. Saved to {output}")
    print(f"  Episode return: {rewards.sum():.4f}")
    print(f"  Mean action MSE: {mse.mean():.6f}")


def render_grid(encoder, actor, dataset, episodes, device, output,
                grid_cols, deterministic):
    n_eps = len(episodes)
    print(f"Rendering grid for {n_eps} episodes...")

    cells = []
    cell_w, cell_h = 280, 180

    for ep_idx in range(n_eps):
        ep_start, ep_end = episodes[ep_idx]
        mid = (ep_start + ep_end) // 2

        obs = {k: dataset._obs[k][mid:mid+1] for k in dataset._split_keys}
        tobs = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
        with torch.no_grad():
            encoded = encoder(tobs)
            pred = actor.act(encoded, deterministic=deterministic).cpu().numpy().reshape(-1)
        demo = dataset.actions[mid]

        wrist = dataset._obs["wrist"][mid]
        if wrist.dtype != np.uint8:
            wrist = (wrist * 255).clip(0, 255).astype(np.uint8)
        wrist_pil = Image.fromarray(wrist).resize((cell_w // 2, cell_h))
        wrist_np = np.array(wrist_pil)

        bars = draw_action_bars(pred, demo, cell_w - cell_w // 2, cell_h)
        cell = np.hstack([wrist_np, bars])
        cells.append(cell)

    n_rows = (n_eps + grid_cols - 1) // grid_cols
    while len(cells) < n_rows * grid_cols:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    rows = []
    for r in range(n_rows):
        row = np.hstack(cells[r * grid_cols:(r + 1) * grid_cols])
        rows.append(row)
    grid = np.vstack(rows)

    Image.fromarray(grid).save(output)
    print(f"Saved grid to {output} ({grid.shape[1]}x{grid.shape[0]})")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    encoder, actor, meta, ckpt_step = load_checkpoint(
        args.checkpoint_dir, device, args.backbone, args.hidden_dims)
    print(f"  arch={meta['arch']} mode={meta['mode']} action_dim={meta['action_dim']} step={ckpt_step}")

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = ManiFeelDataset(args.dataset_path, clip_actions=True)
    print(f"  {dataset.size:,} transitions")

    episodes = get_episode_indices(dataset)
    print(f"  {len(episodes)} episodes")

    if args.grid:
        render_grid(encoder, actor, dataset, episodes, device,
                    args.output, args.grid_cols, args.deterministic)
    else:
        if args.episode >= len(episodes):
            print(f"Episode {args.episode} out of range (0-{len(episodes)-1})")
            sys.exit(1)
        render_episode_video(encoder, actor, dataset, args.episode, episodes,
                             device, args.output, args.fps, args.max_frames,
                             args.deterministic)


if __name__ == "__main__":
    main()
