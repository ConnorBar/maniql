import argparse
import os
import pickle

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.array(img)
    # Expect shape (1, H, W, C); drop batch dim if present
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    if img.dtype != np.uint8:
        maxv = float(img.max())
        # If in [0, 1], scale to [0, 255]
        if maxv <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def make_combined_frame(wrist: np.ndarray, taxim: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    Create a horizontal, movie-like frame where we:
    - label each view separately (like two divs in HTML),
    - then put them side by side,
    - and optionally upscale.
    """
    wrist_u8 = to_uint8(wrist)   # (H1, W1, 3)
    taxim_u8 = to_uint8(taxim)   # (H2, W2, 3)

    h_w, w_w, _ = wrist_u8.shape
    h_t, w_t, _ = taxim_u8.shape

    # Match heights so they tile cleanly.
    out_h = min(h_w, h_t)

    def center_crop_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, _, _ = img.shape
        if h == target_h:
            return img
        start = (h - target_h) // 2
        end = start + target_h
        return img[start:end, :, :]

    wrist_c = center_crop_height(wrist_u8, out_h)
    taxim_c = center_crop_height(taxim_u8, out_h)

    # Draw subtle labels on each half independently (no chance of overlap).
    def label_single(img_np: np.ndarray, text: str) -> np.ndarray:
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        w, h = img_pil.size

        # Smaller, unobtrusive font.
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(10, h // 30))
        except Exception:
            font = ImageFont.load_default()

        margin_y = max(4, h // 60)
        margin_x = max(6, w // 100)

        x = margin_x
        y = margin_y

        # Light text with a very thin shadow for readability, but no big box.
        shadow_color = (0, 0, 0)
        text_color = (255, 255, 255)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), text, font=font, fill=shadow_color)
        draw.text((x, y), text, font=font, fill=text_color)

        return np.array(img_pil)

    wrist_labeled = label_single(wrist_c, "Wrist camera")
    taxim_labeled = label_single(taxim_c, "Right tactile camera (TAXIM)")

    frame = np.hstack([wrist_labeled, taxim_labeled])

    # Optional upscale for nicer viewing/export.
    if scale != 1.0:
        new_w = int(frame.shape[1] * scale)
        new_h = int(frame.shape[0] * scale)
        frame = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC))

    return frame


def extract_series(transitions):
    """Precompute time-series arrays (reward, done, success, actions)."""
    n = len(transitions)
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.int32)
    successes = np.zeros(n, dtype=np.int32)
    timeouts = np.zeros(n, dtype=np.int32)
    actions = np.zeros((n, 7), dtype=np.float32)

    for i, tr in enumerate(transitions):
        rewards[i] = float(np.asarray(tr["reward"]).reshape(-1)[0])
        dones[i] = int(np.asarray(tr["done"]).reshape(-1)[0])
        successes[i] = int(np.asarray(tr["success"]).reshape(-1)[0])
        timeouts[i] = int(np.asarray(tr["timeout"]).reshape(-1)[0])
        actions[i] = np.asarray(tr["action"]).reshape(-1)

    return {
        "rewards": rewards,
        "dones": dones,
        "successes": successes,
        "timeouts": timeouts,
        "actions": actions,
    }


def _series_bounds(arr: np.ndarray, pad: float = 0.05):
    vmin = float(arr.min())
    vmax = float(arr.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1.0, 1.0
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0
    span = vmax - vmin
    return vmin - pad * span, vmax + pad * span


def make_plot_panel(step_idx: int, series, width: int) -> np.ndarray:
    """
    Build a bottom panel with:
    - reward timeline (+ success/done markers),
    - 7D action curves,
    - step counter text.
    """
    rewards = series["rewards"]
    dones = series["dones"]
    successes = series["successes"]
    timeouts = series["timeouts"]
    actions = series["actions"]

    n = len(rewards)
    width = int(width)
    plot_h = max(180, width // 5)
    row_h = plot_h // 2

    panel = Image.new("RGB", (width, plot_h), (20, 20, 20))
    draw = ImageDraw.Draw(panel)

    def t_to_x(t: int) -> int:
        if n <= 1:
            return width // 2
        return int(t / (n - 1) * (width - 1))

    # Reward row
    y0_r = 0
    y1_r = row_h - 1
    r_min, r_max = _series_bounds(rewards)

    def norm_y(val, vmin, vmax, y_top, y_bot):
        if vmax <= vmin:
            return (y_top + y_bot) // 2
        alpha = (val - vmin) / (vmax - vmin)
        alpha = max(0.0, min(1.0, float(alpha)))
        return int(y_bot - alpha * (y_bot - y_top))

    # Reserve some vertical space at the very top of this row for labels.
    text_band_h = max(14, row_h // 5)

    pts_reward = []
    for t in range(n):
        x = t_to_x(t)
        # Start a bit lower so we don't cover the text band.
        y = norm_y(rewards[t], r_min, r_max, y0_r + text_band_h, y1_r - 6)
        pts_reward.append((x, y))
    if len(pts_reward) >= 2:
        draw.line(pts_reward, fill=(80, 200, 120), width=2)

    # Success/done/timeout markers as a thin strip at bottom of reward row.
    for t in range(n):
        x = t_to_x(t)
        if successes[t]:
            color = (0, 220, 0)  # green for success
        elif timeouts[t]:
            color = (0, 180, 220)  # blue for timeout
        elif dones[t]:
            color = (220, 80, 80)  # red for done
        else:
            continue
        draw.line([(x, y1_r - 3), (x, y1_r)], fill=color, width=1)

    # Actions row
    y0_a = row_h
    y1_a = plot_h - 1
    a_min, a_max = _series_bounds(actions)
    action_colors = [
        (255, 99, 132),
        (54, 162, 235),
        (255, 206, 86),
        (75, 192, 192),
        (153, 102, 255),
        (255, 159, 64),
        (200, 200, 200),
    ]
    for j in range(actions.shape[1]):
        pts = []
        for t in range(n):
            x = t_to_x(t)
            y = norm_y(actions[t, j], a_min, a_max, y0_a + 4, y1_a - 4)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=action_colors[j % len(action_colors)], width=1)

    # Vertical time cursor across both rows.
    x_cur = t_to_x(step_idx)
    draw.line([(x_cur, 0), (x_cur, plot_h - 1)], fill=(255, 255, 0), width=2)

    # Small text for axis labels.
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=10)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 2), "Reward (green)", font=font, fill=(220, 220, 220))
    draw.text((6, y0_a + 2), "Actions (7D)", font=font, fill=(220, 220, 220))

    # Step counter
    draw.text(
        (width - 160, 2),
        f"step {step_idx + 1}/{n}",
        font=font,
        fill=(220, 220, 220),
    )

    return np.array(panel)


def load_transitions(path: str):
    with open(path, "rb") as f:
        transitions = pickle.load(f)
    if not isinstance(transitions, list):
        raise ValueError(f"Expected list of transitions, got {type(transitions)}")
    return transitions


def main():
    parser = argparse.ArgumentParser(
        description="Visualize wrist and taxim observations from transitions.pkl as a video."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to *_transitions.pkl file (e.g. data/2026-02-10-13-50-16_transitions.pkl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="demo_wrist_taxim.mp4",
        help="Output video file (MP4).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames (for quick tests).",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    transitions = load_transitions(args.input)
    n = len(transitions)
    print(f"Loaded {n} transitions from {args.input}")

    max_frames = args.max_frames if args.max_frames is not None else n

    # Precompute time-series info once for all frames.
    series = extract_series(transitions)

    print(f"Writing up to {max_frames} frames to {args.output} at {args.fps} fps ...")
    writer = imageio.get_writer(args.output, fps=args.fps)

    try:
        for idx, tr in enumerate(transitions[:max_frames]):
            obs = tr["obs"]
            wrist = obs["wrist"]  # shape (1, 256, 256, 3)
            taxim = obs["right_tactile_camera_taxim"]  # shape (1, 160, 120, 3)
            cams_frame = make_combined_frame(wrist, taxim)
            plot_panel = make_plot_panel(idx, series, width=cams_frame.shape[1])

            frame = np.vstack([cams_frame, plot_panel])
            writer.append_data(frame)

            if (idx + 1) % 100 == 0:
                print(f"  wrote {idx + 1} frames ...")

    finally:
        writer.close()

    print(f"Done. Saved video to: {args.output}")


if __name__ == "__main__":
    main()