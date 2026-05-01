"""Post-hoc action normalization on an already-preprocessed pickle.

Normalizes each action dimension to [-1, 1] by dividing by its max absolute
value.  Dimensions already within [-1, 1] are left untouched.  Overwrites the
pickle in-place (pass --output to write to a new file instead).

Usage:
    python fix_actions.py data/preprocessed/raw_wrist_state.pkl
    python fix_actions.py data/preprocessed/raw_wrist_state.pkl --output data/preprocessed/raw_wrist_state_fixed.pkl
"""

import argparse
import pickle
import sys

import numpy as np

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pkl", help="Path to preprocessed pickle.")
    p.add_argument("--output", default="", help="Output path (default: overwrite in-place).")
    args = p.parse_args()

    print(f"[INFO] Loading {args.pkl}")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    actions = data["actions"]
    n, d = actions.shape
    print(f"[INFO] {n:,} transitions, {d} action dims")

    print("\n  Before normalization:")
    for i in range(d):
        col = actions[:, i]
        print(f"    dim {i}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}")

    scales = np.ones(d, dtype=np.float32)
    for i in range(d):
        max_abs = np.max(np.abs(actions[:, i]))
        if max_abs > 1.0:
            scales[i] = max_abs

    if (scales == 1.0).all():
        print("\n[INFO] All dimensions already in [-1, 1]. Nothing to do.")
        return

    data["actions"] = actions / scales[None, :]

    print("\n  After normalization:")
    for i in range(d):
        col = data["actions"][:, i]
        tag = f" (scaled by {scales[i]:.4f})" if scales[i] > 1.0 else ""
        print(f"    dim {i}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}{tag}")

    if "metadata" in data and isinstance(data["metadata"], dict):
        data["metadata"]["action_normalization"] = "per_dim_max_abs"
        data["metadata"]["action_scales"] = scales.tolist()

    out_path = args.output or args.pkl
    print(f"\n[INFO] Writing {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[DONE]")


if __name__ == "__main__":
    main()
