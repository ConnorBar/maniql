"""Quick summary stats for raw transition files and/or preprocessed pickle."""

import argparse
import glob
import os
import pickle

import numpy as np


def inspect_raw(input_dir: str, pattern: str = "*_transitions.pkl"):
    raw_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not raw_files:
        print(f"No files matching '{pattern}' in {input_dir}")
        return

    print(f"\n{'='*70}")
    print(f"  RAW DATA: {len(raw_files)} files in {input_dir}")
    print(f"{'='*70}\n")

    per_file = []
    all_ep_lengths = []
    all_ep_returns = []
    all_rewards = []

    for i, fp in enumerate(raw_files):
        with open(fp, "rb") as f:
            data = pickle.load(f)
        transitions = data.get("transitions", data) if isinstance(data, dict) else data
        n = len(transitions)

        file_done = 0
        file_success = 0
        file_timeout = 0
        ep_len = 0
        ep_ret = 0.0

        for tr in transitions:
            r = float(np.asarray(tr["reward"]).ravel()[0])
            d = bool(np.asarray(tr["done"]).ravel()[0])
            s = bool(np.asarray(tr["success"]).ravel()[0])
            t = bool(np.asarray(tr["timeout"]).ravel()[0])

            all_rewards.append(r)
            ep_len += 1
            ep_ret += r

            file_done += int(d)
            file_success += int(s)
            file_timeout += int(t)

            if d:
                all_ep_lengths.append(ep_len)
                all_ep_returns.append(ep_ret)
                ep_len = 0
                ep_ret = 0.0

        if ep_len > 0:
            all_ep_lengths.append(ep_len)
            all_ep_returns.append(ep_ret)

        per_file.append({
            "file": os.path.basename(fp),
            "transitions": n,
            "done": file_done,
            "success": file_success,
            "timeout": file_timeout,
        })

    total_trans = sum(f["transitions"] for f in per_file)
    total_done = sum(f["done"] for f in per_file)
    total_success = sum(f["success"] for f in per_file)
    total_timeout = sum(f["timeout"] for f in per_file)
    n_eps = len(all_ep_lengths)
    rewards = np.array(all_rewards)
    ep_lens = np.array(all_ep_lengths)
    ep_rets = np.array(all_ep_returns)

    # --- Per-file table ---
    print(f"{'File':<45} {'Trans':>6} {'Done':>5} {'Succ':>5} {'T/O':>5} {'Eps':>5} ")
    print("-" * 80)
    for f in per_file:
        print(f"  {f['file']:<43} {f['transitions']:>6} {f['done']:>5} "
              f"{f['success']:>5} {f['timeout']:>5} {f['done']:>5}")
    print("-" * 80)
    print(f"  {'TOTAL':<43} {total_trans:>6} {total_done:>5} "
          f"{total_success:>5} {total_timeout:>5} {n_eps:>5}\n")

    # --- Proportions ---
    print("Termination breakdown (per transition with done=True):")
    if total_done > 0:
        n_fail = total_done - total_success - total_timeout
        print(f"  success:  {total_success:>6} / {total_done}  ({total_success/total_done:>7.2%})")
        print(f"  timeout:  {total_timeout:>6} / {total_done}  ({total_timeout/total_done:>7.2%})")
        print(f"  failure:  {n_fail:>6} / {total_done}  ({n_fail/total_done:>7.2%})  (done & !success & !timeout)")
    else:
        print("  No done=True transitions found.")

    # --- Episode stats ---
    print(f"\nEpisode stats ({n_eps} episodes):")
    print(f"  length  — mean: {ep_lens.mean():.1f}  median: {np.median(ep_lens):.0f}  "
          f"min: {ep_lens.min()}  max: {ep_lens.max()}  std: {ep_lens.std():.1f}")
    print(f"  return  — mean: {ep_rets.mean():.4f}  median: {np.median(ep_rets):.4f}  "
          f"min: {ep_rets.min():.4f}  max: {ep_rets.max():.4f}  std: {ep_rets.std():.4f}")

    # --- Reward stats ---
    print(f"\nReward stats ({total_trans:,} transitions):")
    print(f"  mean: {rewards.mean():.6f}  std: {rewards.std():.6f}")
    print(f"  min:  {rewards.min():.6f}  max: {rewards.max():.6f}")
    print(f"  percentiles: 5%={np.percentile(rewards,5):.6f}  25%={np.percentile(rewards,25):.6f}  "
          f"50%={np.percentile(rewards,50):.6f}  75%={np.percentile(rewards,75):.6f}  "
          f"95%={np.percentile(rewards,95):.6f}")

    print(f"\n  Reward histogram:")
    hist, edges = np.histogram(rewards, bins=15)
    max_bar = max(hist)
    for j in range(len(hist)):
        bar = "█" * int(40 * hist[j] / max_bar) if max_bar > 0 else ""
        print(f"    [{edges[j]:>8.4f}, {edges[j+1]:>8.4f})  {hist[j]:>6}  {bar}")


def inspect_preprocessed(pkl_path: str):
    print(f"\n{'='*70}")
    print(f"  PREPROCESSED: {pkl_path}")
    print(f"{'='*70}\n")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Top-level keys: {[k for k in data.keys() if k != 'metadata']}")
    if "metadata" in data:
        md = data["metadata"]
        print(f"\nMetadata:")
        for k, v in md.items():
            print(f"  {k}: {v}")

    if "obs" in data and isinstance(data["obs"], dict):
        print(f"\nobs (split dict) keys: {list(data['obs'].keys())}")
        for subk, arr in data["obs"].items():
            arr = np.asarray(arr)
            print(f"\n  obs['{subk}']:")
            print(f"    shape: {arr.shape}  dtype: {arr.dtype}")
            if arr.dtype in (np.float32, np.float64, np.int32, np.int64):
                print(f"    min: {arr.min():.6f}  max: {arr.max():.6f}  mean: {arr.mean():.6f}")
                if np.any(np.isnan(arr)):
                    print(f"    WARNING: {np.isnan(arr).sum()} NaN values!")
        print(f"\nnext_obs (split dict) keys: {list(data['next_obs'].keys())}")
        for subk, arr in data["next_obs"].items():
            arr = np.asarray(arr)
            print(f"\n  next_obs['{subk}']:")
            print(f"    shape: {arr.shape}  dtype: {arr.dtype}")

    for key in ["obs", "next_obs", "actions", "rewards", "dones", "terminals", "success", "timeouts"]:
        if key not in data or key in ("obs", "next_obs") and isinstance(data.get(key), dict):
            continue
        arr = np.asarray(data[key])
        print(f"\n{key}:")
        print(f"  shape: {arr.shape}  dtype: {arr.dtype}")
        if arr.dtype in (np.float32, np.float64, np.int32, np.int64):
            print(f"  min: {arr.min():.6f}  max: {arr.max():.6f}  mean: {arr.mean():.6f}")
            if arr.ndim == 1:
                n_nonzero = np.count_nonzero(arr)
                print(f"  nonzero: {n_nonzero} / {len(arr)} ({n_nonzero/len(arr):.2%})")
            if np.any(np.isnan(arr)):
                print(f"  WARNING: {np.isnan(arr).sum()} NaN values!")
            if np.any(np.isinf(arr)):
                print(f"  WARNING: {np.isinf(arr).sum()} Inf values!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ManiFeel data files.")
    parser.add_argument("--raw-dir", type=str, default="", help="Directory with raw *_transitions.pkl files.")
    parser.add_argument("--preprocessed", type=str, default="", help="Path to preprocessed pickle.")
    parser.add_argument("--pattern", type=str, default="*_transitions.pkl")
    args = parser.parse_args()

    if not args.raw_dir and not args.preprocessed:
        args.raw_dir = "data"
        args.preprocessed = "data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl"

    if args.raw_dir:
        inspect_raw(args.raw_dir, args.pattern)
    if args.preprocessed and os.path.isfile(args.preprocessed):
        inspect_preprocessed(args.preprocessed)
