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

    for fp in raw_files:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        transitions = (data.get("transitions", data)
                       if isinstance(data, dict) else data)
        n = len(transitions)

        file_done = file_success = file_timeout = 0
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

    print(f"{'File':<45} {'Trans':>6} {'Done':>5} "
          f"{'Succ':>5} {'T/O':>5} {'Eps':>5}")
    print("-" * 80)
    for f in per_file:
        print(f"  {f['file']:<43} {f['transitions']:>6} "
              f"{f['done']:>5} {f['success']:>5} "
              f"{f['timeout']:>5} {f['done']:>5}")
    print("-" * 80)
    print(f"  {'TOTAL':<43} {total_trans:>6} {total_done:>5} "
          f"{total_success:>5} {total_timeout:>5} {n_eps:>5}\n")

    if total_done > 0:
        n_fail = total_done - total_success - total_timeout
        print("Termination breakdown:")
        print(f"  success: {total_success:>6} / {total_done} "
              f"({total_success/total_done:>7.2%})")
        print(f"  timeout: {total_timeout:>6} / {total_done} "
              f"({total_timeout/total_done:>7.2%})")
        print(f"  failure: {n_fail:>6} / {total_done} "
              f"({n_fail/total_done:>7.2%})")

    print(f"\nEpisode stats ({n_eps} episodes):")
    print(f"  length — mean: {ep_lens.mean():.1f}  "
          f"median: {np.median(ep_lens):.0f}  "
          f"min: {ep_lens.min()}  max: {ep_lens.max()}")
    print(f"  return — mean: {ep_rets.mean():.4f}  "
          f"median: {np.median(ep_rets):.4f}  "
          f"min: {ep_rets.min():.4f}  max: {ep_rets.max():.4f}")

    print(f"\nReward stats ({total_trans:,} transitions):")
    print(f"  mean: {rewards.mean():.6f}  std: {rewards.std():.6f}")
    print(f"  min:  {rewards.min():.6f}  max: {rewards.max():.6f}")


def inspect_preprocessed(pkl_path: str):
    print(f"\n{'='*70}")
    print(f"  PREPROCESSED: {pkl_path}")
    print(f"{'='*70}\n")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Top-level keys: {[k for k in data if k != 'metadata']}")
    if "metadata" in data:
        print("\nMetadata:")
        for k, v in data["metadata"].items():
            print(f"  {k}: {v}")

    if "obs" in data and isinstance(data["obs"], dict):
        print(f"\nobs keys: {list(data['obs'].keys())}")
        for subk, arr in data["obs"].items():
            arr = np.asarray(arr)
            print(f"\n  obs['{subk}']:")
            print(f"    shape: {arr.shape}  dtype: {arr.dtype}")
            if arr.dtype in (np.float32, np.float64):
                print(f"    min: {arr.min():.6f}  max: {arr.max():.6f}  "
                      f"mean: {arr.mean():.6f}")
                if np.any(np.isnan(arr)):
                    print(f"    WARNING: {np.isnan(arr).sum()} NaN values!")
            elif arr.dtype == np.uint8:
                print(f"    min: {arr.min()}  max: {arr.max()}")

        print(f"\nnext_obs keys: {list(data['next_obs'].keys())}")
        for subk, arr in data["next_obs"].items():
            arr = np.asarray(arr)
            print(f"\n  next_obs['{subk}']:")
            print(f"    shape: {arr.shape}  dtype: {arr.dtype}")

    for key in ["actions", "rewards", "dones", "terminals", "success",
                "timeouts"]:
        if key not in data:
            continue
        if key in ("obs", "next_obs") and isinstance(data.get(key), dict):
            continue
        arr = np.asarray(data[key])
        print(f"\n{key}:")
        print(f"  shape: {arr.shape}  dtype: {arr.dtype}")
        if arr.dtype in (np.float32, np.float64, np.int32, np.int64):
            print(f"  min: {arr.min():.6f}  max: {arr.max():.6f}  "
                  f"mean: {arr.mean():.6f}")
            if arr.ndim == 1:
                nz = np.count_nonzero(arr)
                print(f"  nonzero: {nz} / {len(arr)} ({nz/len(arr):.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ManiFeel data.")
    parser.add_argument("--raw-dir", type=str, default="")
    parser.add_argument("--preprocessed", type=str, default="")
    parser.add_argument("--pattern", type=str, default="*_transitions.pkl")
    args = parser.parse_args()

    if not args.raw_dir and not args.preprocessed:
        args.raw_dir = "data"

    if args.raw_dir:
        inspect_raw(args.raw_dir, args.pattern)
    if args.preprocessed and os.path.isfile(args.preprocessed):
        inspect_preprocessed(args.preprocessed)
