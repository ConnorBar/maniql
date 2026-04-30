#!/usr/bin/env bash
# Build and run IQL training in Docker.
#
# Usage:
#   ./docker-run.sh --dataset_path /data/your_dataset [extra train_iql.py args]
#
# Env vars:
#   WANDB_API_KEY  — set this before running if using --wandb
#   DATA_DIR       — host path to dataset dir (default: ../data)
#   RUNS_DIR       — host path to save checkpoints (default: ../runs)

set -e

IMAGE="maniql-train"
DATA_DIR="${DATA_DIR:-$(realpath ../data)}"
RUNS_DIR="${RUNS_DIR:-$(realpath ../runs)}"

cd "$(dirname "$0")"

echo "Building image..."
docker build -t "$IMAGE" .

mkdir -p "$RUNS_DIR"

echo "Launching training..."
docker run --rm --gpus all \
    -v "${DATA_DIR}:/data:ro" \
    -v "${RUNS_DIR}:/runs" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    "$IMAGE" \
    --save_dir /runs/manifeel_iql \
    "$@"
