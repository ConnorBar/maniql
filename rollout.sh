#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tacsl

# Isaac Gym needs libpython and its own bindings on LD_LIBRARY_PATH.
# Use $CONDA_PREFIX so it works on any machine (vast.ai, lab, local).
ISAAC_DIR="${SCRIPT_DIR}/IsaacGym_Preview_TacSL_Package"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${ISAAC_DIR}/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

cd "${SCRIPT_DIR}"
#xvfb-run -a python torch-maniql/rollout_watch_isaac.py \
#      --save_dir ./runs/manifeel_iql \
#      --task TacSLTaskBulb \
#      --once

xvfb-run -a python torch-maniql/rollout_watch_isaac.py \
      --save_dir ./runs/manifeel_iql \
      --task TacSLTaskBulb \
      --once \
      --num_envs 1 \
      --graphics_device_id 0
