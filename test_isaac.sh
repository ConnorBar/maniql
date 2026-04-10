#!/bin/bash

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate tacsl

export LD_LIBRARY_PATH=/home/${USER}/miniconda3/envs/tacsl/lib:\
/home/${USER}/capstone/IsaacGym_Preview_TacSL_Package/isaacgym/python/isaacgym/_bindings/linux-x86_64:\
${LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$ISAACGYM_PATH/_bindings/linux-x86_64:$LD_LIBRARY_PATH

cd IsaacGym_Preview_TacSL_Package/isaacgym/python/examples
python franka_osc.py
