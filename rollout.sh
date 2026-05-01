
#!/bin/bash

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate tacsl

export LD_LIBRARY_PATH=/home/${USER}/miniconda3/envs/tacsl/lib:\
/home/${USER}/capstone/IsaacGym_Preview_TacSL_Package/isaacgym/python/isaacgym/_bindings/linux-x86_64:\
${LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$ISAACGYM_PATH/_bindings/linux-x86_64:$LD_LIBRARY_PATH

python torch-maniql/rollout_watch_isaac.py \
    --save_dir ./runs/manifeel_iql \
    --task TacSLTaskBulb --once --record_video
