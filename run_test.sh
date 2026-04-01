conda activate tacsl

export LD_LIBRARY_PATH=/home/${USER}/miniconda3/envs/tacsl/lib:\
/home/${USER}/cap/IsaacGym_Preview_TacSL_Package/isaacgym/python/isaacgym/_bindings/linux-x86_64:\
${LD_LIBRARY_PATH}

cd IsaacGym_Preview_TacSL_Package/isaacgym/python/examples
python franka_osc.py
