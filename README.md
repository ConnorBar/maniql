# ManiQL: TacSL Multimodal Implicit Q-Learning

This repository contains the implementation of ManiQL, a multimodal implicit Q-learning framework for TacSL (Tactile Sensing for Learning) robotic manipulation tasks. The project integrates visual (wrist camera) and tactile modalities using R3M (Representation for Robotic Manipulation) for visual encoding and Implicit Q-Learning (IQL) algorithms.

## Project Structure

- `maniql/`: Main implementation directory
  - `implicit_q_learning/`: Reference IQL implementation (adapted for multimodal use)
  - `r3m/`: Reference R3M model for visual representation learning
  - `train_iql.py`: Training script for IQL
  - `manifeel_iql.py`: ManiQL-specific IQL implementation
  - `multimodal_nets.py`: Neural networks for multimodal processing
  - `obs_modality.py`: Observation modality handling
  - `seed_data.py`: Data seeding and preprocessing utilities
  - `visualize_transitions.py`: Visualization tools
  - `inspect_data.py`: Data inspection scripts

- `manifeel-isaacgymenvs/`: Cloned Isaac Gym environments for tactile manipulation
- `IsaacGym_Preview_TacSL_Package/`: TacSL-specific Isaac Gym package (downloaded separately)
- `data/`: Dataset storage
- `runs/`: Experiment outputs and logs

## Setup

1. **Prerequisites**: Ensure Conda is installed.

2. **Create Conda Environment**:
   ```bash
   conda create --name tacsl python=3.8 -y
   conda activate tacsl
   ```

3. **Download Isaac Gym Package**:
   - Download the TacSL-specific Isaac Gym from: [Purdue SharePoint Link](https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX)
   - Place the `IsaacGym_Preview_TacSL_Package` folder in the root directory.

4. **Run Setup Script**:
   ```bash
   ./setup.sh
   ```
   This will:
   - Install Isaac Gym
   - Clone and set up `manifeel-isaacgymenvs`
   - Install dependencies (including patching warp-lang to 0.11.0)
   - Install additional TacSL sensor requirements

## Data Preprocessing

Use `seed_data.py` to preprocess datasets with R3M encoding for wrist images and other modalities.

### Example: Preprocess wrist and state only
```bash
python maniql/seed_data.py \
  --input-dir data \
  --output data/preprocessed/all_transitions_r3m_wrist_state.pkl \
  --features wrist,state \
  --wrist-encoder r3m \
  --wrist-model resnet50 \
  --device cuda
```

### Example: Preprocess wrist, tactile, forcefield, and state
```bash
python maniql/seed_data.py \
  --input-dir data \
  --output data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl \
  --features wrist,tactile,forcefield,state \
  --wrist-encoder r3m \
  --wrist-model resnet18 \
  --device cuda
```

Note: Tactile and forcefield modalities are processed using CNN and MLP respectively, trained with standard IQL loss.

## Training

Train the ManiQL model using `train_iql.py`.

### Basic Training
```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --save_dir runs/manifeel_iql_multimodal \
  --eval_episodes 10 \
  --normalize_rewards True
```

### Full Training Configuration
```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --save_dir runs/manifeel_iql_r3m18_wtfs \
  --max_steps 200000 \
  --batch_size 128 \
  --eval_interval 2000 \
  --log_interval 1000 \
  --save_interval 50000 \
  --validate True \
  --clip_actions True \
  --normalize_rewards True
```

### Debug Training (Small Steps)
```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --save_dir runs/debug_small \
  --max_steps 1000 \
  --batch_size 32 \
  --validate True \
  --normalize_rewards False
```

## Evaluation

Use the provided scripts in `maniql/` for evaluation and visualization, such as `visualize_transitions.py` and `inspect_data.py`.

## References

- **Implicit Q-Learning**: Adapted from the original IQL implementation.
- **R3M**: Used for visual representation learning.
- **Isaac Gym**: Environments cloned from NVIDIA Isaac Gym for simulation.
- **TacSL**: Tactile Sensing for Learning, integrated via `manifeel-isaacgymenvs`.


pip install tqdm ml-collections jax "jax[cuda12]" 

pip install jaxlib==0.4.13+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


