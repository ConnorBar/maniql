# ManiQL: TacSL Multimodal Implicit Q-Learning

This repository contains the implementation of ManiQL, a multimodal implicit Q-learning framework for TacSL (Tactile Sensing for Learning) robotic manipulation tasks. The current pipeline stores wrist and tactile images as raw observations and trains Flax ResNet vision backbones end-to-end during IQL, with optional R3M checkpoint initialization.

## Project Structure

- `maniql/`: Main implementation directory
  - `implicit_q_learning/`: Reference IQL implementation (adapted for multimodal use)
  - `vision_backbone.py`: Flax ResNet backbone and R3M weight-loading utilities
  - `r3m/`: Reference R3M code and checkpoints for optional pretrained initialization
  - `train_iql.py`: Training script for IQL
  - `manifeel_iql.py`: Dataset loading and ManiQL-specific IQL integration
  - `multimodal_nets.py`: Multimodal actor/critic/value networks with shared modality handling
  - `obs_modality.py`: Observation schemas for `wrist_state` and `full` modes
  - `seed_data.py`: Raw-data preprocessing into IQL-ready pickles
  - `visualize_transitions.py`: Visualization tools
  - `inspect_data.py`: Data inspection for the new raw-observation schemas

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

Use `seed_data.py` to preprocess transition files into raw-observation pickles. Images are stored as `uint8` tensors and encoded during training so the vision backbone can be finetuned end-to-end.

Two preprocessing modes are supported:
- `wrist_state`: wrist image + robot state
- `full`: wrist image + tactile image + force vector + robot state

### Example: Preprocess wrist and state only
```bash
python maniql/seed_data.py \
  --input-dir data \
  --mode wrist_state \
  --output data/preprocessed/raw_wrist_state.pkl
```

### Example: Preprocess wrist, tactile, forcefield, and state
```bash
python maniql/seed_data.py \
  --input-dir data \
  --mode full \
  --output data/preprocessed/raw_full.pkl
```

## Training

Train the ManiQL model using `train_iql.py`. The `--backbone` flag selects the ResNet variant (`resnet18`, `resnet34`, or `resnet50`). Use `--r3m_checkpoint` to initialize the Flax backbone from pretrained R3M weights, or leave it empty to train from scratch.

### Wrist + state
```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_wrist_state.pkl \
  --backbone resnet18 \
  --r3m_checkpoint ~/.r3m/r3m_18/model.pt \
  --save_dir runs/iql_ws_r3m18 \
  --max_steps 200000 \
  --batch_size 128 \
  --normalize_rewards True
```

### Full multimodal
```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_full.pkl \
  --backbone resnet18 \
  --r3m_checkpoint ~/.r3m/r3m_18/model.pt \
  --save_dir runs/iql_full_r3m18 \
  --max_steps 200000 \
  --batch_size 64 \
  --normalize_rewards True
```

### Full Training Configuration
```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_full.pkl \
  --backbone resnet18 \
  --r3m_checkpoint ~/.r3m/r3m_18/model.pt \
  --save_dir runs/iql_full_r3m18 \
  --max_steps 200000 \
  --batch_size 64 \
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
  --dataset_path data/preprocessed/raw_wrist_state.pkl \
  --backbone resnet18 \
  --save_dir runs/debug_small \
  --max_steps 1000 \
  --batch_size 32 \
  --validate True
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

