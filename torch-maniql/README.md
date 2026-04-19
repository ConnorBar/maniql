## Data Preprocessing

Images are stored **raw** (uint8, 224x224x3). R3M encoding happens
during training so the backbone is finetuned end-to-end with IQL loss.

### Wrist + State only

```bash
python seed_data.py \
  --input-dir /home/connors/capstone/data \
  --output /home/connors/capstone/data/preprocessed/raw_wrist_state.pkl \
  --mode wrist_state
```

### Full multimodal (wrist + tactile + force + state)

```bash
python seed_data.py \
  --input-dir /home/connors/capstone/data \
  --output /home/connors/capstone/data/preprocessed/raw_full.pkl \
  --mode full
```

## Training

The `--backbone` flag selects the ResNet variant (resnet18, resnet34, resnet50).
`--r3m_checkpoint` loads pretrained R3M weights; omit to train from scratch.

If you need to download the weights for the first time (stored in `/home/.r3m`):
```bash
cd maniql && python -c "import sys; sys.path.insert(0, 'r3m'); from r3m import load_r3m; load_r3m('resnet18')"
```

### Wrist + State

```bash
python torch-maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_wrist_state.pkl \
  --backbone resnet18 \
  --r3m_checkpoint ~/.r3m/r3m_18/model.pt \
  --save_dir runs/iql_ws_r3m18 \
  --max_steps 200000 \
  --batch_size 128 \
  --normalize_rewards True
```

#### Logging / W&B

By default training writes:
- logs to `runs/<exp>/logs/*.log`
- metrics (JSONL) to `runs/<exp>/metrics/metrics.jsonl`

To enable Weights & Biases logging:

```bash
python torch-maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_wrist_state.pkl \
  --save_dir runs/iql_ws_r3m18 \
  --wandb \
  --wandb_project torch-maniql \
  --wandb_entity <your_entity_optional> \
  --wandb_name iql_ws_r3m18
```

If you want to avoid network access, use `--wandb_mode offline`.

### Full multimodal

```bash
python torch-maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_full.pkl \
  --backbone resnet18 \
  --r3m_checkpoint ~/.r3m/r3m_18/model.pt \
  --save_dir runs/iql_full_r3m18 \
  --max_steps 200000 \
  --batch_size 64 \
  --normalize_rewards True
```

### Debug (small run)

```bash
python torch-maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_wrist_state.pkl \
  --backbone resnet18 \
  --save_dir runs/debug_small \
  --max_steps 1000 \
  --batch_size 32 \
  --validate True
```
