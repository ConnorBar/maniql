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

### Wrist + State

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

### Debug (small run)

```bash
python maniql/train_iql.py \
  --dataset_path data/preprocessed/raw_wrist_state.pkl \
  --backbone resnet18 \
  --save_dir runs/debug_small \
  --max_steps 1000 \
  --batch_size 32 \
  --validate True
```
