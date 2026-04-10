Preprocess only wrist with R3M, and include only `[wrist, state]`:

```bash
python seed_data.py \
  --input-dir /home/purduerm/cap/data \
  --output /home/purduerm/cap/data/preprocessed/all_transitions_r3m_wrist_state.pkl \
  --features wrist,state \
  --wrist-encoder r3m \
  --wrist-model resnet50 \
  --device cuda
```

````bash
python seed_data.py \
    --input-dir /Users/connorbarnsley/work/capstone/cap/data \
    --output /Users/connorbarnsley/work/capstone/cap/data/preprocessed/all_transitions_r3m_wrist_state.pkl \
    --features wrist,state \
    --wrist-encoder r3m \
    --wrist-model resnet50 \
    --r3m-repo /Users/connorbarnsley/work/capstone/cap/r3m \
    --device mps
````

or 

If you want `[wrist, tactile, forcefield, state]` (still no pre-encoding for tactile/forcefield):

```bash
python seed_data.py \
  --input-dir /home/connors/capstone/data \
  --output /home/purduerm/cap/data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl \
  --features wrist,tactile,forcefield,state \
  --wrist-encoder r3m \
  --wrist-model resnet50 \
  --device cuda
```

```bash
python seed_data.py \
  --input-dir /home/connors/capstone/data \
  --output /home/connors/capstone/data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --r3m-rpo
  --wrist-encoder r3m \
  --wrist-model resnet18 \
  --device cuda
```

tactile and forefield will be processed using a cnn and mlp respectively and trained on the standard IQL loss.


```bash
python train_iql.py \
  --dataset_path /home/connors/capstone/data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --save_dir runs/manifeel_iql_multimodal \
  --eval_episodes 10 \ 
  --normalize_rewards True \
```


```bash
python train_iql.py \
  --dataset_path /home/connors/capstone/data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --save_dir runs/manifeel_iql_multimodal \
  --env_name YOUR_ENV_ID_HERE \
  --eval_episodes 10
```

```bash
python maniql/train_iql.py \
  --dataset_path /home/connors/capstone/data/preprocessed/all_transitions_r3m18_wtfs.pkl \
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


python maniql/train_iql.py \
  --dataset_path /home/connors/capstone/data/preprocessed/all_transitions_r3m18_wtfs.pkl \
  --save_dir runs/debug_small \
  --max_steps 1000 \
  --batch_size 32 \
  --validate True \
  --normalize_rewards False

