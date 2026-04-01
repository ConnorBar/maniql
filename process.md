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
  --input-dir /home/purduerm/cap/data \
  --output /home/purduerm/cap/data/preprocessed/all_transitions_r3m_wrist_tactile_force_state.pkl \
  --features wrist,tactile,forcefield,state \
  --wrist-encoder r3m \
  --wrist-model resnet50 \
  --device cuda
```

tactile and forefield will be processed using a cnn and mlp respectively and trained on the standard IQL loss.




