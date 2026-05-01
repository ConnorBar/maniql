# IQL Pipeline Audit — torch-maniql

Holistic review of the offline IQL pipeline: algorithm correctness, data pipeline,
architecture, regularization, and training dynamics.

**Dataset context:** 40 human demonstration episodes (~36 train / ~4 test),
224x224x3 wrist camera + 7-dim state, 7-dim continuous actions, sparse rewards.

NOTE: we are looking primarily only using the wrist camera and the state, not the full feature set that we see in the one example file.

---

## Bugs

### B1. Terminal/timeout masking uses `dones` instead of `terminals`
- **File:** `manifeel_iql.py:61`
- **Severity:** Medium
- **Issue:** The dataset loader computes `masks = 1 - dones`, where `dones` includes
  timeouts. `seed_data.py` correctly computes and stores a separate `terminals` field
  (`dones - timeouts`, clipped), but the loader ignores it. Timeout episodes get
  `mask=0`, telling the critic the future value is zero — biasing V downward for
  late-episode states and creating an artificial value cliff near the time horizon.
- **Fix:** `self.masks = 1.0 - data["terminals"].astype(np.float32)`
- **Status:** [ ] TODO

### B2. Reward normalization scale (x1000) causes advantage saturation
- **File:** `train_iql.py:36-42`
- **Severity:** Medium
- **Issue:** `normalize_rewards` divides by return range then multiplies by 1000.
  With discount=0.99 and ~300-step episodes, cumulative discounted returns can reach
  ~200,000. This makes advantages huge, causing `exp(adv * temperature)` to saturate
  at the clamp of 100 for most successful transitions and near-zero for failures —
  collapsing advantage weighting to a binary signal.
- **Fix:** Normalize returns to unit range (divide by range, do not multiply by 1000).
  Adjust temperature upward (3.0–10.0) to compensate.
- **Status:** [x] DONE — removed x1000 multiplier; rewards now normalized to unit range.

### B3. Default temperature=0.1 with sparse rewards disables IQL advantage weighting
- **File:** `train_iql.py` (default args)
- **Severity:** High (design)
- **Issue:** With sparse rewards (mostly 0, occasionally 1) and no normalization,
  advantages are small. `exp(small * 0.1) ≈ 1.0` for all transitions, reducing
  the actor update to unweighted MLE — pure behavioral cloning. IQL's entire edge
  over BC comes from advantage weighting, and this setting disables it.
- **Fix:** Enable reward normalization (fixed per B2) and increase temperature to
  3.0–10.0 so the weighting has dynamic range.
- **Status:** [x] DONE — default temperature changed from 0.1 to 3.0; normalize_rewards
  now defaults to True.

---

## Architectural Issues

### A1. Separate encoders for actor/critic/value — no parameter sharing
- **File:** `multimodal_nets.py`
- **Severity:** High
- **Issue:** Each of actor, critic (x2 for DoubleQ), value, and target critic (x2)
  has its own `MultiModalEncoder` containing a full ResNet18. That's 6 ResNet18s in
  `wrist_state` mode (18 in `full` mode). Problems:
  - **Memory:** ~66M params (wrist_state) / ~200M (full) just for encoders, plus 2x
    for Adam optimizer state.
  - **Advantage consistency:** IQL computes `adv = Q(s,a) - V(s)`. If Q and V use
    different encoders extracting different features from the same image, the advantage
    is comparing apples to oranges — noisy and potentially meaningless.
  - **Data efficiency:** Each encoder gets gradient signal from only its own loss,
    fragmenting the already scarce supervision from 36 episodes.
- **Fix:** Share a single encoder across actor, critic, and value. Use stop-gradient
  from actor loss to the shared encoder (let critic/value losses drive encoder training).
  Alternatively, freeze a pretrained R3M backbone and train only MLP heads.
- **Status:** [x] DONE — shared encoder with separate head-only classes (ValueHead,
  QHead, DoubleQHead, PolicyHead). Encoder gets its own optimizer; value+critic losses
  train the encoder; actor loss is detached. Target EMA covers both encoder and critic heads.

### A2. Force field encoded via full ResNet18 — wasteful and lossy (not using right now though)
- **File:** `vision_backbone.py:force_to_image`, `multimodal_nets.py`
- **Severity:** Low-Medium
- **Issue:** The 420-dim force vector is reshaped to 14x10x3, upsampled to 224x224,
  ImageNet-normalized, and fed through a full ResNet18 (~11M params). The upsampled
  image is a smooth low-frequency blob — ResNet's convolutional filters (designed for
  natural image edges/textures) extract meaningless features. A 2-layer MLP could
  extract the same information with ~200K params.
- **Additional concern:** Per-sample normalization in `force_to_image` amplifies
  float noise to large magnitudes when force readings are near-zero.
- **Fix:** Replace the force ResNet with `MLP(420, [256, 256], feat_dim)`. Handle
  near-zero force norm with a guard.
- **Status:** [ ] TODO

---

## Regularization & Overfitting

### R1. No regularization anywhere — severe overfitting risk
- **File:** `multimodal_nets.py`, `train_iql.py`
- **Severity:** High
- **Issue:** ~36 training episodes (~7,000–18,000 transitions), batch_size=128,
  1M steps = **7,000–18,000 effective epochs**. No dropout, no weight decay, no
  data augmentation, no spectral normalization. The networks will memorize every
  training transition. Train losses go to zero; test losses diverge; the policy
  memorizes actions rather than learning a generalizable mapping.
- **Fix (priority order):**
  1. Image augmentation — random crops (DrQ-style: pad 4px, random crop to 224x224)
     and color jitter. Single most impactful regularizer for vision RL.
  2. Weight decay — `weight_decay=1e-4` in all optimizers.
  3. Differential learning rates — 1e-5 for backbone, 3e-4 for MLP heads.
  4. Dropout — `Dropout(0.1)` between MLP layers.
- **Status:** [~] PARTIAL — weight decay (1e-4) added to all optimizers. DrQ-style
  random shift augmentation added (default on, pad=4). Dropout still TODO.

### R2. No gradient clipping — instability risk with sparse rewards
- **File:** `multimodal_nets.py:IQLLearner.update`
- **Severity:** Medium
- **Issue:** No `clip_grad_norm_` anywhere. With end-to-end ResNet fine-tuning and
  sparse rewards, a single outlier batch can cause a large parameter jump that
  destabilizes the value function, propagating through advantage estimates to the
  actor. The training will not crash (no NaN likely) but can silently plateau.
- **Fix:** Add `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` after each
  `.backward()` for all three optimizers.
- **Status:** [x] DONE — clip_grad_norm_(max_norm=1.0) on all four param groups
  after each backward pass. Configurable via --max_grad_norm.

### R3. No learning rate scheduling
- **File:** `train_iql.py`
- **Severity:** Low-Medium
- **Issue:** Fixed 3e-4 for all parameters over 1M steps. Standard for MLP-only IQL,
  but too aggressive for ResNet fine-tuning on a tiny dataset.
- **Fix:** Linear warmup for the first 1000 steps, cosine decay to 1e-5 over the
  full run. Or at minimum, differential LR (see R1).
- **Status:** [x] DONE — differential LR (backbone 1e-5, heads 3e-4) with linear
  warmup (1000 steps) + cosine decay to 10% of initial LR. All configurable via CLI.

---

## Evaluation & Monitoring Gaps

### E1. Eval metric is IQL losses, not policy quality
- **File:** `train_iql.py:eval_on_dataset`
- **Severity:** Low (can't fix without environment access)
- **Issue:** `eval_on_dataset` computes IQL losses on test transitions. Decreasing
  test critic loss means Q generalizes to unseen transitions from the same demos —
  useful for detecting overfitting but says nothing about whether the policy would
  succeed in the environment. Both train and test come from the same demonstration
  distribution.
- **Note:** Environment rollouts require the Isaac Gym setup (deferred to lab).
  The current eval is the best available signal without the simulator.
- **Status:** [ ] Acknowledged — no action until Isaac Gym integration

### E2. Backbone gradient norm tracked for only one encoder
- **File:** `multimodal_nets.py:IQLLearner.update`
- **Severity:** Low
- **Issue:** Only `critic.q1.encoder.wrist_backbone.net.conv1.weight.grad` is tracked.
  This misses actor and value encoders entirely. If those encoders have vanishing or
  exploding gradients, there's no signal.
- **Fix:** Track global grad norm across all parameters, or at minimum one gradient
  norm per network (actor, critic, value). Will be partially addressed by A1 (shared
  encoder).
- **Status:** [ ] TODO

---

## Data Pipeline Edge Cases

### D1. Episodes without a `done` signal may corrupt next_obs at boundaries
- **File:** `seed_data.py:preprocess_file`
- **Severity:** Low
- **Issue:** `preprocess_file` trims to the first `done=True`. If an episode file
  has no done signal (truncated recording), all transitions are kept with `done=0`
  on the last transition. After concatenation, that transition's `next_obs` is valid
  (from the raw data, not from the next episode), but `mask=1` means the critic
  bootstraps through whatever that `next_obs` is. If the raw recording was truncated
  mid-episode, this is correct behavior. If the `next_obs` on the last transition is
  garbage/uninitialized, it leaks into training.
- **Fix:** Validate that all raw episode files contain at least one `done=True`.
  Log a warning if not.
- **Status:** [ ] TODO

---

## Priority Order for Implementation

| Priority  | Item                              | Impact     | Effort  | Done |
|-----------|-----------------------------------|------------|---------|------|
| 1 | A1    — Shared encoder                    | High       | Medium  |  X   |
| 2 | B3+B2 — Fix rewards + temperature         | High       | Low     |  X   |
| 3 | R1    — Image augmentation + weight decay | High       | Low     |  X   |
| 4 | B1    — Terminal/timeout masking          | Medium     | Trivial |      |
| 5 | R2    — Gradient clipping                 | Medium     | Trivial |  X   |
| 6 | A2    — Force MLP (full mode only)        | Low-Medium | Low     |      |
| 7 | R3    — LR scheduling                     | Low-Medium | Low     |  X   |
| 8 | E2    — Better grad tracking              | Low        | Trivial |      |
| 9 | D1    — Done-signal validation            | Low        | Trivial |      |
