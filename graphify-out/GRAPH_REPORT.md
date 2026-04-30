# Graph Report - torch-maniql  (2026-04-29)

## Corpus Check
- Corpus is ~12,049 words - fits in a single context window. You may not need a graph.

## Summary
- 323 nodes · 453 edges · 27 communities detected
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 56 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_IQL Training Loop|IQL Training Loop]]
- [[_COMMUNITY_IQL Policy Networks|IQL Policy Networks]]
- [[_COMMUNITY_Logging & Rollout Watcher|Logging & Rollout Watcher]]
- [[_COMMUNITY_R3M Logger Internals|R3M Logger Internals]]
- [[_COMMUNITY_R3M Utils|R3M Utils]]
- [[_COMMUNITY_R3M Pretraining Pipeline|R3M Pretraining Pipeline]]
- [[_COMMUNITY_R3M Model & Optimizer|R3M Model & Optimizer]]
- [[_COMMUNITY_Obs Modality & Data Preprocessing|Obs Modality & Data Preprocessing]]
- [[_COMMUNITY_R3M Training Utilities|R3M Training Utilities]]
- [[_COMMUNITY_ManiFeel Dataset Loader|ManiFeel Dataset Loader]]
- [[_COMMUNITY_R3M Language Module|R3M Language Module]]
- [[_COMMUNITY_Vision Backbone & Preprocessing|Vision Backbone & Preprocessing]]
- [[_COMMUNITY_Transition Visualization|Transition Visualization]]
- [[_COMMUNITY_Data Inspection|Data Inspection]]
- [[_COMMUNITY_R3M Weight Loading|R3M Weight Loading]]
- [[_COMMUNITY_Data Inspector Script|Data Inspector Script]]
- [[_COMMUNITY_Transition Video Export|Transition Video Export]]
- [[_COMMUNITY_R3M OSS Meta|R3M OSS Meta]]
- [[_COMMUNITY_Modality Type|Modality Type]]
- [[_COMMUNITY_R3M Load Result|R3M Load Result]]
- [[_COMMUNITY_IQL Info Dataclass|IQL Info Dataclass]]
- [[_COMMUNITY_Batch Namedtuple|Batch Namedtuple]]
- [[_COMMUNITY_R3M Utils Module|R3M Utils Module]]
- [[_COMMUNITY_Timer Utility|Timer Utility]]
- [[_COMMUNITY_Truncated Normal|Truncated Normal]]
- [[_COMMUNITY_LR Schedule|LR Schedule]]
- [[_COMMUNITY_Code of Conduct|Code of Conduct]]

## God Nodes (most connected - your core abstractions)
1. `ResNetBackbone` - 14 edges
2. `main()` - 13 edges
3. `ManiFeelDataset` - 12 edges
4. `train_iql main` - 12 edges
5. `main()` - 11 edges
6. `Workspace` - 11 edges
7. `MetersGroup` - 11 edges
8. `Logger` - 10 edges
9. `R3M nn.Module (Visual Encoder + Reward)` - 10 edges
10. `rollout_watch_isaac main` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Async rollout watcher: load PyTorch IQL checkpoints and evaluate in IsaacGym.  E` --uses--> `DiagGaussianPolicy`  [INFERRED]
  rollout_watch_isaac.py → multimodal_nets.py
- `IQLLearner` --uses--> `ResNetBackbone`  [INFERRED]
  multimodal_nets.py → vision_backbone.py
- `Multi-modal networks + PyTorch IQL learner (end-to-end trainable R3M).  This fil` --uses--> `ResNetBackbone`  [INFERRED]
  multimodal_nets.py → vision_backbone.py
- `_load_actor()` --calls--> `DiagGaussianPolicy`  [INFERRED]
  rollout_watch_isaac.py → multimodal_nets.py
- `main()` --calls--> `ManiFeelDataset`  [INFERRED]
  train_iql.py → manifeel_iql.py

## Hyperedges (group relationships)
- **IQL End-to-End Training Pipeline** — manifeel_iql_manifeeldataset, multimodal_nets_iqllearner, train_iql_main [EXTRACTED 1.00]
- **Multimodal Observation Encoding (wrist + tactile + force + state)** — multimodal_nets_multimodalencoder, vision_backbone_resnetbackbone, obs_modality_modes [EXTRACTED 1.00]
- **Raw Data to IQL-Ready Dataset Pipeline** — seed_data_pipeline, manifeel_iql_manifeeldataset, obs_modality_getsplitkeys [EXTRACTED 0.95]
- **R3M Multi-Objective Training Loss (InfoNCE + TCN + LP)** — r3m_infonceloss, r3m_tcnloss, r3m_lploss [EXTRACTED 0.95]
- **R3M Language-Vision Grounding Pipeline** — r3m_lang_encoder, r3m_language_reward, r3m_get_reward [EXTRACTED 0.95]
- **R3M Training Orchestration (Workspace + Trainer + DataLoader)** — r3m_workspace, r3m_trainer_class, r3m_r3mbuffer [EXTRACTED 0.90]

## Communities

### Community 0 - "IQL Training Loop"
Cohesion: 0.09
Nodes (33): AWR-Style Actor Update, Implicit Q-Learning (IQL), IsaacGym Environment, init_wandb, setup_logging, wandb_log, write_jsonl, DiagGaussianPolicy (+25 more)

### Community 1 - "IQL Policy Networks"
Cohesion: 0.13
Nodes (15): act(), compute_losses(), DiagGaussianPolicy, DoubleQNet, encoded_obs_dim(), expectile_loss(), IQLInfo, MLP (+7 more)

### Community 2 - "Logging & Rollout Watcher"
Cohesion: 0.12
Nodes (26): _coerce_config(), init_wandb(), Configure console + file logging.      Writes to <save_dir>/logs/<timestamp>.log, Initialize a W&B run. Returns wandb module if enabled else None., setup_logging(), wandb_log(), write_jsonl(), IQLLearner (+18 more)

### Community 3 - "R3M Logger Internals"
Cohesion: 0.12
Nodes (5): object, AverageMeter, LogAndDumpCtx, Logger, MetersGroup

### Community 4 - "R3M Utils"
Cohesion: 0.08
Nodes (7): accuracy(), eval_mode, Every, Computes the precision@k for the specified values of k, Timer, TruncatedNormal, Until

### Community 5 - "R3M Pretraining Pipeline"
Cohesion: 0.13
Nodes (7): IterableDataset, main(), make_network(), Workspace, Trainer, get_ind(), R3MBuffer

### Community 6 - "R3M Model & Optimizer"
Cohesion: 0.13
Nodes (23): Adam Optimizer for R3M Encoder, cleanup_config() Function, Cosine Similarity Metric, DistilBERT Pretrained Model, R3M Inference Example Script, R3M.forward() Method (Image Encoding), R3M.get_reward() Method, InfoNCE Loss (Language Reward) (+15 more)

### Community 7 - "Obs Modality & Data Preprocessing"
Cohesion: 0.17
Nodes (15): get_split_keys(), Observation modality constants for the two pipeline modes.  Mode "wrist_state":, _empty_buffers(), flush_chunk(), main(), merge_chunks(), parse_args(), preprocess_file() (+7 more)

### Community 8 - "R3M Training Utilities"
Cohesion: 0.13
Nodes (18): AverageMeter Metric Tracker, Ego4D Dataset Source, Every Step Predicate Class, get_ind() Frame Loader Function, Logger Class, make_network() Factory Function, Ego4D Manifest CSV Index, MetersGroup CSV+Console Logger (+10 more)

### Community 9 - "ManiFeel Dataset Loader"
Cohesion: 0.13
Nodes (6): ManiFeelDataset, Dataset loader: preprocessed pickle -> IQL Batch interface.  Supports both pipel, # IMPORTANT: Use index-based split so we don't duplicate multi-GB tactile arrays, Run sanity checks and print warnings.  Returns True if clean., Loads a preprocessed pickle and exposes ``sample(batch_size) -> Batch``.      Ar, Single observation with leading batch dim 1 (for model init).

### Community 10 - "R3M Language Module"
Cohesion: 0.19
Nodes (3): LangEncoder, LanguageReward, R3M

### Community 11 - "Vision Backbone & Preprocessing"
Cohesion: 0.21
Nodes (9): force_to_image(), load_r3m_resnet_weights(), r3m_preprocess_bhwc(), R3MLoadResult, PyTorch-native vision utilities + R3M-backed ResNet feature extractor.  This pro, Convert BHWC images to normalized BCHW for ImageNet-pretrained ResNets.      - I, (B,420) -> (B,224,224,3) float (unnormalized)., Load R3M convnet weights into a torchvision ResNet backbone.      Accepts checkp (+1 more)

### Community 12 - "Transition Visualization"
Cohesion: 0.29
Nodes (10): extract_series(), load_transitions(), main(), make_combined_frame(), make_plot_panel(), Build a bottom panel with:     - reward timeline (+ success/done markers),     -, Create a horizontal, movie-like frame where we:     - label each view separately, Precompute time-series arrays (reward, done, success, actions). (+2 more)

### Community 13 - "Data Inspection"
Cohesion: 0.22
Nodes (11): inspect_preprocessed, inspect_raw, ManiFeel Dataset (MARS Lab), ManiFeelDataset, get_split_keys, IMAGE_KEYS frozenset, flush_chunk, seed_data main (+3 more)

### Community 14 - "R3M Weight Loading"
Cohesion: 0.8
Nodes (4): cleanup_config(), load_r3m(), load_r3m_reproduce(), remove_language_head()

### Community 15 - "Data Inspector Script"
Cohesion: 0.5
Nodes (1): Quick summary stats for raw transition files and/or preprocessed pickle.

### Community 16 - "Transition Video Export"
Cohesion: 0.5
Nodes (4): extract_series, visualize_transitions main, make_combined_frame, make_plot_panel

### Community 18 - "R3M OSS Meta"
Cohesion: 1.0
Nodes (2): CONTRIBUTING.md - R3M Contribution Guide, R3M Package Init / Loader

### Community 22 - "Modality Type"
Cohesion: 1.0
Nodes (1): Modality (Literal type)

### Community 23 - "R3M Load Result"
Cohesion: 1.0
Nodes (1): R3MLoadResult

### Community 24 - "IQL Info Dataclass"
Cohesion: 1.0
Nodes (1): IQLInfo

### Community 25 - "Batch Namedtuple"
Cohesion: 1.0
Nodes (1): Batch (namedtuple)

### Community 26 - "R3M Utils Module"
Cohesion: 1.0
Nodes (1): R3M Utils Module

### Community 27 - "Timer Utility"
Cohesion: 1.0
Nodes (1): Timer Utility Class

### Community 28 - "Truncated Normal"
Cohesion: 1.0
Nodes (1): TruncatedNormal Distribution

### Community 29 - "LR Schedule"
Cohesion: 1.0
Nodes (1): schedule() LR/Param Schedule Function

### Community 30 - "Code of Conduct"
Cohesion: 1.0
Nodes (1): CODE_OF_CONDUCT.md - Meta OSS Code of Conduct

## Knowledge Gaps
- **64 isolated node(s):** `Quick summary stats for raw transition files and/or preprocessed pickle.`, `Configure console + file logging.      Writes to <save_dir>/logs/<timestamp>.log`, `Initialize a W&B run. Returns wandb module if enabled else None.`, `Preprocessor: raw transition files -> IQL-ready pickle with raw observations.  U`, `Ensure an image array is uint8 HWC.` (+59 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Data Inspector Script`** (4 nodes): `inspect_data.py`, `inspect_preprocessed()`, `inspect_raw()`, `Quick summary stats for raw transition files and/or preprocessed pickle.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `R3M OSS Meta`** (2 nodes): `CONTRIBUTING.md - R3M Contribution Guide`, `R3M Package Init / Loader`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Modality Type`** (1 nodes): `Modality (Literal type)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `R3M Load Result`** (1 nodes): `R3MLoadResult`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `IQL Info Dataclass`** (1 nodes): `IQLInfo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Batch Namedtuple`** (1 nodes): `Batch (namedtuple)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `R3M Utils Module`** (1 nodes): `R3M Utils Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Timer Utility`** (1 nodes): `Timer Utility Class`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Truncated Normal`** (1 nodes): `TruncatedNormal Distribution`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `LR Schedule`** (1 nodes): `schedule() LR/Param Schedule Function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Code of Conduct`** (1 nodes): `CODE_OF_CONDUCT.md - Meta OSS Code of Conduct`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ManiFeelDataset` connect `ManiFeel Dataset Loader` to `Logging & Rollout Watcher`, `Obs Modality & Data Preprocessing`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `main()` connect `Logging & Rollout Watcher` to `ManiFeel Dataset Loader`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `IQLLearner` connect `Logging & Rollout Watcher` to `IQL Policy Networks`?**
  _High betweenness centrality (0.046) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `ResNetBackbone` (e.g. with `MLP` and `MultiModalEncoder`) actually correct?**
  _`ResNetBackbone` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `main()` (e.g. with `setup_logging()` and `ManiFeelDataset`) actually correct?**
  _`main()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ManiFeelDataset` (e.g. with `Offline IQL training on ManiFeel datasets (pure PyTorch, end-to-end vision finet` and `main()`) actually correct?**
  _`ManiFeelDataset` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `train_iql main` (e.g. with `Weights & Biases Integration` and `rollout_watch_isaac main`) actually correct?**
  _`train_iql main` has 2 INFERRED edges - model-reasoned connections that need verification._