# EfficientZero V2 Copilot Instructions

## Overview
EfficientZero V2 is a sample-efficient RL framework supporting discrete/continuous actions and visual/state-based inputs across Atari, DMControl, and custom environments. Published at ICML 2024.

## Architecture & Key Components

### High-Level Data Flow
1. **Data Workers** (`ez/worker/data_worker.py`) → Collect trajectories via MCTS planning in parallel
2. **Replay Buffer** (`ez/data/replay_buffer.py` - Ray remote actor) → Stores prioritized trajectories, supports priority-based sampling
3. **Batch Workers** (`ez/worker/batch_worker.py`) → Sample minibatches and prepare context for training
4. **Trainer** (`ez/agents/base.py::Agent.train()`) → Main training loop using DDP for distributed training
5. **Global Storage** (`ez/data/global_storage.py` - Ray remote actor) → Manages model weights accessible to all workers

### Core Components

**Agents** (`ez/agents/`):
- `base.py::Agent` - Abstract base with `build_model()`, `train()`, and `update_config()` methods
- `ez_atari.py::EZAtariAgent` - Discrete action agent for Atari 100k
- `ez_dmc_state.py::EZDMCStateAgent` - Continuous action agent for low-dim state input
- `ez_dmc_image.py::EZDMCImageAgent` - Continuous action agent for visual input
- Agents register in `__init__.py` as `agents.names[agent_name]`

**Models** (`ez/agents/models/base_model.py`):
- `RepresentationNetwork` - Encodes observations to hidden states (downsampling for images)
- `SpectralDynamicsNetwork` - Predicts next latent state from (state_seq, action_seq) via OSF (image-based: `(B, L, C, H, W)` → `(B, C, H, W)`)
- `SpectralDynamicsNetwork1D` - Same as above but for flat state vectors (state-based: `(B, L, D)` → `(B, D)`)
- `OSFPredictor` (`osf_predictor.py`) - Core spectral filtering module using Hankel eigenpairs
- `ValuePolicyNetwork` - Outputs value & policy from hidden states
- `EfficientZero` (`__init__.py`) - Orchestrates all networks; `recurrent_inference(state_seq, action_seq, reward_hidden)`

**Latent Buffers** (`ez/utils/latent_buffer.py`):
- `LatentStateActionBuffer` - Per-env sliding window of `(seq_len, *state_shape)` states + `(seq_len, action_dim)` actions
- `BatchLatentBuffer` - Batched version `(B, seq_len, ...)` for training unroll
- Methods: `push()`, `push_state_only()`, `clone()`, `get_state_sequence()`, `get_action_sequence()`, `reset()`

**Data Flow**:
- `Transforms` (augmentation.py) - Applies shift/intensity augmentations in training loop
- `GameTrajectory` (trajectory.py) - Stores (state, action, reward, value, policy) tuples with snapshots
- Prioritized sampling via `DiscreteSupport` (format.py) for categorical value/reward distributions

**MCTS** (`ez/mcts/`):
- `base.py::MCTS` - Abstract search interface with phase-based action pruning
- `py_mcts.py` / `cy_mcts.py` - Python/Cython implementations (discrete actions)
- `ctree_v2/` - C++/Cython tree for Gumbel sampling (faster, used by default)

### Configuration System
- Base config: `ez/config/config.yaml` (framework-level settings)
- Domain configs: `ez/config/exp/{domain}.yaml` (Atari, DMC-state, DMC-image)
- Hydra merging: `python ez/train.py exp_config=ez/config/exp/atari.yaml` applies domain config on top of base
- Key sections: `agent_name`, `env`, `rl` (discount/unroll), `optimizer`, `train`, `model`, `mcts`

## Critical Developer Workflows

### Setup & Build
```bash
conda create -n ezv2 python=3.8
pip install torch==2.0.1 -r requirements.txt
cd ez/mcts/ctree_v2 && bash make.sh  # Compile Cython MCTS (required for training)
```

### Training
```bash
# Single GPU, single process
python ez/train.py exp_config=ez/config/exp/atari.yaml

# Multi-GPU with DDP
export CUDA_VISIBLE_DEVICES=0,1
python ez/train.py exp_config=ez/config/exp/atari.yaml ddp.world_size=2

# Debug/test with single process Ray
python ez/train.py exp_config=ez/config/exp/atari.yaml ray.single_process=True
```

### Evaluation
```bash
# Edit model_path in ez/config/config.yaml, then:
bash scripts/eval.sh
```

### Common Tasks
- **Add new domain**: Create `ez/config/exp/new_domain.yaml`, implement `EZNewAgent` inheriting from `base.Agent`
- **Modify model**: Edit `ez/agents/models/base_model.py` (keep `build_model()` signature compatible)
- **Change loss functions**: Edit `ez/utils/loss.py` (note: `symlog_loss` vs `kl_loss` for value support types)
- **Adjust data augmentation**: Modify `Transforms` in `augmentation.py` or config `augmentation: ['shift', 'intensity']`

## Project-Specific Patterns & Conventions

### 1. Configuration Management
- **Multi-layer merging**: Base config + domain config + CLI overrides (Hydra default behavior)
- **Dynamic config updates**: Agents call `update_config()` in `__init__` to compute action space, support sizes
- **OmegaConf pattern**: Use `with open_dict(config): config.field = value` for dynamic updates
- Example: `EZAtariAgent.update_config()` sets `env.action_space_size`, `model.reward_support.size` based on environment

### 2. Ray Actor Pattern
- **Remote objects**: `ReplayBuffer.remote()` and `GlobalStorage.remote()` are Ray actors (persistent servers)
- **Synchronization**: Workers fetch latest model via `ray.get(storage.get_weights.remote(model_name))`
- **Ray init**: Called once with GPU/CPU counts; use `config.ray.single_process=True` for debugging

### 3. Discrete Support Distributions
- Values/rewards represented as categorical distributions (not raw scalars)
- `DiscreteSupport` class (format.py): `scalar_to_vector()` → one-hot encoding, `vector_to_scalar()` → scalar reconstruction
- Loss computation uses KL divergence: target as one-hot distribution, prediction as logits
- For symlog support: use symlog/symexp wrappers around scalar values

### 4. DDP Distributed Training
- Rank 0 is main process (saves checkpoints, logs)
- `get_ddp_model_weights()` (format.py) extracts DDP wrapper state_dict
- Gradient syncing automatic via `DistributedDataParallel` wrapper
- Workers on non-main ranks skip evaluation and checkpointing

### 5. Data Format Conventions
- **Observations**: Images as `[C, H, W]` (channels-first PyTorch format)
- **Actions**: Discrete = scalar int, Continuous = vector of shape `[batch, action_dim]`
- **Batch context**: Tuples of `(trajectories, transition_pos, weights, ...)` passed to training loop
- **Snapshot storage**: `GameTrajectory` stores observation snapshots separately for memory efficiency

### 6. Loss Function Patterns
- **Value loss** (`Value_loss`): Uses IQL weighting to handle overestimation asymmetrically
- **Policy loss** (continuous): Log-probability under policy distribution (see `continuous_loss`)
- **Consistency loss**: Cosine similarity between representations (self-supervised auxiliary task)
- Coefficients configurable in domain `.yaml` files (e.g., `consistency_coeff: 5.0`)

### 7. Model Inference Phases
- **Initial inference**: `model.initial_inference(observations)` → value, policy, hidden state
- **Recurrent inference**: `model.recurrent_inference(state_seq, action_seq, reward_hidden)` → next state, value prefix, value, policy
  - `state_seq`: `(B, L, C, H, W)` for image-based or `(B, L, D)` for state-based
  - `action_seq`: `(B, L, action_dim)` — 1 for discrete, `action_space_size` for continuous
- Value prefix used for planning; value is terminal estimate

### 8. SpectralDynamicsNetwork (Observation Spectral Filtering)
- **Architecture**: Uses `OSFPredictor` with Hankel eigenpairs for spectral temporal filtering
- **Two variants**:
  - `SpectralDynamicsNetwork` — for image-based envs (spatial latent states `(B, L, C, H, W)`)
  - `SpectralDynamicsNetwork1D` — for state-based envs (flat latent vectors `(B, L, D)`)
- **Key files**: 
  - `ez/agents/models/base_model.py` - Both SpectralDynamicsNetwork variants
  - `ez/agents/models/osf_predictor.py` - OSFPredictor core (Hankel eigenpairs, spectral filtering)
  - `ez/utils/latent_buffer.py` - LatentStateActionBuffer / BatchLatentBuffer
- **How it works**: 
  - Maintains sliding window of `seq_len` latent states and raw actions
  - Action embeddings via Conv1x1 (spatial) or Linear (1D) + LayerNorm + ReLU
  - OSFPredictor processes (action_embeddings, flattened_states) → predicted next state
  - AR terms (J, P) for recent history + spectral terms (M, N) over Hankel eigenbasis
- **Config options**:
  ```yaml
  model:
    spectral_dynamics:
      sequence_length: 20          # T: sliding window length
      use_mlp: True                # MLP for nonlinear transforms in OSF
      mlp_hidden_dim: null         # (null = output_dim * 2)
      mlp_num_layers: 2
      mlp_dropout: 0.1
      mlp_activation: 'gelu'
  ```
- **Buffer management during self-play/MCTS**:
  - Real `LatentStateActionBuffer` per env persists across steps
  - Before search: `push_state_only(state)` records observed latent state
  - During MCTS: `buffer_pool` dict maps `(ix, iy)` → cloned buffers; each simulation clones parent, pushes (state, action)
  - After search: `buffer.action_buffer[-1] = chosen_action` records the played action
  - Episode reset: `buffer.reset()`
- **Training unroll**: `BatchLatentBuffer` accumulates (state, action) pairs; extracts `(B, L, ...)` sequences each step

## Important Files to Study

| File | Purpose |
|------|---------|
| `ez/train.py` | Entry point; sets up Ray, DDP, workers |
| `ez/agents/base.py` | Core training loop, model updates, loss computation |
| `ez/worker/data_worker.py` | Trajectory collection via MCTS self-play |
| `ez/worker/batch_worker.py` | Batch preparation (augmentation, context building) |
| `ez/data/replay_buffer.py` | Prioritized experience storage (Ray remote) |
| `ez/config/exp/atari.yaml` | Complete configuration example |
| `ez/agents/models/base_model.py` | Network architecture (representation, SpectralDynamics, value-policy) |
| `ez/agents/models/osf_predictor.py` | OSFPredictor: spectral filtering with Hankel eigenpairs |
| `ez/agents/models/__init__.py` | `EfficientZero` model class: orchestrates all sub-networks |
| `ez/utils/latent_buffer.py` | LatentStateActionBuffer / BatchLatentBuffer for spectral dynamics |
| `ez/utils/format.py` | Utilities: `DiscreteSupport`, `symlog/symexp`, DDP helpers |
| `ez/mcts/cy_mcts.py` | Cython MCTS with Gumbel search + latent buffer integration |
| `ez/mcts/ctree_v2/cytree.pyx` | Fast MCTS with Gumbel sampling (Cython) |

## Common Pitfalls

1. **Forgot to compile MCTS**: Training hangs. Run `cd ez/mcts/ctree_v2 && bash make.sh`
2. **Mismatched support types**: Value/reward support types must be consistent (symlog vs categorical)
3. **Batch dimension inconsistency**: Many operations expect batch-first tensors; check shapes in loss functions
4. **Ray object store overflow**: Increase `object_store_memory` in `train.py` if collecting too many trajectories
5. **Config not propagating**: Use `OmegaConf.load()` + `OmegaConf.merge()` for proper Hydra integration
6. **DDP rank confusion**: Only rank 0 should save models/logs; guard with `if rank == 0:`

## Testing & Validation

- No dedicated test suite; validate via:
  - Small config: `training_steps: 1000, data.buffer_size: 100` for quick smoke test
  - Single-process Ray mode (`ray.single_process=True`) for sequential debugging
  - Check W&B logs for loss convergence (NaN indicates training instability)
  - Eval script generates video outputs (requires ffmpeg)
