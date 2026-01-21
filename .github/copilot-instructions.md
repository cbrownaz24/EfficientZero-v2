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
- `DynamicsNetwork` - Predicts next state + reward given current state + action
- `ValuePolicyNetwork` - Outputs value & policy from hidden states
- `EfficientZero` - Orchestrates all three networks (see `ez_atari.py` for instantiation)

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
- **Initial inference**: `model.initial_inference(observations)` → value, policy, (hidden state for recurrent)
- **Recurrent inference**: `model.recurrent_inference(state, action, reward_hidden)` → next state, value prefix, value, policy
- Value prefix used for planning; value is terminal estimate

### 8. MiniSTU Dynamics Integration
- **Toggle via config**: `model.use_mini_stu_dynamics: True/False` in any `config/exp/{domain}.yaml`
- **Key files**: 
  - `ez/agents/models/mini_stu_dynamics.py` - MiniSTU implementation (spectral temporal units for action sequences)
  - `ez/utils/action_history.py` - ActionHistoryBuffer manages historical action sequences
- **How it works**: 
  - MiniSTU processes action history buffer a_{(t-T):t} instead of single (state, action)
  - Returns predicted state s^_{t+1} from spectral transforms over action sequence
  - Optional MLP for nonlinear transformations (configurable)
- **Config options**:
  ```yaml
  model:
    use_mini_stu_dynamics: True  # Enable MiniSTU
    mini_stu:
      sequence_length: 5          # T: history window
      num_filters: 24             # Spectral filters
      use_mlp: True               # Nonlinear transforms
      mlp_hidden_dim: null        # (null = output_dim * 2)
  ```
- **Training integration**: 
  - ActionHistoryBuffer automatically maintained during unroll loop (base.py)
  - Fallback to original dynamics if action_history not provided
  - Loss computation (MSE on predicted vs. actual states) remains unchanged

## Important Files to Study

| File | Purpose |
|------|---------|
| `ez/train.py` | Entry point; sets up Ray, DDP, workers |
| `ez/agents/base.py` | Core training loop, model updates, loss computation |
| `ez/workers/data_worker.py` | Trajectory collection via MCTS self-play |
| `ez/workers/batch_worker.py` | Batch preparation (augmentation, context building) |
| `ez/data/replay_buffer.py` | Prioritized experience storage (Ray remote) |
| `ez/config/exp/atari.yaml` | Complete configuration example (includes MiniSTU config) |
| `ez/agents/models/base_model.py` | Network architecture (representation, dynamics, value-policy) |
| `ez/agents/models/mini_stu_dynamics.py` | MiniSTU-based dynamics (spectral temporal units for action histories) |
| `ez/utils/format.py` | Utilities: `DiscreteSupport`, `symlog/symexp`, DDP helpers |
| `ez/utils/action_history.py` | ActionHistoryBuffer for MiniSTU temporal context |
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
