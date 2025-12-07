# MiniSTU: Lightweight Spectral Transform Unit

A simplified, standalone implementation of the Spectral Transform Unit (STU) focused on the core spectral filtering innovation. Perfect for learning linear dynamical systems and research applications.

## Features

- ⚡️ **Core STU Innovation**: Spectral filtering in frequency domain
- 🎯 **LDS Learning**: Built-in support for learning linear dynamical systems
- 🧪 **Research-Friendly**: Clean, understandable code for experimentation
- 📦 **Lightweight**: Minimal dependencies (PyTorch + NumPy only)

## Installation

```bash
# From the flash-stu-2 directory
pip install -e .
```

Or just copy the `mini_stu/` directory into your project!

## Quick Start

### Train STU to Learn an LDS

```python
import torch
from mini_stu import MiniSTU, random_LDS, train_stu_on_lds

# Create a random LDS
lds = random_LDS(state_dim=20, input_dim=10, output_dim=5)

# Train STU to approximate it
stu, losses = train_stu_on_lds(
    lds,
    seq_len=128,
    num_filters=24,
    num_steps=1000,
    batch_size=32,
)

# Test the learned STU
test_input = torch.randn(1, 128, 10)
stu_output = stu(test_input)
lds_output = lds(test_input)

print(f"MSE: {torch.mean((stu_output - lds_output)**2):.6f}")
```

### Use MiniSTU Standalone

```python
from mini_stu import MiniSTU

# Create STU without MLP (default)
stu = MiniSTU(
    seq_len=128,
    num_filters=24,
    input_dim=10,
    output_dim=5,
)

# Forward pass
x = torch.randn(4, 128, 10)  # batch of 4 sequences
y = stu(x)
print(y.shape)  # torch.Size([4, 128, 5])
```

### Use MiniSTU with Optional MLP

```python
from mini_stu import MiniSTU

# Create STU with MLP for non-linear transformations
stu = MiniSTU(
    seq_len=128,
    num_filters=24,
    input_dim=10,
    output_dim=5,
    use_mlp=True,              # Enable MLP
    mlp_hidden_dim=20,         # Hidden dimension (default: output_dim * 2)
    mlp_num_layers=2,          # Number of MLP layers
    mlp_dropout=0.1,           # Dropout rate
    mlp_activation='gelu',     # Activation: 'relu', 'gelu', or 'tanh'
)

# Forward pass
x = torch.randn(4, 128, 10)
y = stu(x)  # MLP is applied after spectral transform
print(y.shape)  # torch.Size([4, 128, 5])
```

### Pre-compute Spectral Filters

```python
from mini_stu import get_spectral_filters, MiniSTU

# Compute filters once
phi = get_spectral_filters(seq_len=128, K=24)

# Reuse across multiple models
stu1 = MiniSTU(128, 24, 10, 5, precomputed_filters=phi)
stu2 = MiniSTU(128, 24, 10, 8, precomputed_filters=phi)
```

## API Reference

### `MiniSTU`

```python
MiniSTU(
    seq_len: int,              # Maximum sequence length
    num_filters: int,          # Number of spectral filters (K)
    input_dim: int,            # Input feature dimension
    output_dim: int,           # Output feature dimension
    use_hankel_L: bool = False, # Use Hankel-L (faster, single branch)
    use_mlp: bool = False,     # Enable MLP after spectral transform
    mlp_hidden_dim: int = None, # MLP hidden dimension (default: output_dim * 2)
    mlp_num_layers: int = 2,   # Number of MLP layers
    mlp_dropout: float = 0.0,  # Dropout rate for MLP
    mlp_activation: str = 'relu', # Activation: 'relu', 'gelu', or 'tanh'
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
    precomputed_filters: torch.Tensor = None,
)
```

**Shape:**
- Input: `[batch_size, seq_len, input_dim]`
- Output: `[batch_size, seq_len, output_dim]`

### `get_spectral_filters`

```python
get_spectral_filters(
    seq_len: int,              # Sequence length
    K: int,                    # Number of filters
    use_hankel_L: bool = False, # Hankel-L formulation
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor  # [seq_len, K]
```

Computes top-K eigenvectors of the Hankel matrix, scaled by eigenvalue^0.25.

### `LDS`

```python
LDS(
    state_dim: int,    # Hidden state dimension
    input_dim: int,    # Input dimension
    output_dim: int,   # Output dimension
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
)
```

Linear Dynamical System: `h_{t+1} = A*h_t + B*u_t`, `y_t = C*h_t`

### `train_stu_on_lds`

```python
train_stu_on_lds(
    lds: LDS,
    seq_len: int,
    num_filters: int,
    num_steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: bool = True,
    use_mlp: bool = False,     # Enable MLP in trained STU
    mlp_hidden_dim: int = None, # MLP hidden dimension
    mlp_num_layers: int = 2,   # Number of MLP layers
    mlp_dropout: float = 0.0,  # Dropout rate for MLP
    mlp_activation: str = 'relu', # Activation function
) -> tuple[MiniSTU, list]  # (trained_stu, loss_history)
```

## Examples

See `examples/mini_stu_example.py` for complete examples.

## Comparison with Full Flash STU

| Feature | MiniSTU | Flash STU |
|---------|---------|-----------|
| **Core STU** | ✅ | ✅ |
| **Attention Layers** | ❌ | ✅ |
| **Transformer Model** | ❌ | ✅ |
| **LDS Learning** | ✅ | ❌ |
| **HuggingFace Compatible** | ❌ | ✅ |
| **Dependencies** | Minimal | Heavy |

## Citation

If you use MiniSTU in your research, please cite:

```bibtex
@article{flashstu,
  title={Flash STU: Fast Spectral Transform Units},
  author={Y. Isabel Liu, Windsor Nguyen, Yagiz Devre, Evan Dogariu, Anirudha Majumdar, Elad Hazan},
  journal={arXiv preprint arXiv:2409.10489},
  year={2024},
  url={https://arxiv.org/abs/2409.10489}
}
```

## License

Apache 2.0 - See LICENSE file for details.

