# ⚡️ Flash STU ⚡️

<div align="center">
  <img src="docs/flash-stu.webp" alt="Flash STU Logo" width="720">
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Examples](#usage-examples)
6. [MiniSTU](#ministu)
7. [Configuration](#configuration)
8. [Training](#training)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)

## Introduction

This repository complements the [Flash STU: Fast Spectral Transform Units](https://arxiv.org/abs/2409.10489) paper and contains an optimized, open-source PyTorch implementation of the Spectral Transform Unit (STU) as proposed in [*Spectral State Space Models*](https://arxiv.org/abs/2312.06837) by Agarwal et al. (2024).

Flash STU is a hybrid architecture that interleaves spectral state space model layers with sliding window attention, enabling scalability to billions of parameters for language modeling while maintaining near-linear time complexity. The STU module is a fast and flexible building block that can be adapted into a wide range of neural network architectures, especially those that aim to solve tasks with long-range dependencies.

## Features

- ⚡️ **Hybrid Architecture**: Interleaves STU and sliding window attention layers
- 🚀 **Fast Convolutions**: Optimized spectral convolutions using [Flash FFT](https://github.com/HazyResearch/flash-fft-conv)
- 💨 **Efficient Attention**: Sliding window attention using [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- 🔧 **HuggingFace Compatible**: Fully compatible with HuggingFace `transformers` API
- 🎯 **Advanced Features**: 
  - KV caching for generation
  - Gradient checkpointing
  - STU MLP sandwiching
  - Memory-efficient tiling
- 🌐 **Distributed Training**: Support for [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://pytorch.org/docs/stable/fsdp.html)
- 📦 **Flexible Building Blocks**: Use standalone `FlashSTUBlock` in your own architectures

## Installation

> **Note**: CUDA is required to run code from this repository.

For optimal performance, this repository was tested with:
- Python 3.12.5
- PyTorch 2.4.1
- Triton 3.0.0
- CUDA 12.4

and may be incompatible with other versions.

### Lightweight Installation (Recommended for Development)

For a minimal setup without CUDA-heavy dependencies like Flash Attention and Flash FFT:

1. Install PyTorch with CUDA support:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    ```

2. Install core dependencies only:
   ```bash
   pip install -e .
   ```
   
   Or alternatively, use the minimal requirements file:
   ```bash
   pip install -r requirements-minimal.txt
   ```

This lightweight installation includes all core dependencies (numpy, einops, transformers, etc.) but excludes the optional CUDA-heavy performance optimizations below.

### Full Installation (For Maximum Performance)

For optimal performance with Flash Attention and Flash FFT Conv optimizations:

1. Install PyTorch with CUDA support:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    ```

2. Install core dependencies:
   ```bash
   pip install -e .
   ```

3. Install Flash Attention (optional):
   ```bash
   MAX_JOBS=4 pip install flash-attn --no-build-isolation
   ```

4. Install Flash FFT Conv (optional):
   ```bash
    pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
    pip install git+https://github.com/HazyResearch/flash-fft-conv.git
    ```

### Install from Source

Or install directly from GitHub:
```bash
pip install git+https://github.com/hazan-lab/flash-stu-2.git
```

> **Note**: Installing from source will only install the lightweight version. For full performance, manually install Flash Attention and Flash FFT Conv as shown above.

## Quick Start

```python
import torch
from flash_stu import FlashSTU, FlashSTUConfig

# Create configuration
config = FlashSTUConfig(
    n_embd=512,
    n_layers=12,
    n_heads=8,
    seq_len=2048,
    vocab_size=50257,
)

# Initialize model (spectral filters computed automatically)
model = FlashSTU(config).cuda()

# Forward pass (HuggingFace compatible)
input_ids = torch.randint(0, config.vocab_size, (2, 128)).cuda()
outputs = model(input_ids=input_ids)

# Generate text
generated = model.generate(
    input_ids=input_ids[:, :10],
    max_length=50,
    temperature=0.8,
    top_k=40,
)
```

## Usage Examples

### 1. Basic Language Modeling

```python
from flash_stu import FlashSTU, FlashSTUConfig

# Configure model
config = FlashSTUConfig(
    n_embd=768,
    n_layers=12,
    n_heads=12,
    seq_len=2048,
    vocab_size=50257,
    window_size=512,  # Sliding window size
    num_eigh=24,      # Number of spectral filters
)

# Create model
model = FlashSTU(config).cuda()

# Training loop
input_ids = torch.randint(0, config.vocab_size, (4, 512)).cuda()
labels = input_ids.clone()

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs['loss']
loss.backward()
```

### 2. Using Standalone STU Block

```python
from flash_stu import FlashSTUBlock

# Create standalone STU block
stu_block = FlashSTUBlock(
    d_model=512,
    sequence_length=2048,
    num_filters=24,
    use_attention=False,  # Pure STU, no attention
).cuda()

# Use in your own architecture
x = torch.randn(2, 2048, 512).cuda()
output = stu_block(x)
```

### 3. Alternating STU and Attention Layers

```python
from flash_stu import FlashSTUBlock
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, d_model=512, n_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashSTUBlock(
                d_model=d_model,
                sequence_length=2048,
                num_filters=24,
                use_attention=(i % 2 == 1),  # Alternate STU and Attention
            )
            for i in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x, use_cache=False)
        return x
```

### 4. STU with MLP Sandwiching

```python
# Enable sandwiching for better expressiveness
config = FlashSTUConfig(
    n_embd=512,
    n_layers=12,
    stu_enable_mlp_sandwich=True,
    stu_mlp_hidden_size=2048,  # Sandwich MLP hidden size
)

model = FlashSTU(config).cuda()
```

### 5. Save and Load Model

```python
# Save model
model.save_pretrained("./my_flash_stu_model")

# Load model
from flash_stu import FlashSTU
model = FlashSTU.from_pretrained("./my_flash_stu_model").cuda()

# Generate
input_ids = torch.randint(0, config.vocab_size, (1, 10)).cuda()
output = model.generate(input_ids, max_length=100)
```

### 6. Memory-Efficient Tiling

```python
# Use tiling for large models with limited memory
config = FlashSTUConfig(
    n_embd=2048,
    n_layers=24,
    use_approx=True,     # Required for tiling
    d_in_tile=512,       # Tile input dimension
    d_out_tile=512,      # Tile output dimension
)

model = FlashSTU(config).cuda()
```

## MiniSTU

For research and experimentation with the core spectral filtering innovation, we provide **MiniSTU**: a lightweight, standalone implementation focused on learning linear dynamical systems.

### Features

- 🎯 **Core STU Only**: Pure spectral transform without attention/transformer layers
- 📦 **Minimal Dependencies**: Just PyTorch + NumPy
- 🧪 **LDS Learning**: Built-in utilities for learning dynamical systems
- 📚 **Educational**: Clean, well-documented code for understanding STU

### Quick Example

```python
from mini_stu import MiniSTU, random_LDS, train_stu_on_lds

# Create a random linear dynamical system
lds = random_LDS(state_dim=20, input_dim=10, output_dim=5)

# Train MiniSTU to approximate it
stu, losses = train_stu_on_lds(
    lds,
    seq_len=128,
    num_filters=24,
    num_steps=1000,
)

# Use the trained model
import torch
x = torch.randn(1, 128, 10)
y = stu(x)  # Shape: [1, 128, 5]
```

See [`mini_stu/README.md`](mini_stu/README.md) for complete documentation and [`examples/mini_stu_example.py`](examples/mini_stu_example.py) for a full working example.

## Configuration

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_embd` | int | 1536 | Embedding/hidden dimension |
| `n_layers` | int | 26 | Total number of layers |
| `n_heads` | int | 8 | Number of attention heads |
| `seq_len` | int | 8192 | Maximum sequence length |
| `window_size` | int/tuple | 1024 | Sliding window size for attention |
| `num_eigh` | int | 24 | Number of spectral filters for STU |
| `vocab_size` | int | 200064 | Vocabulary size |
| `use_hankel_L` | bool | False | Use Hankel-L (single branch) formulation |
| `use_approx` | bool | True | Use approx mode (~50x fewer STU params, recommended) |
| `use_flash_fft` | bool | True | Use Flash FFT for convolutions |
| `stu_enable_mlp_sandwich` | bool | False | Enable MLP sandwiching for STU |
| `torch_dtype` | dtype | bfloat16 | Model parameter dtype |

### Example Configurations

**Small Model (125M parameters)**:
```python
config = FlashSTUConfig(
    n_embd=768,
    n_layers=12,
    n_heads=12,
    seq_len=2048,
    num_eigh=24,
)
```

**Medium Model (350M parameters)**:
```python
config = FlashSTUConfig(
    n_embd=1024,
    n_layers=24,
    n_heads=16,
    seq_len=4096,
    num_eigh=32,
)
```

**Large Model (1B+ parameters)**:
```python
config = FlashSTUConfig(
    n_embd=2048,
    n_layers=32,
    n_heads=32,
    seq_len=8192,
    num_eigh=48,
    use_gradient_checkpointing=True,  # Save memory
)
```

## Training

An example LLM pretraining script is provided in [`example.py`](training/example.py) for you to test out the repository.

If your compute cluster does not have internet access, you will need to pre-download the entire dataset before running the example training script.

To download the dataset, run:
```bash
cd training
python data.py
```

> **Note**: The FineWeb-Edu 10B-token sample is a relatively large dataset. It can be swapped out for something smaller, e.g. [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (476.6M tokens).

To begin training, make sure you are in the `training` directory and run the following command in your terminal:

```bash
torchrun example.py
```

If you are in a compute cluster that uses Slurm and [environment modules](https://modules.readthedocs.io/en/latest/index.html), you can submit a job using the following command:
```bash
sbatch job.slurm
```

Model configurations can be adjusted as needed in [`config.json`](training/config.json). Be sure to adjust the configurations of the [Slurm job](training/job.slurm) based on your cluster's constraints.

> **Note**: PyTorch's `torch.compile` currently does not have great support for distributed wrapper modules like DDP or FSDP. If you encounter errors during training, try disabling `torch.compile`. For more information on `torch.compile`, see this [informal manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab).

## Contributing

Contributions are welcomed! Writing performant distributed code is always tricky. We welcome contributors to:

- Submit pull requests
- Report issues
- Help improve the project overall

## License

Apache 2.0 License

You can freely use, modify, and distribute the software, **even in proprietary products**, as long as you:
- Include proper attribution
- Include a copy of the license
- Mention any changes made

It also provides an express grant of patent rights from contributors.

See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

### Flash STU Paper Authors
- Y. Isabel Liu, Windsor Nguyen, Yagiz Devre, Evan Dogariu, Anirudha Majumdar, Elad Hazan

### Flash-STU-2 Implementation Contributors
- Kia Ghods, Hubert Strauss 

### Additional Thanks
Special thanks to (in no particular order):
- Elad Hazan and the authors of the [Spectral State Space Models](https://arxiv.org/abs/2312.06837) paper
- The Flash Attention team
- The Flash FFT team
- The PyTorch team
- Princeton Research Computing and Princeton Language and Intelligence, for supplying compute
- Andrej Karpathy, for his awesome [NanoGPT](https://github.com/karpathy/build-nanogpt) repository

## Citation

If you use this repository, or otherwise find our work valuable, please cite Flash STU:
```bibtex
@article{flashstu,
  title={Flash STU: Fast Spectral Transform Units},
  author={Y. Isabel Liu, Windsor Nguyen, Yagiz Devre, Evan Dogariu, Anirudha Majumdar, Elad Hazan},
  journal={arXiv preprint arXiv:2409.10489},
  year={2024},
  url={https://arxiv.org/abs/2409.10489}
}
```