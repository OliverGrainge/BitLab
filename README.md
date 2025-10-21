# BitLab

## BitLab is the home of Ternary Nerual Netowrks. But what are Ternary Networks ?

**Ternary networks** represent a breakthrough in neural network efficiency, using weights that can only take three values: **-1, 0, and +1**. This seemingly simple constraint unlocks extraordinary computational and memory advantages:

### The Science Behind Ternary Networks

Traditional neural networks use 32-bit floating-point numbers to represent their weights. Whilst this might not sound like much, as models scale to millions and billions of parameters, the memory and computational demands of such networks can become prohibitive. Ternary networks, however, address this problem by compressing the number of bits required to represent each weight to just **1.58 bits** (log₂(3) ≈ 1.58), efficiently encoding the three possible values.

**Why 1.58 bits?** With only three possible values (-1, 0, +1), we need log₂(3) ≈ 1.58 bits of information to encode each weight, compared to 32 bits for full precision!

### Why Ternary Networks are So Useful

1. ** Massive Memory Savings**: Ternary networks can reduce model memory usage by over 95% compared to standard 32-bit floating-point models, allowing much larger models to fit on smaller devices.
2. ** Simpler, Faster Computation**: Ternary weights mean most multiplications can be replaced with lightweight additions and subtractions. For example, a calculation like (1 × A + 0 × B + -1 × C) is reduced to just (A - C), enabling much faster matrix operations—especially in hardware.
3. ** Ultra-Low Power Consumption**: Traditional Multiply-Accumulate (MAC) operations are energy intensive. Ternary arithmetic eliminates nearly all multiplications, leading to dramatically lower power requirements—crucial for mobile and edge AI applications.
4. ** Perfect for Edge and Mobile**: The extreme efficiency and tiny memory footprint of ternary networks make them ideal for deploying advanced AI models directly on smartphones, IoT devices, wearables, and microcontrollers where resources are limited.


## How BitLab Harnesses Ternary Power

BitLab is a cutting-edge framework that brings the power of **1.58-bit quantization** to PyTorch models. By reducing model weights to just 1.58 bits per parameter, BitLab delivers:

- ** 16x memory reduction** - Deploy larger models on smaller devices
- ** Faster inference** - Optimized C++ and CUDA kernels for maximum speed  
- ** Minimal accuracy loss** - Advanced quantization techniques preserve model performance
- ** Drop-in replacement** - Seamlessly replace PyTorch layers with quantized versions

### BitLab's Revolutionary Approach

BitLab makes ternary networks accessible through and brings their theoretical efficiency's into practical performance gains through a standardised API. In particular, we provide: 

- **bitcore**: A package for optimzed training and inference ternary operators 
- **bitlayers**: A package of standalone layer replacements for pytorch 'nn.Linear -> bilayers.bitlinear'
- **bitmodels**: A package, of well tested model ternary model architectures from language models, to object detectors and image generators. 

## 🚀 Key Features

BitLab offers everything you need to leverage the power of ternary networks:

- ** Optimized Kernels**: C++ and CUDA implementations with PyTorch fallbacks for maximum performance.
- ** Flexible Quantization**: Supports both per-tensor and per-channel quantization schemes.
- ** Smart Dispatch**: Automatically selects the best kernel based on your hardware configuration.
- ** Easy Integration**: Drop-in replacement for PyTorch layers—minimal code changes required.
- ** Demonstrated Efficiency Gains**: Translates the theoretical efficiency gains of ternary networks into real-world performance improvements.



##  Quick Start

Ready to harness the power of ternary networks? Get up and running with BitLab in under 5 minutes! 

### Prerequisites

- **Python 3.7+** 
- **PyTorch** (any recent version)
- **C++ compiler** (clang++ on macOS, g++ on Linux) for optimal performance

### Installation

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd BitLab

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install PyTorch
pip install torch

# 4. Build optimized C++ extensions (recommended for performance)
python build_extensions.py

# 5. Install in development mode
pip install -e .
```

###  Verify Installation

```python
python -c "
from bitcore.config import BitQuantConfig
from bitcore.kernels import KernelRegistry
from bitlayers.bitlinear import BitLinear

# Create config
config = BitQuantConfig(
    activation_dtype='int8',
    activation_granularity='per_tensor',
    weight_granularity='per_tensor'
)

# Test kernel dispatch
kernel = KernelRegistry.get_kernel_for_config(config)
print(f'✅ BitLab installation successful!')
print(f'Kernel: {type(kernel).__name__}')
print(f'Has C++ backend: {kernel.has_cpp_backend}')
"
```

**You're ready to start training ternary networks. !** Let's dive into some examples.


##  Hello World Examples

Let's start with some simple examples to get you familiar with BitLab and see ternary networks in action!

### Example 1: Your First Ternary Network Layer

Let's create your first ternary network layer and see the incredible memory savings in action!

```python
import torch
from bitcore.config import BitQuantConfig
from bitlayers.bitlinear import BitLinear

# Step 1: Create quantization configuration for ternary networks
config = BitQuantConfig(
    activation_dtype="int8",           # Use int8 for activations
    activation_granularity="per_tensor",  # Per-tensor quantization
    weight_granularity="per_tensor"    # Per-tensor quantization
)

# Step 2: Create a ternary network layer (just like nn.Linear!)
layer = BitLinear(
    in_features=512,
    out_features=256,
    bias=True,
    quant_config=config
)

# Step 3: Use the layer
x = torch.randn(32, 512)  # Batch size 32, 512 input features

# Training mode (full precision weights for stable training)
layer.train()
output_train = layer(x)
print(f"Training output shape: {output_train.shape}")

# Evaluation mode (ternary weights - this is where the magic happens!)
layer.eval()
output_eval = layer(x)
print(f"Ternary network output shape: {output_eval.shape}")

# Check if weights are actually quantized to ternary values
print(f"Quantized weight dtype: {layer.qweight.dtype}")  # Should be torch.int8
print(f"Memory savings: {layer.qweight.numel() * 1.58 / (layer.qweight.numel() * 32):.1%}")
print(f"Weight values: {torch.unique(layer.qweight)}")  # Should show [-1, 0, 1]
```

**Output:**
```
Training output shape: torch.Size([32, 256])
Ternary network output shape: torch.Size([32, 256])
Quantized weight dtype: torch.int8
Memory savings: 4.9%
Weight values: tensor([-1,  0,  1])  # True ternary values!
```

### Example 2: Convert PyTorch Models to Ternary Networks

Transform any PyTorch model into a high-efficiency ternary network with minimal code changes!

```python
import torch.nn as nn
from bitlayers.bitlinear import BitLinear
from bitcore.config import BitQuantConfig

# Replace this PyTorch model...
class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),  # Regular PyTorch layer
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# ...with this ternary network version (just change nn.Linear to BitLinear!)
class TernaryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Create quantization config once
        self.quant_config = BitQuantConfig(
            activation_dtype="int8",
            activation_granularity="per_tensor",
            weight_granularity="per_tensor"
        )
        
        self.layers = nn.Sequential(
            BitLinear(784, 256, quant_config=self.quant_config),  # Ternary weights!
            nn.ReLU(),
            BitLinear(256, 10, quant_config=self.quant_config)   # Ternary weights!
        )
    
    def forward(self, x):
        return self.layers(x)

# Use it exactly like a regular PyTorch model
model = TernaryNetwork()
x = torch.randn(64, 784)
output = model(x)
print(f"Ternary network output shape: {output.shape}")
```

### Example 3: Pre-built Ternary Network Models

Create powerful ternary network models with just a few lines of code!

```python
import torch
from bitmodels import AutoBitModel
from bitmodels.mlp import BitMLPConfig

# Create a ternary network MLP with just a few lines!
config = BitMLPConfig(
    n_layers=3,
    in_channels=256,
    hidden_dim=128,
    out_channels=10,
    dropout=0.1,
)

# AutoBitModel handles all the ternary network setup for you
model = AutoBitModel.from_config(config)

# Test it
x = torch.randn(1, 256)
output = model(x)
print(f"Ternary network model output: {output.shape}")
```


**Output:**
```
Traditional layer memory: 3906.2 KB
Ternary network memory: 197.3 KB
Memory reduction: 95.0%
Weight values: tensor([-1,  0,  1])  # True ternary values!
```

## 🎛️ Configuration Options
 
```python
from bitcore.config import BitQuantConfig

# Available configurations
config = BitQuantConfig(
    weight_granularity="per_tensor",     # "per_tensor" or "per_channel"
    activation_dtype="int8",             # "float32" or "int8"
    activation_granularity="per_tensor"  # "per_tensor" or "per_channel"
)
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Documentation Index](docs/index.md)** - Complete documentation overview
- **[Quick Start Guide](docs/README.md)** - Get up and running in 5 minutes
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[API Reference](docs/api/)** - Complete API documentation
  - [Core API](docs/api/core.md) - Core quantization functionality
  - [Layers API](docs/api/layers.md) - Quantized layer implementations
  - [Models API](docs/api/models.md) - Pre-built quantized models
- **[Examples](docs/examples/)** - Code examples and tutorials
- **[Benchmarks](docs/benchmarks/)** - Performance results and comparisons

## 🏗️ Architecture

### Package Structure

```
BitLab/
├── bitcore/                    # Core quantization functionality
│   ├── config.py              # BitQuantConfig
│   ├── ops/                   # Operations (bitlinear, etc.)
│   └── kernels/               # Optimized kernels
│       ├── registry.py        # Kernel dispatch system
│       ├── bindings/          # C++ extensions
│       └── bitlinear_*.py     # Kernel implementations
├── bitlayers/                 # PyTorch layer wrappers
│   ├── bitlinear.py           # Quantized linear layer
│   └── base.py               # Base layer class
├── bitmodels/                 # Model definitions
├── docs/                      # Documentation
│   ├── api/                   # API reference
│   ├── examples/              # Code examples
│   ├── tutorials/             # Step-by-step guides
│   └── benchmarks/            # Performance results
├── setup.py                   # Package setup
├── build_extensions.py        # Build script
└── README.md                  # This guide
```



## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** and clone your fork
2. **Create a feature branch** for your changes
3. **Add tests** for any new functionality
4. **Ensure C++ extensions build** correctly on your system
5. **Submit a pull request** with a clear description

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Build extensions
python build_extensions.py
```

## 📄 License


## 🙏 Acknowledgments


---

**Happy quantizing!** 🎯
