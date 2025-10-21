# Getting Started with BitLab

This tutorial will guide you through the basics of using BitLab for 1.58-bit quantization.

## What is BitLab?

BitLab is a framework that enables **1.58-bit quantization** of PyTorch models. This means:
- **16x memory reduction** compared to full precision (32-bit)
- **Faster inference** through optimized kernels
- **Minimal accuracy loss** with advanced quantization techniques
- **Drop-in replacement** for PyTorch layers

## Why 1.58-bit Quantization?

Traditional quantization often uses 8-bit or 4-bit weights. BitLab pushes the boundaries with **1.58-bit quantization** using ternary weights (-1, 0, +1). This provides:

- **Extreme compression** without sacrificing too much accuracy
- **Hardware-friendly** operations that map well to modern processors
- **Research-grade** quantization for cutting-edge applications

## Installation

Follow the [installation guide](../installation.md) to set up BitLab on your system.

## Your First Quantized Layer

Let's start with a simple example:

```python
import torch
from bitcore.config import BitQuantConfig
from bitlayers.bitlinear import BitLinear

# 1. Create quantization configuration
config = BitQuantConfig(
    activation_dtype="int8",           # Use int8 for activations
    activation_granularity="per_tensor",  # Per-tensor quantization
    weight_granularity="per_tensor"    # Per-tensor quantization
)

# 2. Create a quantized linear layer
layer = BitLinear(
    in_features=512,
    out_features=256,
    bias=True,
    quant_config=config
)

# 3. Test the layer
x = torch.randn(32, 512)  # Batch size 32, 512 input features

# Training mode (full precision weights)
layer.train()
output_train = layer(x)
print(f"Training output shape: {output_train.shape}")

# Evaluation mode (quantized weights)
layer.eval()
output_eval = layer(x)
print(f"Quantized output shape: {output_eval.shape}")

# Check quantization
print(f"Quantized weight dtype: {layer.qweight.dtype}")
print(f"Memory savings: {layer.qweight.numel() * 1.58 / (layer.qweight.numel() * 32):.1%}")
```

## Understanding Quantization Modes

BitLab layers have two distinct modes:

### Training Mode (`layer.train()`)
- Uses **full precision weights** (float32)
- Enables **gradient computation**
- Removes quantized weights and scales
- Used during model training

### Evaluation Mode (`layer.eval()`)
- Uses **quantized weights** (int8)
- Disables **gradient computation**
- Creates quantized weights and scales
- Used during inference

## Configuration Options

### Weight Granularity
```python
# Per-tensor: All weights use the same scale (faster)
config = BitQuantConfig(weight_granularity="per_tensor")

# Per-channel: Each output channel has its own scale (more accurate)
config = BitQuantConfig(weight_granularity="per_channel")
```

### Activation Quantization
```python
# Keep activations in float32 (faster, less accurate)
config = BitQuantConfig(activation_dtype="float32")

# Quantize activations to int8 (slower, more accurate)
config = BitQuantConfig(activation_dtype="int8")
```

## Converting PyTorch Models

Replace PyTorch layers with quantized versions:

```python
import torch.nn as nn
from bitlayers.bitlinear import BitLinear
from bitcore.config import BitQuantConfig

# Original PyTorch model
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

# Quantized version
class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create quantization config once
        self.quant_config = BitQuantConfig(
            activation_dtype="int8",
            activation_granularity="per_tensor",
            weight_granularity="per_tensor"
        )
        
        self.layers = nn.Sequential(
            BitLinear(784, 256, quant_config=self.quant_config),  # Quantized!
            nn.ReLU(),
            BitLinear(256, 10, quant_config=self.quant_config)   # Quantized!
        )
    
    def forward(self, x):
        return self.layers(x)

# Use it exactly like a regular PyTorch model
model = QuantizedModel()
x = torch.randn(64, 784)
output = model(x)
print(f"Model output shape: {output.shape}")
```

## Using Pre-built Models

BitLab includes pre-built quantized models:

```python
from bitmodels import AutoBitModel
from bitmodels.mlp import BitMLPConfig

# Create a quantized MLP
config = BitMLPConfig(
    n_layers=3,
    in_channels=256,
    hidden_dim=128,
    out_channels=10,
    dropout=0.1,
)

# AutoBitModel handles all the quantization setup
model = AutoBitModel.from_config(config)

# Test it
x = torch.randn(1, 256)
output = model(x)
print(f"Auto model output: {output.shape}")
```

## Memory Comparison

See the dramatic memory savings:

```python
import torch
from bitcore.config import BitQuantConfig
from bitlayers.bitlinear import BitLinear

# Compare memory usage
config = BitQuantConfig(
    activation_dtype="int8",
    activation_granularity="per_tensor",
    weight_granularity="per_tensor"
)

# Regular PyTorch layer
regular_layer = torch.nn.Linear(1000, 1000)
regular_memory = regular_layer.weight.numel() * 4  # 4 bytes per float32

# Quantized BitLab layer  
quantized_layer = BitLinear(1000, 1000, quant_config=config)
quantized_layer.eval()  # Enter eval mode to quantize weights
quantized_memory = quantized_layer.qweight.numel() * 1.58 / 8  # 1.58 bits per weight

print(f"Regular layer memory: {regular_memory / 1024:.1f} KB")
print(f"Quantized layer memory: {quantized_memory / 1024:.1f} KB")
print(f"Memory reduction: {(1 - quantized_memory / regular_memory) * 100:.1f}%")
```

## Best Practices

### 1. Choose the Right Configuration
```python
# For speed (faster inference)
config = BitQuantConfig(
    weight_granularity="per_tensor",
    activation_dtype="int8",
    activation_granularity="per_tensor"
)

# For accuracy (better results)
config = BitQuantConfig(
    weight_granularity="per_channel",
    activation_dtype="int8",
    activation_granularity="per_channel"
)
```

### 2. Training Workflow
```python
# 1. Train in full precision first
model.train()
for epoch in range(num_epochs):
    # Training loop
    pass

# 2. Switch to evaluation mode for inference
model.eval()
with torch.no_grad():
    output = model(input)
```

### 3. Memory Management
```python
# Check if layer is quantized
if hasattr(layer, 'qweight'):
    print("Layer is quantized (eval mode)")
else:
    print("Layer is in training mode")
```

## Common Pitfalls

### 1. Forgetting to Call `eval()`
```python
# Wrong: Layer still in training mode
layer.train()
output = layer(x)  # Uses full precision weights

# Correct: Switch to evaluation mode
layer.eval()
output = layer(x)  # Uses quantized weights
```

### 2. Mixing Configurations
```python
# Wrong: Different layers with different configs
layer1 = BitLinear(100, 50, quant_config=config1)
layer2 = BitLinear(50, 10, quant_config=config2)  # Different config!

# Correct: Use the same config for all layers
layer1 = BitLinear(100, 50, quant_config=config)
layer2 = BitLinear(50, 10, quant_config=config)
```

### 3. Not Handling Deployment
```python
# For production deployment
layer.deploy()  # Permanently quantize the layer
# Layer is now inference-only
```

## Next Steps

Now that you understand the basics:

1. **Explore Examples**: Check out the [examples directory](../examples/)
2. **Learn Advanced Usage**: Read the [API reference](../api/)
3. **Try Different Models**: Experiment with [model configurations](../api/models.md)
4. **Optimize Performance**: Use the [performance analysis tools](../examples/performance_analysis.py)

## Getting Help

- **Documentation**: Browse the [API reference](../api/)
- **Examples**: Check out the [examples directory](../examples/)
- **Issues**: Report problems on GitHub
- **Community**: Join discussions and share results

---

**Happy quantizing!** 🎯
