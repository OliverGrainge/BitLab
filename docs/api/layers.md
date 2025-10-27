# Layers API Reference

This section documents the quantized layer implementations in BitLab. These layers provide high-level interfaces to BitLab's core quantization operations, automatically handling the complexity of training with straight-through estimators (STE) and efficient inference with optimized ternary kernels.

## Overview

BitLab layers are built on top of the core `bitlinear.forward` operation, which provides the fundamental 1.58-bit quantization functionality. The layers abstract away the complexity of:

- **Training Mode**: Full-precision weights with automatic quantization and STE gradient flow
- **Evaluation Mode**: Quantized weights with automatic kernel dispatch for optimal performance
- **Deployment**: Permanent quantization for production inference

The layers automatically manage the transition between these modes and ensure optimal performance through BitLab's kernel registry system.

## BitLinear

A quantized linear layer that replaces `torch.nn.Linear` with 1.58-bit quantization.

```python
from bitlayers.bitlinear import BitLinear
from bitcore.config import BitQuantConfig

# Create quantized linear layer
layer = BitLinear(
    in_features=512,
    out_features=256,
    bias=True,
    init_method='xavier_uniform',
    quant_config=BitQuantConfig()
)
```

### Parameters

- **in_features** (`int`): Number of input features
- **out_features** (`int`): Number of output features  
- **bias** (`bool`, optional): Whether to use bias. Default: `True`
- **init_method** (`str`, optional): Weight initialization method
  - `'xavier_uniform'`: Xavier/Glorot uniform initialization (default)
  - `'kaiming_uniform'`: Kaiming/He uniform initialization
- **quant_config** (`BitQuantConfig`, optional): Quantization configuration

### Methods

#### forward(x)

Forward pass through the quantized network. The behavior depends on the layer's current mode:

**Training Mode** (`layer.training=True`):
- Uses full-precision weights internally
- Automatically quantizes weights during forward pass
- Applies straight-through estimator (STE) for gradient flow
- Maintains training stability while using quantized computations

**Evaluation Mode** (`layer.training=False`):
- Uses pre-quantized weights and scales
- Dispatches to optimized ternary kernels automatically
- Maximizes inference performance through kernel selection
- No gradient computation

```python
output = layer(x)
```

**Parameters:**
- **x** (`torch.Tensor`): Input tensor of shape `(N, *, in_features)`

**Returns:**
- **output** (`torch.Tensor`): Output tensor of shape `(N, *, out_features)`

**Internal Process:**
1. **Training**: `x` → quantize → `bitlinear.forward` with full weights → output
2. **Evaluation**: `x` → quantize → `bitlinear.forward` with quantized weights → output

#### train(mode=True)

Set layer to training mode.

```python
layer.train()
```

In training mode:
- **Weight Management**: Maintains full-precision weights for gradient updates
- **Forward Pass**: Automatically quantizes weights during computation using `bitlinear.forward`
- **Gradient Flow**: Straight-through estimator (STE) enables gradients to flow through discrete quantization operations
- **Training Stability**: Preserves training dynamics while using quantized forward computations
- **Memory**: Stores both full-precision weights and quantized versions as needed 

#### eval()

Set layer to evaluation mode.

```python
layer.eval()
```

In evaluation mode:
- **Weight Quantization**: Automatically quantizes weights using `bitlinear.prepare_weights`
- **Kernel Dispatch**: Automatically selects and dispatches the most efficient ternary kernel for your configuration
- **Performance Optimization**: Leverages hardware-specific optimizations (AVX, CUDA, etc.) when available
- **Memory Efficiency**: Uses quantized weights (1.58-bit) instead of full-precision
- **No Gradients**: Disables gradient computation for inference efficiency
- **Automatic Selection**: No manual kernel selection required - BitLab chooses optimal backend 

#### deploy()

Permanently quantize the layer for inference.

```python
layer.deploy()
```

After deployment:
- **Inference-Only**: Layer becomes permanently quantized and cannot return to training mode
- **Memory Optimization**: Original full-precision weights are removed, keeping only quantized weights and scales
- **Kernel Optimization**: All forward passes use the most efficient ternary kernels available
- **Production Ready**: Layer is optimized for deployment with minimal memory footprint
- **Irreversible**: Cannot be used for training after deployment 

### Properties

- **in_features** (`int`): Number of input features
- **out_features** (`int`): Number of output features
- **use_bias** (`bool`): Whether bias is used
- **weight** (`torch.Parameter`): Full precision weights (training only)
- **bias** (`torch.Parameter`): Bias tensor
- **qweight** (`torch.Tensor`): Quantized weights (eval only)
- **qweight_scale** (`torch.Tensor`): Quantization scales (eval only)
- **is_deployed** (`bool`): Whether layer has been deployed

### Example Usage

```python
import torch
from bitlayers.bitlinear import BitLinear
from bitcore.config import BitQuantConfig

# Create configuration
config = BitQuantConfig(
    activation_dtype="int8",
    activation_granularity="per_tensor",
    weight_granularity="per_tensor"
)

# Create layer
layer = BitLinear(512, 256, quant_config=config)

# Training
layer.train()
x = torch.randn(32, 512)
output = layer(x)  # Uses full precision weights

# Evaluation
layer.eval()
output = layer(x)  # Uses quantized weights

# Check quantization
print(f"Quantized weight dtype: {layer.qweight.dtype}")
print(f"Memory savings: {layer.qweight.numel() * 1.58 / (layer.qweight.numel() * 32):.1%}")
```



