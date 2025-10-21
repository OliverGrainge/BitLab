# Layers API Reference

This section documents the quantized layer implementations in BitLab.

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

Forward pass through the layer.

```python
output = layer(x)
```

**Parameters:**
- **x** (`torch.Tensor`): Input tensor of shape `(N, *, in_features)`

**Returns:**
- **output** (`torch.Tensor`): Output tensor of shape `(N, *, out_features)`

#### train(mode=True)

Set layer to training mode.

```python
layer.train()
```

In training mode:
- Uses full precision weights
- Enables gradient computation
- Removes quantized weights and scales

#### eval()

Set layer to evaluation mode.

```python
layer.eval()
```

In evaluation mode:
- Uses quantized weights
- Disables gradient computation
- Creates quantized weights and scales

#### deploy()

Permanently quantize the layer for inference.

```python
layer.deploy()
```

After deployment:
- Layer becomes inference-only
- Original weights are removed
- Cannot return to training mode

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

## BitLayerBase

Base class for all quantized layers.

```python
from bitlayers.base import BitLayerBase

class CustomBitLayer(BitLayerBase):
    def _train_forward(self, x):
        # Training forward pass
        pass
    
    def _eval_forward(self, x):
        # Evaluation forward pass
        pass
    
    def _on_enter_training_mode(self):
        # Handle training mode entry
        pass
    
    def _on_enter_eval_mode(self):
        # Handle evaluation mode entry
        pass
    
    def _perform_deployment(self):
        # Handle deployment
        pass
```

### Abstract Methods

Subclasses must implement:

- **`_train_forward(x)`**: Training forward pass
- **`_eval_forward(x)`**: Evaluation forward pass  
- **`_on_enter_training_mode()`**: Training mode entry handler
- **`_on_enter_eval_mode()`**: Evaluation mode entry handler
- **`_perform_deployment()`**: Deployment handler

### Properties

- **quant_config** (`BitQuantConfig`): Quantization configuration
- **is_deployed** (`bool`): Whether layer has been deployed

## Layer Registry

Register custom layers for use with model builders.

```python
from bitlayers import register_layer

@register_layer("CustomBitLinear")
class CustomBitLinear(BitLayerBase):
    # Implementation
    pass
```

### Usage

```python
from bitlayers import LAYER_REGISTRY

# Get registered layer
CustomLayer = LAYER_REGISTRY['CustomBitLinear']
layer = CustomLayer(in_features=10, out_features=5)
```

## Memory Analysis

### estimate_memory_usage

Estimate memory usage of a quantized layer.

```python
from bitlayers.utils import estimate_memory_usage

# Estimate memory for layer
memory_info = estimate_memory_usage(layer)
print(f"Original: {memory_info['original']} bytes")
print(f"Quantized: {memory_info['quantized']} bytes")
print(f"Savings: {memory_info['savings']:.1%}")
```

### compare_layers

Compare memory usage between regular and quantized layers.

```python
from bitlayers.utils import compare_layers
import torch.nn as nn

# Compare layers
regular_layer = nn.Linear(1000, 1000)
quantized_layer = BitLinear(1000, 1000, quant_config=config)

comparison = compare_layers(regular_layer, quantized_layer)
print(f"Memory reduction: {comparison['reduction']:.1%}")
print(f"Speed improvement: {comparison['speedup']:.1f}x")
```

## Best Practices

### 1. Configuration
```python
# Use appropriate granularity for your use case
config = BitQuantConfig(
    weight_granularity="per_tensor",      # Faster
    activation_granularity="per_channel"  # More accurate
)
```

### 2. Training vs Evaluation
```python
# Always use eval() for inference
layer.eval()
with torch.no_grad():
    output = layer(input)
```

### 3. Deployment
```python
# Deploy for production inference
layer.deploy()
# Layer is now inference-only
```

### 4. Memory Management
```python
# Check quantization status
if layer.is_deployed:
    print("Layer is deployed (inference-only)")
elif hasattr(layer, 'qweight'):
    print("Layer is quantized (eval mode)")
else:
    print("Layer is in training mode")
```
