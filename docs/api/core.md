# Core API Reference

This section documents the core functionality of BitLab.

## Configuration

### BitQuantConfig

The main configuration class for quantization settings.

```python
from bitcore.config import BitQuantConfig

config = BitQuantConfig(
    weight_granularity="per_tensor",     # "per_tensor" or "per_channel"
    activation_dtype="int8",             # "float32" or "int8"
    activation_granularity="per_tensor"  # "per_tensor" or "per_channel"
)
```

#### Parameters

- **weight_granularity** (`str`): How weights are quantized
  - `"per_tensor"`: Global quantization for all weights
  - `"per_channel"`: Per-output-channel quantization
- **activation_dtype** (`str`): Data type for quantized activations
  - `"float32"`: Keep activations in float32
  - `"int8"`: Quantize activations to int8
- **activation_granularity** (`str`): How activations are quantized
  - `"per_tensor"`: Global quantization for all activations
  - `"per_channel"`: Per-channel quantization

#### Example Usage

```python
# Per-tensor quantization (faster, less accurate)
config = BitQuantConfig(
    weight_granularity="per_tensor",
    activation_dtype="int8",
    activation_granularity="per_tensor"
)

# Per-channel quantization (slower, more accurate)
config = BitQuantConfig(
    weight_granularity="per_channel",
    activation_dtype="int8", 
    activation_granularity="per_channel"
)
```

## Operations

### bitlinear.forward

The core bitlinear operation that performs quantized linear transformations.

```python
from bitcore.ops import bitlinear

# Training mode (full precision weights)
output = bitlinear.forward(
    x=input_tensor,
    weight=weight_tensor,
    bias=bias_tensor,
    quant_config=config,
    training=True
)

# Evaluation mode (quantized weights)
output = bitlinear.forward(
    x=input_tensor,
    qweight_scale=scale_tensor,
    qweight=quantized_weight_tensor,
    bias=bias_tensor,
    quant_config=config,
    training=False
)
```

#### Parameters

- **x** (`torch.Tensor`): Input tensor of shape `(N, *, in_features)`
- **weight** (`torch.Tensor`, training only): Full precision weights
- **qweight_scale** (`torch.Tensor`, eval only): Quantization scale factors
- **qweight** (`torch.Tensor`, eval only): Quantized weights
- **bias** (`torch.Tensor`, optional): Bias tensor
- **quant_config** (`BitQuantConfig`): Quantization configuration
- **training** (`bool`): Whether in training or evaluation mode

#### Returns

- **output** (`torch.Tensor`): Output tensor of shape `(N, *, out_features)`

### bitlinear.prepare_weights

Prepare weights for quantization.

```python
qweight_scale, qweight = bitlinear.prepare_weights(weight, quant_config)
```

#### Parameters

- **weight** (`torch.Tensor`): Full precision weights to quantize
- **quant_config** (`BitQuantConfig`): Quantization configuration

#### Returns

- **qweight_scale** (`torch.Tensor`): Scale factors for dequantization
- **qweight** (`torch.Tensor`): Quantized weights

## Kernels

### KernelRegistry

Registry for managing and dispatching quantization kernels.

```python
from bitcore.kernels import KernelRegistry

# Get kernel for configuration
kernel = KernelRegistry.get_kernel_for_config(config)

# Get specific kernel by name
kernel = KernelRegistry.get_kernel_by_name("Int8Kernel", config)

# List available kernels
kernels = KernelRegistry.list_available_kernels()
```

#### Methods

- **get_kernel_for_config**(`config`): Get best kernel for configuration
- **get_kernel_by_name**(`name`, `config`): Get specific kernel
- **list_available_kernels**(): List all available kernels

### BitKernelBase

Base class for all quantization kernels.

```python
from bitcore.kernels.base import BitKernelBase

class CustomKernel(BitKernelBase):
    def prepare_weights(self, weight, quant_config):
        # Implement weight preparation
        pass
    
    def __call__(self, x, qweight_scale, qweight, bias=None, quant_config=None):
        # Implement forward pass
        pass
    
    def is_compatible_with(self, quant_config):
        # Check compatibility
        pass
```

## Base Operations

### quantize_weight

Quantize weights using scale factors.

```python
from bitcore.ops.base import quantize_weight

qweight = quantize_weight(weight, scale, quant_config)
```

### quantize_x

Quantize input activations.

```python
from bitcore.ops.base import quantize_x

qx = quantize_x(x, scale, quant_config)
```

### dequantize_weight

Dequantize weights for computation.

```python
from bitcore.ops.base import dequantize_weight

dqweight = dequantize_weight(qweight, scale, quant_config)
```

### dequantize_x

Dequantize activations for computation.

```python
from bitcore.ops.base import dequantize_x

dqx = dequantize_x(qx, scale, quant_config)
```

## Error Handling

BitLab provides specific exceptions for different error conditions:

```python
from bitcore.exceptions import BitLabError, QuantizationError, KernelNotFoundError

try:
    # BitLab operations
    pass
except QuantizationError as e:
    print(f"Quantization failed: {e}")
except KernelNotFoundError as e:
    print(f"No suitable kernel found: {e}")
except BitLabError as e:
    print(f"BitLab error: {e}")
```
