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

> Note: Of course we are 1.58 bit weights, weight_dtype not required. 

#### Example Usage

```python
# Per-tensor quantization
config = BitQuantConfig(
    weight_granularity="per_tensor",
    activation_dtype="int8",
    activation_granularity="per_tensor"
)

# Per-channel quantization 
config = BitQuantConfig(
    weight_granularity="per_channel",
    activation_dtype="int8", 
    activation_granularity="per_channel"
)
```

## Operations

### bitlinear.forward

The `bitlinear.forward` function is the central operator used for both training and inference in BitLab’s quantized linear networks. It switches behavior depending on the mode: training (full precision weights, STE gradients for quantization) versus evaluation/inference (fully quantized fast-path). This API is the gateway to core 1.58-bit operations, and is directly used by both `bitlayers` and `bitmodels`.

---

#### Training Mode (`training=True`)

During training, you provide full-precision weights and inputs. BitLab handles quantization *internally*—including straight-through estimator (STE) gradients for discrete parameters—so you can train as usual, without needing to manage quantization yourself.

```python
from bitcore.ops import bitlinear

# Training: Use float weights, auto-quantization, and STE gradient flow
output = bitlinear.forward(
    x=input_tensor,
    weight=weight_tensor,      # full precision
    bias=bias_tensor,
    quant_config=config,
    training=True
)
```

- **weight** is the full-precision trainable tensor.
- Quantization and STE for gradients are handled inside the function.

---

#### Evaluation Mode (`training=False`)

For inference (or deployment), you typically first quantize weights using `prepare_weights`, then call `bitlinear.forward` with quantized weights and scaling. In this mode, BitLab automatically dispatches the most efficient low-level ternary kernel available for your configuration, maximizing performance on your hardware. These kernels leverage fast bit-packing and ternary (1.58-bit) operations, ensuring highly efficient compute during inference.

You do **not** need to choose a kernel manually—BitLab selects it internally based on your `quant_config`. This automatic dispatch guarantees that all inference computations use quantized weights and are executed on the most suitable optimized backend (for example, AVX, CUDA, etc.) when available.

```python
from bitcore.ops import bitlinear

# Weight quantization (done once before inference)
qweight_scale, qweight = bitlinear.prepare_weights(weight_tensor, config)

# Evaluation: Use quantized weights and scale for fast inference
output = bitlinear.forward(
    x=input_tensor,
    qweight_scale=qweight_scale,
    qweight=qweight,
    bias=bias_tensor,
    quant_config=config,
    training=False  # Triggers efficient ternary kernel dispatch
)
```

- All inference computations now use quantized, ternary weights and scale.
- The optimal kernel is selected and dispatched automatically for your configuration and hardware.

---

#### Parameters

- **x** (`torch.Tensor`): Input tensor of shape `(N, *, in_features)`
- **weight** (`torch.Tensor`, required for `training=True`): Full-precision weights for training
- **qweight_scale** (`torch.Tensor`, required for `training=False`): Quantization scale(s) for weights
- **qweight** (`torch.Tensor`, required for `training=False`): Quantized weights (1.58-bit, typically ternary)
- **bias** (`torch.Tensor`, optional): Optional bias to apply
- **quant_config** (`BitQuantConfig`): Configuration detailing quantization schema
- **training** (`bool`): Selects training (full-precision w/ STE) or evaluation (fully quantized) mode


#### Returns

- **output** (`torch.Tensor`): Output tensor of shape `(N, *, out_features)`



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
