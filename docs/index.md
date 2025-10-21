# BitLab Documentation

Welcome to BitLab - Revolutionary 1.58-bit quantization for modern AI! 🚀

## Quick Start

Get up and running with BitLab in under 5 minutes:

```python
import torch
from bitcore.config import BitQuantConfig
from bitlayers.bitlinear import BitLinear

# Create quantization config
config = BitQuantConfig(
    activation_dtype="int8",
    activation_granularity="per_tensor",
    weight_granularity="per_tensor"
)

# Create quantized layer
layer = BitLinear(512, 256, quant_config=config)

# Use the layer
x = torch.randn(32, 512)
output = layer(x)
print(f"Output shape: {output.shape}")
```

## Documentation Structure

- **[Installation Guide](installation.md)** - Detailed setup instructions
- **[API Reference](api/)** - Complete API documentation
- **[Examples](examples/)** - Code examples and tutorials
- **[Benchmarks](benchmarks/)** - Performance results and comparisons

## Key Features

- 🔥 **16x memory reduction** - Deploy larger models on smaller devices
- ⚡ **Faster inference** - Optimized C++ kernels for maximum speed  
- 🎯 **Minimal accuracy loss** - Advanced quantization techniques preserve model performance
- 🔧 **Drop-in replacement** - Seamlessly replace PyTorch layers with quantized versions

## Why 1.58-bit Quantization?

BitLab uses **ternary weights** (-1, 0, +1) to achieve extreme compression without sacrificing too much accuracy. Perfect for edge deployment, mobile AI, and resource-constrained environments!

## Next Steps

1. **Install BitLab**: Follow the [installation guide](installation.md)
2. **Try examples**: Check out the [examples directory](examples/)
3. **Learn more**: Read the [tutorials](tutorials/)
4. **Explore API**: Browse the [API reference](api/)

---

**Happy quantizing!** 🎯