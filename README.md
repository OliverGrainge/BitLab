# BitLab

A PyTorch library for binary neural networks with efficient quantization and deployment capabilities.

## Features

- **Binary Neural Networks**: Train neural networks with quantized weights (-1, 0, 1)
- **Efficient Quantization**: BitQuantizer for weight quantization during training
- **Deployment Ready**: Optimized inference with packed weights
- **PyTorch Integration**: Seamless integration with existing PyTorch workflows

## Installation

### From Source

```bash
git clone https://github.com/yourusername/BitLab.git
cd BitLab
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
import bitlab.bnn as bnn
from bitlab.bnn import BitLinear, Module

# Create a binary neural network
class SimpleBNN(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BitLinear(784, 256)
        self.layer2 = BitLinear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Training
model = SimpleBNN()
# ... training code ...

# Deploy for efficient inference
deployed_model = model.deploy()
```

## Testing the Installation

After installation, you can verify everything works:

```python
import bitlab
import bitlab.bnn
import bitlab.bitquantizer
print(f"BitLab version: {bitlab.__version__}")
```

## Components

### BitLinear Layer

A binary linear layer that quantizes weights to {-1, 0, 1} during training and uses packed weights for efficient inference.

```python
layer = BitLinear(in_features=784, out_features=256, bias=True)
```

### BitQuantizer

Handles weight quantization with gradient flow during training.

```python
from bitlab.bitquantizer import BitQuantizer

quantizer = BitQuantizer(eps=1e-6)
```

### Module Base Class

Extended PyTorch Module with deployment capabilities.

```python
class MyModel(Module):
    def __init__(self):
        super().__init__()
        # Define your layers
    
    def forward(self, x):
        # Forward pass
        return x

# Deploy for inference
model = model.deploy()
```

## Examples

See the `examples/` directory for complete training examples including:

- MNIST classification with binary neural networks
- Model deployment and inference optimization

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
