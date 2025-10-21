# Models API Reference

This section documents the quantized model implementations in BitLab.

## AutoBitModel

Automatic model builder that creates quantized models from configurations.

```python
from bitmodels import AutoBitModel
from bitmodels.mlp import BitMLPConfig

# Create model from config
config = BitMLPConfig(
    n_layers=3,
    in_channels=256,
    hidden_dim=128,
    out_channels=10,
    dropout=0.1
)

model = AutoBitModel.from_config(config)
```

### Methods

#### from_config(config)

Create a model from configuration.

```python
model = AutoBitModel.from_config(config)
```

**Parameters:**
- **config**: Model configuration object

**Returns:**
- **model**: Instantiated quantized model

## BitMLPModel

A quantized Multi-Layer Perceptron model.

```python
from bitmodels.mlp import BitMLPModel, BitMLPConfig

# Create configuration
config = BitMLPConfig(
    n_layers=3,
    in_channels=256,
    hidden_dim=128,
    out_channels=10,
    dropout=0.1
)

# Create model
model = BitMLPModel(config)
```

### BitMLPConfig

Configuration for BitMLPModel.

```python
from bitmodels.mlp import BitMLPConfig

config = BitMLPConfig(
    n_layers=3,           # Number of layers
    in_channels=256,      # Input dimension
    hidden_dim=128,       # Hidden layer dimension
    out_channels=10,      # Output dimension
    dropout=0.1           # Dropout rate
)
```

#### Parameters

- **n_layers** (`int`): Number of layers in the network
- **in_channels** (`int`): Number of input features
- **hidden_dim** (`int`): Number of hidden units
- **out_channels** (`int`): Number of output features
- **dropout** (`float`, optional): Dropout rate. Default: `0.0`

### Methods

#### forward(x)

Forward pass through the model.

```python
output = model(x)
```

**Parameters:**
- **x** (`torch.Tensor`): Input tensor of shape `(batch_size, in_channels)`

**Returns:**
- **output** (`torch.Tensor`): Output tensor of shape `(batch_size, out_channels)`

#### get_config()

Get model configuration.

```python
config = model.get_config()
```

**Returns:**
- **config**: Model configuration object

#### get_quant_config()

Get quantization configuration.

```python
quant_config = model.get_quant_config()
```

**Returns:**
- **quant_config**: Quantization configuration dict or None

### Example Usage

```python
import torch
from bitmodels.mlp import BitMLPModel, BitMLPConfig
from bitcore.config import BitQuantConfig

# Create model configuration
model_config = BitMLPConfig(
    n_layers=3,
    in_channels=256,
    hidden_dim=128,
    out_channels=10,
    dropout=0.1
)

# Create quantization configuration
quant_config = BitQuantConfig(
    activation_dtype="int8",
    activation_granularity="per_tensor",
    weight_granularity="per_tensor"
)

# Create model
model = BitMLPModel(model_config, quant_config)

# Use model
x = torch.randn(32, 256)
output = model(x)
print(f"Output shape: {output.shape}")
```

## BitModelBase

Base class for all quantized models.

```python
from bitmodels.base import BitModelBase

class CustomBitModel(BitModelBase):
    def __init__(self, config, quant_config=None):
        super().__init__(config, quant_config)
        # Initialize model layers
    
    def forward(self, x):
        # Implement forward pass
        pass
```

### Parameters

- **config**: Model configuration object
- **quant_config** (`dict`, optional): Quantization configuration

### Abstract Methods

Subclasses must implement:

- **`forward(x)`**: Forward pass through the model

### Methods

#### get_config()

Get model configuration.

```python
config = model.get_config()
```

#### get_quant_config()

Get quantization configuration.

```python
quant_config = model.get_quant_config()
```

## Model Registry

Register custom models for use with AutoBitModel.

```python
from bitmodels import register_model

@register_model("CustomBitModel")
class CustomBitModel(BitModelBase):
    # Implementation
    pass
```

### Usage

```python
from bitmodels import MODEL_REGISTRY

# Get registered model
CustomModel = MODEL_REGISTRY['CustomBitModel']
model = CustomModel(config)
```

## Model Conversion

### Convert PyTorch Model

Convert existing PyTorch models to quantized versions.

```python
from bitmodels.utils import convert_pytorch_model
import torch.nn as nn

# Original PyTorch model
class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Convert to quantized model
original_model = OriginalModel()
quantized_model = convert_pytorch_model(
    original_model,
    quant_config=BitQuantConfig()
)
```

### Replace Layers

Replace specific layers in a model.

```python
from bitmodels.utils import replace_linear_layers
from bitlayers.bitlinear import BitLinear

# Replace all Linear layers with BitLinear
model = replace_linear_layers(
    model,
    layer_class=BitLinear,
    quant_config=BitQuantConfig()
)
```

## Model Analysis

### analyze_model

Analyze model structure and quantization status.

```python
from bitmodels.utils import analyze_model

# Analyze model
analysis = analyze_model(model)
print(f"Total parameters: {analysis['total_params']}")
print(f"Quantized parameters: {analysis['quantized_params']}")
print(f"Memory reduction: {analysis['memory_reduction']:.1%}")
```

### compare_models

Compare original and quantized models.

```python
from bitmodels.utils import compare_models

# Compare models
comparison = compare_models(original_model, quantized_model)
print(f"Accuracy difference: {comparison['accuracy_diff']:.2%}")
print(f"Memory reduction: {comparison['memory_reduction']:.1%}")
print(f"Speed improvement: {comparison['speedup']:.1f}x")
```

## Best Practices

### 1. Model Configuration
```python
# Use appropriate model size for your task
config = BitMLPConfig(
    n_layers=3,           # Start with 3-5 layers
    hidden_dim=128,        # Adjust based on complexity
    dropout=0.1           # Use dropout for regularization
)
```

### 2. Quantization Configuration
```python
# Balance speed and accuracy
quant_config = BitQuantConfig(
    weight_granularity="per_tensor",      # Faster
    activation_granularity="per_channel" # More accurate
)
```

### 3. Training Workflow
```python
# Train in full precision first
model.train()
for epoch in range(num_epochs):
    # Training loop
    pass

# Then quantize for inference
model.eval()
```

### 4. Model Deployment
```python
# Deploy model for production
model.eval()
# Model is now ready for inference
```

### 5. Memory Management
```python
# Check model status
if model.training:
    print("Model is in training mode")
else:
    print("Model is in evaluation mode")
```
