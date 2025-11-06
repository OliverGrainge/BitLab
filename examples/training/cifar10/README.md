# CIFAR-10 ResNet18 Training with BitConv2d

This example demonstrates training ResNet18 on CIFAR-10 using quantized convolutional layers (`BitConv2d`) with different quantization schemes.

## Overview

The training script uses the `BitImageClassifierTrainer` (PyTorch Lightning) to train a ResNet18 model on CIFAR-10. The model can use different quantization schemes for the convolutional layers, allowing you to compare their performance.

## Features

- **Parameterized Layers**: Easily switch between standard `Conv2d` and `BitConv2d` layers
- **Multiple Quantization Schemes**: Support for `ai8pc_wpt`, `ai8pg128_wpt`, and `ai8pg256_wpt`
- **W&B Integration**: Automatic logging of metrics, hyperparameters, and model checkpoints
- **Data Augmentation**: Random cropping and horizontal flipping for improved generalization
- **Cosine Annealing**: Learning rate schedule with cosine decay
- **Model Checkpointing**: Saves best models based on validation accuracy

## Quantization Schemes

1. **ai8pc_wpt**: Activation INT8 per-channel, Weight per-tensor
   - Per-channel quantization over spatial dimensions for activations
   - Per-tensor quantization for weights

2. **ai8pg128_wpt**: Activation INT8 per-group (128), Weight per-tensor
   - Per-group (group size 128) quantization for activations
   - Per-tensor quantization for weights

3. **ai8pg256_wpt**: Activation INT8 per-group (256), Weight per-tensor
   - Per-group (group size 256) quantization for activations
   - Per-tensor quantization for weights

## Quick Start

### Single Experiment

Run a single experiment with a specific configuration:

```bash
python resnet.py config_ai8pc_wpt.yaml
```

### Multiple Experiments

Run all experiments automatically:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This will train ResNet18 with all three quantization schemes sequentially.

## Configuration Files

Each experiment has its own YAML configuration file:

- `config_ai8pc_wpt.yaml`: ai8pc_wpt quantization
- `config_ai8pg128_wpt.yaml`: ai8pg128_wpt quantization
- `config_ai8pg256_wpt.yaml`: ai8pg256_wpt quantization
- `config_test.yaml`: Quick test configuration (2 epochs)

### Configuration Parameters

```yaml
# Model settings
layer_type: "bitconv"          # Layer type: "standard" or "bitconv"
quant_type: "ai8pc_wpt"        # Quantization scheme

# Training settings
num_epochs: 200                # Number of training epochs
batch_size: 128                # Batch size
num_workers: 4                 # Number of data loading workers
seed: 42                       # Random seed

# Optimizer settings
learning_rate: 0.1             # Initial learning rate
momentum: 0.9                  # SGD momentum
weight_decay: 0.0005           # Weight decay

# Scheduler settings
scheduler: "cosine"            # Learning rate scheduler

# Data settings
data_root: "../../data"        # Path to dataset
num_classes: 10                # Number of output classes

# W&B settings
wandb_project: "bitlab-bitconv-cifar10-imageclass"
wandb_entity: null             # Your W&B username (optional)
run_name: "ai8pc_wpt"          # Experiment name
```

## Expected Results

ResNet18 on CIFAR-10 typically achieves:

- **Standard Conv2d**: ~93-95% test accuracy
- **BitConv2d (quantized)**: Expected slight accuracy drop (~1-3%) with significant model compression

Training takes approximately:
- **RTX 5090**: ~30-40 minutes for 200 epochs
- **RTX 3090**: ~60-80 minutes for 200 epochs

## Monitoring

All experiments are logged to Weights & Biases:

```
Project: bitlab-bitconv-cifar10-imageclass
```

Metrics logged:
- Training/validation loss and accuracy
- Top-1 and Top-5 accuracy
- Learning rate
- Sample predictions (every 10 epochs)
- Confusion matrix (every 10 epochs)

## Output Structure

```
examples/training/cifar10/
├── checkpoints/                                  # Model checkpoints
│   ├── best_model_ai8pc_wpt-epoch=XX-val_acc=X.XXXX.ckpt
│   ├── best_model_ai8pg128_wpt-epoch=XX-val_acc=X.XXXX.ckpt
│   └── best_model_ai8pg256_wpt-epoch=XX-val_acc=X.XXXX.ckpt
├── wandb/                                        # W&B logs
├── resnet.py                                     # Training script
├── config_*.yaml                                 # Configuration files
├── run_experiments.sh                            # Batch experiment script
└── README.md                                     # This file
```

## Advanced Usage

### Custom Quantization Scheme

To add a new quantization scheme:

1. Implement the scheme in `bitlab/bitquantizer/`
2. Register it in `BitQuantizer`
3. Create a new config file
4. Run: `python resnet.py your_config.yaml`

### Custom Architecture

Modify `resnet.py` to change the model architecture while maintaining the same training pipeline.

### Deployment

After training, deploy the best model:

```python
from bitlab.bnn import BitConv2d
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model_ai8pc_wpt-epoch=XX-val_acc=X.XXXX.ckpt")

# Extract model (from Lightning checkpoint)
model = checkpoint['state_dict']

# Deploy for inference (converts to quantized form)
model.deploy()
model.eval()
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
batch_size: 64  # or 32
```

### Slow Data Loading

Increase number of workers:
```yaml
num_workers: 8  # adjust based on CPU cores
```

### W&B Login Issues

Login to W&B before running:
```bash
wandb login
```

Or disable W&B:
```bash
WANDB_MODE=disabled python resnet.py config_ai8pc_wpt.yaml
```

## References

- ResNet paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- Binary Neural Networks: [Binarized Neural Networks](https://arxiv.org/abs/1602.02830)
