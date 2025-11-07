# CIFAR-10 ResNet18 Image Classification with BitConv2d

This example demonstrates training ResNet18 on CIFAR-10 using quantized convolutional layers (`BitConv2d`) with different quantization schemes.

## Overview

The training script uses the `BitImageClassifierTrainer` (PyTorch Lightning) to train a ResNet18 model on CIFAR-10. The model can use different quantization schemes for the convolutional layers, allowing you to compare their performance.

## Features

- **Lightning DataModule**: Uses `CIFAR10ClassificationDataModule` for clean data loading
- **Parameterized Layers**: Easily switch between standard `Conv2d` and `BitConv2d` layers
- **Multiple Quantization Schemes**: Support for `ai8pc_wpt`, `ai8pg128_wpt`, and `ai8pg256_wpt`
- **W&B Integration**: Automatic logging of metrics, hyperparameters, and model checkpoints
- **Data Augmentation**: Random cropping and horizontal flipping for improved generalization
- **Cosine Annealing**: Learning rate schedule with cosine decay
- **Model Checkpointing**: Saves best models based on validation accuracy

## Directory Structure

```
examples/training/imageclassification/cifar10/
├── main.py                      # Main training script
├── datamodule.py                # CIFAR-10 DataModule
├── config_ai8pc_wpt.yaml        # Config for ai8pc_wpt quantization
├── config_ai8pg128_wpt.yaml     # Config for ai8pg128_wpt quantization
├── config_ai8pg256_wpt.yaml     # Config for ai8pg256_wpt quantization
├── config_standard.yaml         # Config for standard Conv2d (baseline)
├── run_experiments.sh           # Script to run all experiments
├── checkpoints/                 # Model checkpoints (created during training)
├── logs/                        # Training logs and W&B data
└── README.md                    # This file
```

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
python main.py config_ai8pc_wpt.yaml
```

### Multiple Experiments

Run all experiments automatically:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This will train ResNet18 with all quantization schemes sequentially.

## Configuration

Each experiment has its own YAML configuration file. Here's an example structure:

```yaml
# Run settings
run_name: "ai8pc_wpt"
wandb_project: "bitlab-bitconv-cifar10-imageclass"

# Trainer settings
max_epochs: 200
accelerator: "auto"
devices: "auto"
precision: "32-true"

# Data settings
train_batch_size: 128
val_batch_size: 128
num_workers: 4
use_augmentation: true

# Optimizer settings
learning_rate: 0.1
weight_decay: 0.0005
optimizer: "sgd"
sgd_momentum: 0.9

# Scheduler settings
lr_scheduler: "cosine"
lr_warmup_steps: 0

# Model settings
model:
  layer_type: "bitconv"
  quant_type: "ai8pc_wpt"
```

## DataModule

The `CIFAR10ClassificationDataModule` provides:

- **Automatic Download**: Downloads CIFAR-10 dataset automatically
- **Data Augmentation**: Configurable training augmentation (random crop, flip)
- **Proper Normalization**: Uses CIFAR-10 mean and std
- **Efficient Loading**: Supports multi-worker data loading with pin_memory

### Using the DataModule

```python
from datamodule import CIFAR10ClassificationDataModule

datamodule = CIFAR10ClassificationDataModule(
    train_batch_size=128,
    val_batch_size=128,
    num_workers=4,
    use_augmentation=True,
    seed=42,
)

# Use with PyTorch Lightning
trainer.fit(model, datamodule=datamodule)
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

## Advanced Usage

### Custom Quantization Scheme

To add a new quantization scheme:

1. Implement the scheme in `bitlab/bitquantizer/`
2. Register it in `BitQuantizer`
3. Create a new config file
4. Run: `python main.py your_config.yaml`

### Custom Architecture

Modify `main.py` to change the model architecture while maintaining the same training pipeline.

### Test the DataModule

Test the datamodule independently:

```bash
python datamodule.py
```

This will print information about the data loaders and sample batches.

## Deployment

After training, deploy the best model:

```python
import torch
from main import ResNet18, get_conv_layer

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model_ai8pc_wpt-epoch=XX-val_acc=X.XXXX.ckpt")

# Create model
conv_layer = get_conv_layer("bitconv", "ai8pc_wpt")
model = ResNet18(num_classes=10, conv_layer=conv_layer)

# Load weights
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Deploy for inference (converts to quantized form)
model.deploy()
model.eval()
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
train_batch_size: 64  # or 32
val_batch_size: 64    # or 32
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
WANDB_MODE=disabled python main.py config_ai8pc_wpt.yaml
```

### Import Errors

Make sure BitLab is installed:
```bash
pip install -e /path/to/BitLab
```

## References

- ResNet paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- Binary Neural Networks: [Binarized Neural Networks](https://arxiv.org/abs/1602.02830)
- PyTorch Lightning: [https://lightning.ai/](https://lightning.ai/)


