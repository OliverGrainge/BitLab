# MNIST Training Examples

This directory contains training examples for MNIST digit classification using BitLab's quantized layers.

## Available Examples

### 1. MLP with BitLinear (`mlp.py`)

A simple Multi-Layer Perceptron using `BitLinear` layers for fully-connected quantized operations.

**Architecture:**
- Input: 784 (28x28 flattened)
- Hidden layers: 2x BitLinear layers with configurable hidden size
- Output: 10 classes

**Usage:**
```bash
# Basic training
python mlp.py

# Custom hyperparameters
python mlp.py --batch-size 128 --learning-rate 0.001 --num-epochs 10 --hidden-size 256

# Different quantization schemes
python mlp.py --quant-type ai8pc_wpt     # Per-channel activation quantization (default)
python mlp.py --quant-type ai8pg128_wpt  # Per-group (128) activation quantization
python mlp.py --quant-type ai8pg256_wpt  # Per-group (256) activation quantization
```

### 2. CNN with BitConv2d (`conv.py`)

A Convolutional Neural Network using `BitConv2d` layers for quantized 2D convolutions.

**Architecture:**
- Conv1: 1→32 channels (regular Conv2d)
- Conv2: 32→64 channels (BitConv2d, quantized)
- Conv3: 64→64 channels (BitConv2d, quantized, stride=2)
- FC1: 12544→128 (BitLinear, quantized)
- FC2: 128→10 (regular Linear)
- Dropout: 0.25

**Usage:**
```bash
# Basic training
python conv.py

# Custom hyperparameters
python conv.py --batch-size 128 --learning-rate 0.001 --num-epochs 10

# With test evaluation
python conv.py --num-epochs 10 --test

# Different quantization schemes
python conv.py --quant-type ai8pc_wpt     # Per-channel activation quantization (default)
python conv.py --quant-type ai8pg128_wpt  # Per-group (128) activation quantization
python conv.py --quant-type ai8pg256_wpt  # Per-group (256) activation quantization
```

## Command-Line Arguments

### Common Arguments (both scripts)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--quant-type` | str | `ai8pc_wpt` | Quantization scheme to use |
| `--batch-size` | int | 64 | Training batch size |
| `--learning-rate` | float | 0.001 | Learning rate for optimizer |
| `--num-epochs` | int | 10 | Number of training epochs |

### MLP-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hidden-size` | int | 256 | Size of hidden layers |

### CNN-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test` | flag | False | Evaluate on test set after training |

## Quantization Types

BitLab supports multiple quantization schemes:

- **`ai8pc_wpt`**: Activation int8 per-channel, Weight per-tensor (default)
  - Best for most use cases
  - Per-channel quantization for activations
  - Per-tensor quantization for weights
  
- **`ai8pg128_wpt`**: Activation int8 per-group (128), Weight per-tensor
  - Finer-grained activation quantization
  - Useful for very large activation tensors
  
- **`ai8pg256_wpt`**: Activation int8 per-group (256), Weight per-tensor
  - Coarser-grained than pg128
  - Balance between accuracy and memory

## Expected Performance

### MLP Results (10 epochs)
- Training Loss: ~0.05-0.10
- Training Accuracy: ~97-98%
- Parameters: ~460K (hidden_size=256)

### CNN Results (10 epochs)
- Training Loss: ~0.05-0.08
- Training Accuracy: ~98-99%
- Test Accuracy: ~98-99%
- Parameters: ~1.66M

## Data Location

Both scripts download MNIST data to `../../data/` relative to the script location (i.e., `examples/data/`).

## Requirements

- PyTorch
- torchvision
- BitLab (installed from this repository)

## Notes

- The first convolutional/linear layers and final output layers use regular (non-quantized) operations for better accuracy
- Dropout is used in the CNN for regularization
- Both models use Adam optimizer by default
- Models automatically use CUDA if available

