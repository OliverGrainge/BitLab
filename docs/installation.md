# Installation Guide

This guide will help you install BitLab with optimal performance on your system.

## Prerequisites

- **Python 3.7+** (3.8+ recommended)
- **PyTorch** (any recent version)
- **C++ compiler** for optimal performance:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Linux: `g++` or `clang++`
  - Windows: Visual Studio Build Tools

## Quick Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd BitLab

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install PyTorch
pip install torch

# 4. Build optimized C++ extensions
python build_extensions.py

# 5. Install in development mode
pip install -e .
```

## Installation Options

### Option 1: Full Installation (Recommended)
```bash
# Build C++ extensions for maximum performance
python build_extensions.py
pip install -e .
```

### Option 2: PyTorch-only Installation
```bash
# Skip C++ extensions, use PyTorch fallback
BUILD_CPP_EXTENSIONS=0 pip install -e .
```

### Option 3: Development Installation
```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Platform-Specific Instructions

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install with Homebrew (optional)
brew install python@3.9

# Build extensions
python build_extensions.py
```

### Linux (Ubuntu/Debian)
```bash
# Install build tools
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Build extensions
python build_extensions.py
```

### Windows
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Use conda for easier setup
conda create -n bitlab python=3.9
conda activate bitlab
conda install pytorch -c pytorch
pip install -e .
```

## Verify Installation

```python
python -c "
from bitcore.config import BitQuantConfig
from bitcore.kernels import KernelRegistry
from bitlayers.bitlinear import BitLinear

# Test basic functionality
config = BitQuantConfig()
layer = BitLinear(10, 5, quant_config=config)
print('✅ BitLab installation successful!')

# Test kernel dispatch
kernel = KernelRegistry.get_kernel_for_config(config)
print(f'Kernel: {type(kernel).__name__}')
print(f'Has C++ backend: {kernel.has_cpp_backend}')
"
```

## Troubleshooting

### C++ Extensions Not Building
```bash
# Check compiler
clang++ --version  # macOS
g++ --version      # Linux

# Clean and rebuild
rm -rf build/ bitcore/kernels/bindings/*.so
python build_extensions.py
```

### Import Errors
```bash
# Ensure you're in the right directory
cd /path/to/BitLab

# Install in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Performance Issues
- Ensure C++ extensions are built: `python build_extensions.py`
- Check backend status: `kernel.has_cpp_backend`
- Verify quantization is working: `layer.qweight.dtype == torch.int8`

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/

# Build extensions
python build_extensions.py
```

## Next Steps

Once installed, check out:
- [Quick Start Guide](../README.md)
- [Examples](examples/)
- [API Reference](api/)
