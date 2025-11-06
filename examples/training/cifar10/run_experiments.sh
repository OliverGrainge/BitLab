#!/bin/bash
# Run multiple CIFAR-10 ResNet18 experiments to compare BitConv2d quantization types

# Configuration
EPOCHS=200
BATCH_SIZE=128
LR=0.1
SEED=42
LAYER_TYPE="bitconv"
WANDB_PROJECT="bitlab-bitconv-cifar10-imageclass"
DATA_ROOT="../../data"

echo "=================================="
echo "CIFAR-10 ResNet18 Experiments"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Layer Type: $LAYER_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Seed: $SEED"
echo "  WandB Project: $WANDB_PROJECT"
echo ""
echo "Experiments:"
echo "  1. BitConv2d with ai8pc_wpt"
echo "  2. BitConv2d with ai8pg128_wpt"
echo "  3. BitConv2d with ai8pg256_wpt"
echo ""

# 1. BitConv2d with ai8pc_wpt
echo "[1/3] Training BitConv2d with ai8pc_wpt quantization..."
python resnet.py config_ai8pc_wpt.yaml
echo ""

# 2. BitConv2d with ai8pg128_wpt
echo "[2/3] Training BitConv2d with ai8pg128_wpt quantization..."
python resnet.py config_ai8pg128_wpt.yaml
echo ""

# 3. BitConv2d with ai8pg256_wpt
echo "[3/3] Training BitConv2d with ai8pg256_wpt quantization..."
python resnet.py config_ai8pg256_wpt.yaml
echo ""

echo "=================================="
echo "All experiments completed!"
echo "=================================="
echo ""
echo "View results at: https://wandb.ai"
echo "Project: $WANDB_PROJECT"
