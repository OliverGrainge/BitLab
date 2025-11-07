#!/bin/bash
# Run multiple CIFAR-10 ResNet18 experiments to compare quantization schemes

echo "=================================="
echo "CIFAR-10 ResNet18 Experiments"
echo "=================================="
echo ""
echo "This script will train ResNet18 with different quantization schemes:"
echo "  1. Standard Conv2d (baseline)"
echo "  2. BitConv2d with ai8pc_wpt"
echo "  3. BitConv2d with ai8pg128_wpt"
echo "  4. BitConv2d with ai8pg256_wpt"
echo ""

# 0. Standard baseline
echo "[1/4] Training standard ResNet18 (baseline)..."
python main.py config_standard.yaml
echo ""

# 1. BitConv2d with ai8pc_wpt
echo "[2/4] Training BitConv2d with ai8pc_wpt quantization..."
python main.py config_ai8pc_wpt.yaml
echo ""

# 2. BitConv2d with ai8pg128_wpt
echo "[3/4] Training BitConv2d with ai8pg128_wpt quantization..."
python main.py config_ai8pg128_wpt.yaml
echo ""

# 3. BitConv2d with ai8pg256_wpt
echo "[4/4] Training BitConv2d with ai8pg256_wpt quantization..."
python main.py config_ai8pg256_wpt.yaml
echo ""

echo "=================================="
echo "All experiments completed!"
echo "=================================="
echo ""
echo "View results at: https://wandb.ai"
echo "Project: bitlab-bitconv-cifar10-imageclass"


