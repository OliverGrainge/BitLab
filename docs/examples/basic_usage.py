#!/usr/bin/env python3
"""
Basic Usage Example for BitLab

This example demonstrates the fundamental usage patterns of BitLab:
1. Creating quantized layers
2. Training vs evaluation modes
3. Memory comparison
4. Model conversion
"""

import torch
import torch.nn as nn
from bitcore.config import BitQuantConfig
from bitlayers.bitlinear import BitLinear
from bitmodels.mlp import BitMLPModel, BitMLPConfig

def basic_layer_usage():
    """Demonstrate basic BitLinear usage"""
    print("=== Basic Layer Usage ===")
    
    # Create quantization configuration
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor",
        weight_granularity="per_tensor"
    )
    
    # Create quantized layer
    layer = BitLinear(512, 256, quant_config=config)
    print(f"Created BitLinear layer: {layer.in_features} -> {layer.out_features}")
    
    # Test input
    x = torch.randn(32, 512)
    
    # Training mode (full precision)
    layer.train()
    output_train = layer(x)
    print(f"Training output shape: {output_train.shape}")
    print(f"Training weight dtype: {layer.weight.dtype}")
    
    # Evaluation mode (quantized)
    layer.eval()
    output_eval = layer(x)
    print(f"Evaluation output shape: {output_eval.shape}")
    print(f"Quantized weight dtype: {layer.qweight.dtype}")
    
    # Check memory savings
    original_memory = layer.weight.numel() * 4  # 4 bytes per float32
    quantized_memory = layer.qweight.numel() * 1.58 / 8  # 1.58 bits per weight
    savings = (1 - quantized_memory / original_memory) * 100
    print(f"Memory savings: {savings:.1f}%")

def model_conversion_example():
    """Demonstrate converting PyTorch models to quantized versions"""
    print("\n=== Model Conversion Example ===")
    
    # Original PyTorch model
    class OriginalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create original model
    original_model = OriginalModel()
    print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters())}")
    
    # Create quantized version
    config = BitMLPConfig(
        n_layers=3,
        in_channels=784,
        hidden_dim=256,
        out_channels=10,
        dropout=0.1
    )
    
    quantized_model = BitMLPModel(config)
    print(f"Quantized model parameters: {sum(p.numel() for p in quantized_model.parameters())}")
    
    # Test both models
    x = torch.randn(32, 784)
    
    # Original model
    with torch.no_grad():
        original_output = original_model(x)
    
    # Quantized model
    quantized_model.eval()
    with torch.no_grad():
        quantized_output = quantized_model(x)
    
    print(f"Original output shape: {original_output.shape}")
    print(f"Quantized output shape: {quantized_output.shape}")

def memory_comparison():
    """Compare memory usage between regular and quantized layers"""
    print("\n=== Memory Comparison ===")
    
    # Regular PyTorch layer
    regular_layer = nn.Linear(1000, 1000)
    regular_memory = regular_layer.weight.numel() * 4  # 4 bytes per float32
    
    # Quantized layer
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor",
        weight_granularity="per_tensor"
    )
    quantized_layer = BitLinear(1000, 1000, quant_config=config)
    quantized_layer.eval()  # Enter eval mode to quantize weights
    quantized_memory = quantized_layer.qweight.numel() * 1.58 / 8  # Convert bits to bytes
    
    print(f"Regular layer memory: {regular_memory / 1024:.1f} KB")
    print(f"Quantized layer memory: {quantized_memory / 1024:.1f} KB")
    print(f"Memory reduction: {(1 - quantized_memory / regular_memory) * 100:.1f}%")

def training_workflow():
    """Demonstrate proper training workflow"""
    print("\n=== Training Workflow ===")
    
    # Create model
    config = BitMLPConfig(
        n_layers=3,
        in_channels=256,
        hidden_dim=128,
        out_channels=10,
        dropout=0.1
    )
    
    model = BitMLPModel(config)
    
    # Training phase
    print("Training phase:")
    model.train()
    print(f"Model training mode: {model.training}")
    
    # Simulate training step
    x = torch.randn(32, 256)
    y = torch.randint(0, 10, (32,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    
    print(f"Training loss: {loss.item():.4f}")
    print(f"Output shape: {output.shape}")
    
    # Evaluation phase
    print("\nEvaluation phase:")
    model.eval()
    print(f"Model training mode: {model.training}")
    
    with torch.no_grad():
        eval_output = model(x)
        eval_loss = nn.CrossEntropyLoss()(eval_output, y)
    
    print(f"Evaluation loss: {eval_loss.item():.4f}")
    print(f"Output shape: {eval_output.shape}")

def main():
    """Run all examples"""
    print("BitLab Basic Usage Examples")
    print("=" * 40)
    
    basic_layer_usage()
    model_conversion_example()
    memory_comparison()
    training_workflow()
    
    print("\n" + "=" * 40)
    print("All examples completed successfully!")

if __name__ == "__main__":
    main()
