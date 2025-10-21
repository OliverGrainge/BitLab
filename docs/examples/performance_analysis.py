#!/usr/bin/env python3
"""
Performance Analysis Example for BitLab

This example demonstrates how to analyze the performance benefits of BitLab:
1. Memory usage comparison
2. Inference speed comparison
3. Model size analysis
4. Accuracy vs compression trade-offs
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from bitcore.config import BitQuantConfig
from bitlayers.bitlinear import BitLinear
from bitmodels.mlp import BitMLPModel, BitMLPConfig

def measure_memory_usage():
    """Measure memory usage of different layer configurations"""
    print("=== Memory Usage Analysis ===")
    
    # Test different layer sizes
    layer_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for in_features, out_features in layer_sizes:
        print(f"\nLayer size: {in_features} -> {out_features}")
        
        # Regular PyTorch layer
        regular_layer = nn.Linear(in_features, out_features)
        regular_memory = regular_layer.weight.numel() * 4  # 4 bytes per float32
        
        # Quantized layer
        config = BitQuantConfig(
            activation_dtype="int8",
            activation_granularity="per_tensor",
            weight_granularity="per_tensor"
        )
        quantized_layer = BitLinear(in_features, out_features, quant_config=config)
        quantized_layer.eval()
        quantized_memory = quantized_layer.qweight.numel() * 1.58 / 8
        
        # Calculate savings
        savings = (1 - quantized_memory / regular_memory) * 100
        
        print(f"  Regular memory: {regular_memory / 1024:.1f} KB")
        print(f"  Quantized memory: {quantized_memory / 1024:.1f} KB")
        print(f"  Memory reduction: {savings:.1f}%")

def measure_inference_speed():
    """Measure inference speed of regular vs quantized layers"""
    print("\n=== Inference Speed Analysis ===")
    
    # Test configuration
    in_features, out_features = 1000, 1000
    batch_size = 32
    num_iterations = 100
    
    # Create layers
    regular_layer = nn.Linear(in_features, out_features)
    regular_layer.eval()
    
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor",
        weight_granularity="per_tensor"
    )
    quantized_layer = BitLinear(in_features, out_features, quant_config=config)
    quantized_layer.eval()
    
    # Test input
    x = torch.randn(batch_size, in_features)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = regular_layer(x)
            _ = quantized_layer(x)
    
    # Measure regular layer
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = regular_layer(x)
    regular_time = time.time() - start_time
    
    # Measure quantized layer
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = quantized_layer(x)
    quantized_time = time.time() - start_time
    
    # Calculate speedup
    speedup = regular_time / quantized_time
    
    print(f"Regular layer time: {regular_time:.4f}s")
    print(f"Quantized layer time: {quantized_time:.4f}s")
    print(f"Speed improvement: {speedup:.2f}x")

def analyze_model_sizes():
    """Analyze model sizes for different configurations"""
    print("\n=== Model Size Analysis ===")
    
    # Test different model configurations
    model_configs = [
        {"n_layers": 2, "hidden_dim": 128, "name": "Small"},
        {"n_layers": 3, "hidden_dim": 256, "name": "Medium"},
        {"n_layers": 4, "hidden_dim": 512, "name": "Large"},
        {"n_layers": 5, "hidden_dim": 1024, "name": "Extra Large"}
    ]
    
    for config_dict in model_configs:
        name = config_dict.pop("name")
        print(f"\n{name} Model:")
        
        # Create model configuration
        model_config = BitMLPConfig(
            n_layers=config_dict["n_layers"],
            in_channels=256,
            hidden_dim=config_dict["hidden_dim"],
            out_channels=10,
            dropout=0.1
        )
        
        # Create model
        model = BitMLPModel(model_config)
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate memory usage
        # Each parameter is 4 bytes in float32, 1.58 bits in quantized
        regular_memory = total_params * 4
        quantized_memory = total_params * 1.58 / 8
        
        savings = (1 - quantized_memory / regular_memory) * 100
        
        print(f"  Parameters: {total_params:,}")
        print(f"  Regular memory: {regular_memory / 1024 / 1024:.1f} MB")
        print(f"  Quantized memory: {quantized_memory / 1024 / 1024:.1f} MB")
        print(f"  Memory reduction: {savings:.1f}%")

def compare_quantization_granularities():
    """Compare different quantization granularities"""
    print("\n=== Quantization Granularity Comparison ===")
    
    # Test different granularities
    granularities = [
        ("per_tensor", "per_tensor", "Fastest"),
        ("per_channel", "per_tensor", "Balanced"),
        ("per_tensor", "per_channel", "More Accurate"),
        ("per_channel", "per_channel", "Most Accurate")
    ]
    
    in_features, out_features = 1000, 1000
    x = torch.randn(32, in_features)
    
    for weight_gran, act_gran, description in granularities:
        print(f"\n{description} ({weight_gran} weights, {act_gran} activations):")
        
        # Create configuration
        config = BitQuantConfig(
            weight_granularity=weight_gran,
            activation_dtype="int8",
            activation_granularity=act_gran
        )
        
        # Create layer
        layer = BitLinear(in_features, out_features, quant_config=config)
        layer.eval()
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = layer(x)
        inference_time = time.time() - start_time
        
        # Calculate memory usage
        regular_memory = layer.weight.numel() * 4
        quantized_memory = layer.qweight.numel() * 1.58 / 8
        savings = (1 - quantized_memory / regular_memory) * 100
        
        print(f"  Inference time: {inference_time:.4f}s")
        print(f"  Memory reduction: {savings:.1f}%")

def benchmark_different_batch_sizes():
    """Benchmark performance with different batch sizes"""
    print("\n=== Batch Size Performance Analysis ===")
    
    batch_sizes = [1, 8, 32, 64, 128]
    in_features, out_features = 1000, 1000
    
    # Create layers
    regular_layer = nn.Linear(in_features, out_features)
    regular_layer.eval()
    
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor",
        weight_granularity="per_tensor"
    )
    quantized_layer = BitLinear(in_features, out_features, quant_config=config)
    quantized_layer.eval()
    
    print("Batch Size | Regular (ms) | Quantized (ms) | Speedup")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, in_features)
        
        # Measure regular layer
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = regular_layer(x)
        regular_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Measure quantized layer
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = quantized_layer(x)
        quantized_time = (time.time() - start_time) * 1000  # Convert to ms
        
        speedup = regular_time / quantized_time
        
        print(f"{batch_size:10d} | {regular_time:11.2f} | {quantized_time:13.2f} | {speedup:7.2f}x")

def main():
    """Run all performance analyses"""
    print("BitLab Performance Analysis")
    print("=" * 50)
    
    measure_memory_usage()
    measure_inference_speed()
    analyze_model_sizes()
    compare_quantization_granularities()
    benchmark_different_batch_sizes()
    
    print("\n" + "=" * 50)
    print("Performance analysis completed!")

if __name__ == "__main__":
    main()
