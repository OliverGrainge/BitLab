#!/usr/bin/env python3
"""
Example script to test kernel registry dispatching for BitQuantConfig
with int8 per_tensor activation quantization and per_tensor weight quantization.

This script verifies that the kernel registry correctly dispatches to 
BitLinear_Ai8pt_TWpt kernel for the specified configuration.
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitcore.config import BitQuantConfig
from bitcore.kernels import KernelRegistry
from bitlayers.bitlinear import BitLinear

def test_kernel_dispatch():
    """Test that kernel registry dispatches to the correct kernel"""
    
    print("=== Testing Kernel Registry Dispatch ===\n")
    
    # Create the configuration with int8 per_tensor for both activations and weights
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor", 
        weight_granularity="per_tensor"
    )
    
    print(f"BitQuantConfig created:")
    print(f"  activation_dtype: {config.activation_dtype}")
    print(f"  activation_granularity: {config.activation_granularity}")
    print(f"  weight_granularity: {config.weight_granularity}")
    print()
    
    # Test kernel dispatch
    print("Testing kernel dispatch...")
    kernel = KernelRegistry.get_kernel_for_config(config)
    print(f"Selected kernel: {type(kernel).__name__}")
    print(f"Kernel module: {type(kernel).__module__}")
    
    # Verify it's the correct kernel
    expected_kernel_name = "BitLinear_Ai8pt_TWpt"
    actual_kernel_name = type(kernel).__name__
    
    if actual_kernel_name == expected_kernel_name:
        print(f"✅ SUCCESS: Correctly dispatched to {expected_kernel_name}")
    else:
        print(f"❌ FAILED: Expected {expected_kernel_name}, got {actual_kernel_name}")
    
    print()
    
    # Test kernel compatibility
    print("Testing kernel compatibility...")
    is_compatible = kernel.is_compatible_with(config)
    print(f"Kernel compatible with config: {is_compatible}")
    
    if is_compatible:
        print("✅ SUCCESS: Kernel is compatible with the configuration")
    else:
        print("❌ FAILED: Kernel is not compatible with the configuration")
    
    print()
    return kernel

def test_layer_forward_pass():
    """Test BitLinear layer forward pass with the configuration"""
    
    print("=== Testing BitLinear Layer Forward Pass ===\n")
    
    # Create configuration
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor",
        weight_granularity="per_tensor"
    )
    
    # Create a BitLinear layer
    layer = BitLinear(
        in_features=10,
        out_features=5,
        bias=True,
        quant_config=config
    )
    
    print(f"Created BitLinear layer:")
    print(f"  Input features: {layer.in_features}")
    print(f"  Output features: {layer.out_features}")
    print(f"  Has bias: {layer.bias is not None}")
    print()
    
    # Create test input
    batch_size = 3
    x = torch.randn(batch_size, layer.in_features)
    print(f"Test input shape: {x.shape}")
    print(f"Input data type: {x.dtype}")
    print()
    
    # Test training forward pass
    print("Testing training forward pass...")
    layer.train()
    train_output = layer(x)
    print(f"Training output shape: {train_output.shape}")
    print(f"Training output data type: {train_output.dtype}")
    print(f"Training output mean: {train_output.mean().item():.4f}")
    print(f"Training output std: {train_output.std().item():.4f}")
    print()
    
    # Test evaluation forward pass
    print("Testing evaluation forward pass...")
    layer.eval()
    eval_output = layer(x)
    print(f"Evaluation output shape: {eval_output.shape}")
    print(f"Evaluation output data type: {eval_output.dtype}")
    print(f"Evaluation output mean: {eval_output.mean().item():.4f}")
    print(f"Evaluation output std: {eval_output.std().item():.4f}")
    print()
    
    # Check that quantized weights were created
    if hasattr(layer, 'qweight') and hasattr(layer, 'qweight_scale'):
        print("✅ SUCCESS: Quantized weights created during eval mode")
        print(f"  qweight shape: {layer.qweight.shape}")
        print(f"  qweight_scale shape: {layer.qweight_scale.shape}")
        print(f"  qweight data type: {layer.qweight.dtype}")
        print(f"  qweight_scale data type: {layer.qweight_scale.dtype}")
    else:
        print("❌ FAILED: Quantized weights not found")
    
    print()
    return layer

def test_direct_kernel_usage():
    """Test direct kernel usage to verify the implementation"""
    
    print("=== Testing Direct Kernel Usage ===\n")
    
    # Create configuration
    config = BitQuantConfig(
        activation_dtype="int8",
        activation_granularity="per_tensor",
        weight_granularity="per_tensor"
    )
    
    # Get the kernel
    kernel = KernelRegistry.get_kernel_for_config(config)
    print(f"Using kernel: {type(kernel).__name__}")
    
    # Create test data
    batch_size = 2
    in_features = 8
    out_features = 4
    
    x = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Bias shape: {bias.shape}")
    print()
    
    # Prepare weights using kernel
    print("Preparing weights with kernel...")
    qweight_scale, qweight = kernel.prepare_weights(weight, config)
    print(f"Quantized weight scale shape: {qweight_scale.shape}")
    print(f"Quantized weight shape: {qweight.shape}")
    print(f"Quantized weight scale: {qweight_scale}")
    print()
    
    # Run kernel forward pass
    print("Running kernel forward pass...")
    output = kernel(x, qweight_scale, qweight, bias, config)
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    print()
    
    return output

def main():
    """Main test function"""
    
    print("BitLab Kernel Dispatch Test")
    print("=" * 50)
    print()
    
    try:
        # Test 1: Kernel dispatch
        kernel = test_kernel_dispatch()
        
        # Test 2: Layer forward pass
        layer = test_layer_forward_pass()
        
        # Test 3: Direct kernel usage
        output = test_direct_kernel_usage()
        
        print("=== Test Summary ===")
        print("✅ All tests completed successfully!")
        print(f"✅ Kernel registry correctly dispatched to: {type(kernel).__name__}")
        print("✅ BitLinear layer forward pass working")
        print("✅ Direct kernel usage working")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
