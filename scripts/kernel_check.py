#!/usr/bin/env python3
"""
Simple kernel selection checker.

Shows which kernels are selected for different quantization configurations.
"""

import sys
import os
import logging
from itertools import product

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Suppress warning messages for cleaner output
logging.getLogger('bitcore.kernels.registry').setLevel(logging.ERROR)

from bitcore.config import BitQuantConfig
from bitcore.kernels import KernelRegistry


def check_kernel_selection():
    """Check and display kernel selection for different configurations."""
    
    # Generate all possible configurations
    configs = []
    for activation_dtype in ["float32", "int8"]:
        for activation_granularity in ["per_tensor", "per_channel"]:
            for weight_granularity in ["per_tensor", "per_channel"]:
                configs.append(BitQuantConfig(
                    activation_dtype=activation_dtype,
                    activation_granularity=activation_granularity,
                    weight_granularity=weight_granularity
                ))
    
    print("Kernel Selection for Different Configurations")
    print("=" * 70)
    print()
    
    for i, config in enumerate(configs, 1):
        # Get the selected kernel
        kernel = KernelRegistry.get_kernel_from_config(config)
        
        # Check if it's using C++ backend or fallback
        backend_info = "Reference"
        if hasattr(kernel, 'has_cpp_backend'):
            if kernel.has_cpp_backend:
                backend_info = "C++ optimized"
            else:
                backend_info = "Python fallback"
        
        # Format the configuration nicely with better spacing
        config_str = f"{config.activation_dtype:>6} + {config.activation_granularity:>12} + {config.weight_granularity:>12}"
        
        print(f"{i:2d}. {config_str:<40} → {kernel.__class__.__name__:<25} ({backend_info})")
    
    print()
    print("Legend:")
    print("  C++ optimized  = Using compiled C++ backend")
    print("  Python fallback = Using pure Python implementation") 
    print("  Reference       = Using reference kernel (fallback)")


if __name__ == "__main__":
    check_kernel_selection()
