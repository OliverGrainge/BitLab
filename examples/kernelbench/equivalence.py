#!/usr/bin/env python3
"""
Equivalence test to verify that BitLinear kernels and PyTorch fallback produce identical results.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add the src directory to the path to import bitlab modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Import bitlab modules
import bitlab.bnn.functional as BF

# Direct import of compiled kernel
try:
    from torch.utils.cpp_extension import load

    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(
        current_dir,
        "..",
        "..",
        "src",
        "bitlab",
        "bnn",
        "functional",
        "kernels",
        "cpu",
        "bitlinear_kernel.cpp",
    )

    bitlinear_cpu_module = load(
        name="bitlinear_cpu", sources=[kernel_path], extra_cflags=["-O3"], verbose=False
    )
    KERNELS_AVAILABLE = True
except Exception as e:
    print(f"Error: CPU kernels not available: {e}")
    bitlinear_cpu_module = None
    KERNELS_AVAILABLE = False


def reference_bitlinear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reference implementation using PyTorch operations.
    """
    delta = w.abs().mean()
    dqw = delta * (w / (delta + eps)).round().clamp(-1, 1)
    return F.linear(x, dqw, b)


def extension_bitlinear(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Extension implementation using compiled kernels.
    """
    if not KERNELS_AVAILABLE:
        raise RuntimeError(
            "CPU extension kernels are not available. Cannot run equivalence test."
        )

    packed_weight, scale = BF.bitlinear_prepare_weights(w, eps)
    return BF.bitlinear(x, packed_weight, scale, b)


def test_equivalence(
    batch_size: int = 32,
    input_features: int = 128,
    output_features: int = 64,
    eps: float = 1e-8,
    tolerance: float = 1e-4,
) -> bool:
    """
    Test equivalence between reference and extension implementations.

    Args:
        batch_size: Batch size for test
        input_features: Input feature dimension
        output_features: Output feature dimension
        eps: Epsilon for quantization
        tolerance: Tolerance for numerical comparison

    Returns:
        True if outputs are equivalent within tolerance
    """
    print(
        f"Testing equivalence with batch_size={batch_size}, input_features={input_features}, output_features={output_features}"
    )

    # Generate random test data
    torch.manual_seed(42)  # For reproducible results
    x = torch.randn(batch_size, input_features, dtype=torch.float32)
    w = torch.randn(output_features, input_features, dtype=torch.float32)
    b = torch.randn(output_features, dtype=torch.float32)

    print("Running reference implementation...")
    try:
        ref_output = reference_bitlinear(x, w, b, eps)
        print(f"Reference output shape: {ref_output.shape}")
        print(
            f"Reference output range: [{ref_output.min().item():.6f}, {ref_output.max().item():.6f}]"
        )
    except Exception as e:
        print(f"Reference implementation failed: {e}")
        return False

    print("Running extension implementation...")
    try:
        ext_output = extension_bitlinear(x, w, b, eps)
        print(f"Extension output shape: {ext_output.shape}")
        print(
            f"Extension output range: [{ext_output.min().item():.6f}, {ext_output.max().item():.6f}]"
        )
    except Exception as e:
        print(f"Extension implementation failed: {e}")
        return False

    # Compare outputs
    print("Comparing outputs...")
    diff = torch.abs(ref_output - ext_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    print(f"Tolerance: {tolerance}")

    is_equivalent = max_diff < tolerance

    if is_equivalent:
        print("✅ PASS: Outputs are equivalent within tolerance")
    else:
        print("❌ FAIL: Outputs differ beyond tolerance")
        print(f"Difference exceeds tolerance by factor of {max_diff / tolerance:.2f}")

    return is_equivalent


def main():
    """
    Main function to run equivalence tests.
    """
    print("BitLinear Equivalence Test")
    print("=" * 50)

    if not KERNELS_AVAILABLE:
        print("❌ ERROR: CPU extension kernels are not available!")
        print(
            "Please ensure the kernels are properly compiled before running this test."
        )
        sys.exit(1)

    print("✅ CPU extension kernels are available")
    print()

    # Test with different configurations
    test_configs = [
        {"batch_size": 16, "input_features": 64, "output_features": 32},
        {"batch_size": 32, "input_features": 128, "output_features": 64},
        {"batch_size": 8, "input_features": 256, "output_features": 128},
    ]

    all_passed = True

    for i, config in enumerate(test_configs, 1):
        print(f"Test {i}/{len(test_configs)}")
        print("-" * 30)

        passed = test_equivalence(**config)
        all_passed = all_passed and passed

        print()

    print("=" * 50)
    if all_passed:
        print(
            "🎉 ALL TESTS PASSED: Reference and extension implementations are equivalent!"
        )
    else:
        print("💥 SOME TESTS FAILED: Reference and extension implementations differ!")
        sys.exit(1)


if __name__ == "__main__":
    main()
