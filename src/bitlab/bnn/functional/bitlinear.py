import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import os

# Try to import pre-compiled kernel first, fall back to JIT compilation if needed
try:
    # First try to import the pre-compiled extension
    from bitlab.bnn.functional.kernels.cpu import bitlinear_cpu as bitlinear_cpu_module
    KERNELS_AVAILABLE = True
except ImportError:
    # Fall back to JIT compilation if pre-compiled module not available
    try:
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(current_dir, "kernels", "cpu", "bitlinear_kernel.cpp")

        bitlinear_cpu_module = load(
            name="bitlinear_cpu",
            sources=[kernel_path],
            extra_cflags=["-O3", "-march=native", "-mtune=native"],
            verbose=False
        )
        KERNELS_AVAILABLE = True
    except Exception as e:
        print(f"Warning: CPU kernels not available: {e}")
        bitlinear_cpu_module = None
        KERNELS_AVAILABLE = False


def bitlinear_prepare_weights(weight: torch.Tensor,
                           eps: float = 0.0) -> Tuple[torch.Tensor, float]:
    """
    Pack weights to ternary format and return scale factor.
    
    Returns (packed_weight_uint8 [O, ceil(I/4)], scale_float)
    """
    if KERNELS_AVAILABLE:
        packed, scale = bitlinear_cpu_module.prepare_weights(weight, eps)
        return packed, float(scale)
    else:
        # Fallback: no packing; emulate by returning identity "packed" and scale=1.0
        return weight.contiguous(), 1.0


def bitlinear(x: torch.Tensor,
              packed_weight: torch.Tensor,
              scale: float,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward with packed ternary weights.
    
    Args:
      x: [B, I] float32
      packed_weight: [O, ceil(I/4)] uint8 when KERNELS_AVAILABLE, else dense [O, I] float32 fallback
      scale: per-tensor scale (float)
      bias: [O] float32 or None
    """
    if KERNELS_AVAILABLE:
        if bias is None:
            return bitlinear_cpu_module.bitlinear_forward(x, packed_weight, float(scale))
        else:
            return bitlinear_cpu_module.bitlinear_forward(x, packed_weight, float(scale), bias)
    else:
        # Fallback: if we don't have kernels, assume packed_weight is actually dense float weights
        return F.linear(x, packed_weight, bias)