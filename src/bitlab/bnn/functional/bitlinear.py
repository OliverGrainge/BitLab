import torch
import torch.nn.functional as F
from typing import Optional
import os

# Direct import of compiled kernel
try:
    from torch.utils.cpp_extension import load
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(current_dir, "kernels", "cpu", "bitlinear_kernel.cpp")
    
    bitlinear_cpu_module = load(
        name="bitlinear_cpu",
        sources=[kernel_path],
        extra_cflags=["-O3"],
        verbose=False
    )
    KERNELS_AVAILABLE = True
except Exception as e:
    print(f"Warning: CPU kernels not available: {e}")
    KERNELS_AVAILABLE = False

def bitlinear(x: torch.Tensor, qweight: torch.Tensor, qbias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Optimized linear layer for quantized weights.
    
    Args:
        x: Input tensor
        qweight: Quantized and packed weight tensor
        qbias: Optional quantized bias tensor
        
    Returns:
        Output tensor after linear transformation
    """
    if KERNELS_AVAILABLE:
        # Handle None bias case - create zero bias tensor
        if qbias is None:
            qbias = torch.zeros(qweight.shape[0], dtype=x.dtype, device=x.device)
        return bitlinear_cpu_module.bitlinear_forward(x, qweight, qbias)
    else:
        # Simple fallback
        return F.linear(x, qweight, qbias)

def bitlinear_pack_weights(weight: torch.Tensor, bias: Optional[torch.Tensor] = None, eps: float = 1e-6) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pack weights for deployment by quantizing and optimizing storage.
    
    Args:
        weight: Original weight tensor
        bias: Optional bias tensor
        eps: Small epsilon for numerical stability
        
    Returns:xw
        Tuple of (packed_weight, packed_bias)
    """
    if KERNELS_AVAILABLE:
        packed_weight = bitlinear_cpu_module.pack_weights(weight, eps)
        return packed_weight, bias
    else:
        # Simple fallback
        delta = weight.abs().mean()
        qweight = (weight / (delta + eps)).round().clamp(-1, 1)
        return qweight, bias