import torch
import torch.nn.functional as F
from typing import Optional



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
    # For now, use standard linear - this is where you'd implement
    # specialized bit-packing/unpacking kernels for deployment
    return F.linear(x, qweight, qbias)


def bitlinear_pack_weights(weight: torch.Tensor, bias: Optional[torch.Tensor] = None, eps: float = 1e-6) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pack weights for deployment by quantizing and optimizing storage.
    
    Args:
        weight: Original weight tensor
        bias: Optional bias tensor
        eps: Small epsilon for numerical stability
        
    Returns:
        Tuple of (packed_weight, packed_bias)
    """
    # Quantize weights to {-1, 0, 1}
    delta = weight.abs().mean()
    qweight = (weight / (delta + eps)).round().clamp(-1, 1)
    
    # Pack quantized weights (this is where you'd implement bit-packing)
    # For now, just return the quantized weights
    packed_weight = qweight
    
    packed_bias = None
    if bias is not None:
        # Quantize bias similarly
        packed_bias = bias
    
    return packed_weight, packed_bias
