"""Activation quantization functions with support for transformers (3D tensors)."""

import math
from typing import Tuple

import torch
from torch import Tensor


def quantize_act_abf16(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert to bfloat16 (no actual quantization)."""
    orig_dtype = x.dtype
    qx = x.to(torch.bfloat16)
    qx = qx.to(orig_dtype)
    # Return dummy scale of 1.0 for API consistency
    qxs = torch.tensor(1.0, dtype=torch.bfloat16, device=x.device)
    return qxs, qx


def quantize_act_af16(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert to float16 (no actual quantization)."""
    orig_dtype = x.dtype
    qx = x.to(torch.float16)
    qx = qx.to(orig_dtype)
    # Return dummy scale of 1.0 for API consistency
    qxs = torch.tensor(1.0, dtype=torch.float16, device=x.device)
    return qxs, qx


def quantize_act_ai8pt(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8pt (int8 per-tensor) scheme.
    
    Single scale factor for entire tensor.

    Args:
        x: Activation tensor of any shape
        eps: Minimum scale value to prevent division by zero
        
    Returns:
        qxs: Scale tensor (scalar)
        qx: Quantized tensor with same shape as input
    """
    qxs = x.abs().amax() / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx


def quantize_act_ai8ptk(
    x: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8ptk (int8 per-token) scheme.
    
    Supports both 2D (batch, features) and 3D (batch, seq_length, hidden_dim) tensors.
    Computes one scale per token (per row in 2D, per sequence position in 3D).

    Args:
        x: Activation tensor 
           - 2D: [batch, features] for linear layers
           - 3D: [batch, seq_length, hidden_dim] for transformers
        eps: Minimum scale value to prevent division by zero
        
    Returns:
        qxs: Scale tensor 
             - 2D input: [batch, 1]
             - 3D input: [batch, seq_length, 1]
        qx: Quantized tensor with same shape as input
    """

    if x.ndim == 2:
        # Linear layer: [batch, features]
        qxs = x.abs().max(dim=-1, keepdim=True).values / 127.0
    elif x.ndim == 3:
        # Transformer: [batch, seq_length, hidden_dim]
        qxs = x.abs().max(dim=-1, keepdim=True).values / 127.0
    else:
        raise ValueError(
            f"ai8ptk expects 2D (linear) or 3D (transformer) tensor, got {x.ndim}D"
        )
    
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-128, 127)
    return qxs, qx


def quantize_act_ai8pc(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize conv activations using ai8pc (int8 per-channel) scheme.

    Args:
        x: Activation tensor [batch, channels, height, width]
        eps: Minimum scale value to prevent division by zero
        
    Returns:
        qxs: Scale tensor [batch, channels, 1, 1]
        qx: Quantized tensor [batch, channels, height, width]
    """
    if x.ndim != 4:
        raise ValueError(
            f"ai8pc expects 4D conv tensor (batch, channels, height, width), got {x.ndim}D"
        )

    qxs = x.abs().amax(dim=(2, 3), keepdim=True) / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx