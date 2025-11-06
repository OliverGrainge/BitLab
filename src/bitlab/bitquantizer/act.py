"""Activation quantization functions."""
from typing import Tuple
import torch
from functools import partial


def quantize_act_ai8pc(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8pc scheme.
    
    Supports both linear (2D) and convolutional (4D) tensors:
    - 2D [batch, features]: per-channel quantization over features
    - 4D [batch, channels, height, width]: per-channel quantization over spatial dims
    """
    if x.ndim == 2:  # Linear: [batch, features]
        qxs = x.abs().max(dim=-1, keepdim=True).values / 127.0
    elif x.ndim == 4:  # Conv: [batch, channels, height, width]
        # Per-channel quantization over spatial dimensions
        qxs = x.abs().amax(dim=(2, 3), keepdim=True) / 127.0
    else:
        raise ValueError(f"Unsupported activation tensor dimension: {x.ndim}. Expected 2D or 4D.")
    
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx


def quantize_act_ai8pg(
    x: torch.Tensor, eps: float = 1e-6, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8pg scheme with group-wise quantization.
    
    Supports both linear (2D) and convolutional (4D) tensors by flattening
    and grouping elements for quantization.
    
    Returns:
        qxs: Scale tensor with shape [num_groups, 1] (unexpanded to save memory)
        qx: Quantized tensor with original shape
    """
    orig_shape = x.shape
    assert x.numel() % group_size == 0, f"Number of elements ({x.numel()}) must be divisible by group size ({group_size})"
    
    # Flatten and reshape to groups (works for any tensor shape)
    x_reshaped = x.reshape(-1, group_size)
    qxs = x_reshaped.abs().max(dim=-1, keepdim=True).values / 127.0
    qxs = qxs.clamp(min=eps)
    qx_reshaped = (x_reshaped / qxs).round().clamp(-127, 127)
    
    # Return unexpanded scales [num_groups, 1] and quantized values in original shape
    qx = qx_reshaped.reshape(orig_shape)
    return qxs, qx


def quantize_act_ai8pg128(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_act_ai8pg(x, eps, group_size=128)


def quantize_act_ai8pg256(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_act_ai8pg(x, eps, group_size=256)