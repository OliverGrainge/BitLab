"""Activation quantization functions."""
from typing import Tuple
import torch


def quantize_act_ai8pc(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8pc scheme."""
    qxs = x.abs().max(dim=-1, keepdim=True).values / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx


def quantize_act_ai8pg(
    x: torch.Tensor, eps: float = 1e-6, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8pg scheme with group-wise quantization."""
    orig_shape = x.shape
    
    # Reshape to groups
    x_reshaped = x.reshape(-1, group_size)
    qxs = x_reshaped.abs().max(dim=-1, keepdim=True).values / 127.0
    qxs = qxs.clamp(min=eps)
    qx_reshaped = (x_reshaped / qxs).round().clamp(-127, 127)
    
    # Reshape back to original shape
    qxs = qxs.reshape(orig_shape[:-1] + (1,))
    qx = qx_reshaped.reshape(orig_shape)
    return qxs, qx

