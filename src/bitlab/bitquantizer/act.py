"""Activation quantization functions."""
from typing import Tuple
import torch
from functools import partial


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
    assert x.numel() % group_size == 0, "Number of elements must be divisible by group size"
    
    # Reshape to groups
    x_reshaped = x.reshape(-1, group_size)
    qxs = x_reshaped.abs().max(dim=-1, keepdim=True).values / 127.0
    qxs = qxs.clamp(min=eps)
    qx_reshaped = (x_reshaped / qxs).round().clamp(-127, 127)
    
    # Reshape scales to broadcast correctly: [num_groups, 1] -> [num_groups, group_size]
    # Then reshape to original shape
    qxs = qxs.expand(-1, group_size).reshape(orig_shape)
    qx = qx_reshaped.reshape(orig_shape)
    return qxs, qx


def quantize_act_ai8pg128(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_act_ai8pg(x, eps, group_size=128)


def quantize_act_ai8pg256(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_act_ai8pg(x, eps, group_size=256)