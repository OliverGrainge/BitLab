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
    # Return dummy inverse scale of 1.0 for API consistency
    inv_scale = torch.tensor(1.0, dtype=torch.bfloat16, device=x.device)
    return inv_scale, qx


def quantize_act_af16(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert to float16 (no actual quantization)."""
    orig_dtype = x.dtype
    qx = x.to(torch.float16)
    qx = qx.to(orig_dtype)
    # Return dummy inverse scale of 1.0 for API consistency
    inv_scale = torch.tensor(1.0, dtype=torch.float16, device=x.device)
    return inv_scale, qx


def quantize_act_ai8pt(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations using ai8pt (int8 per-tensor) scheme.

    Single scale factor for entire tensor.

    Args:
        x: Activation tensor of any shape
        eps: Minimum scale value to prevent division by zero

    Returns:
        inv_scale: Inverse scale tensor (scalar)
        qx: Quantized tensor with same shape as input
    """
    scale = x.abs().amax() / 127.0
    scale = scale.clamp(min=eps)
    inv_scale = 1.0 / scale
    qx = (x * inv_scale).round().clamp(-127, 127)
    return inv_scale, qx


def quantize_act_ai8ptk(
    x: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: (inv_scale, qtensor)
      - inv_scale: multiplier applied to activations before rounding (i.e. 1/scale)
      - qtensor: quantized values (kept in same dtype as input to avoid dtype surprises)
    """
    if x.ndim not in (2, 3):
        raise ValueError(f"ai8ptk expects 2D or 3D tensor, got {x.ndim}D")

    orig_dtype = x.dtype
    x_float = x.float()

    maxval = x_float.abs().amax(dim=-1, keepdim=True).clamp(min=eps)  # shape (...,1)
    inv_scale = 127.0 / maxval  # multiplier applied before rounding
    q = (
        (x_float * inv_scale).round().clamp(-128, 127)
    )  # integer-valued tensor (still float dtype)

    # keep qtensor in same dtype as input to avoid dequantize casting surprises
    return inv_scale.to(orig_dtype), q.to(orig_dtype)


def quantize_act_ai8pc(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize conv activations using ai8pc (int8 per-channel) scheme.

    Args:
        x: Activation tensor [batch, channels, height, width]
        eps: Minimum scale value to prevent division by zero

    Returns:
        inv_scale: Inverse scale tensor [batch, channels, 1, 1]
        qx: Quantized tensor [batch, channels, height, width]
    """
    if x.ndim != 4:
        raise ValueError(
            f"ai8pc expects 4D conv tensor (batch, channels, height, width), got {x.ndim}D"
        )

    scale = x.abs().amax(dim=(2, 3), keepdim=True) / 127.0
    scale = scale.clamp(min=eps)
    inv_scale = 1.0 / scale
    qx = (x * inv_scale).round().clamp(-127, 127)
    return inv_scale, qx
