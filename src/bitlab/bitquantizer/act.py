"""Activation quantization functions."""

from functools import partial
from typing import Tuple

import torch


def quantize_act_abf16(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Convert to bfloat16
    orig_dtype = x.dtype
    qx = x.to(torch.bfloat16)
    qx = qx.to(orig_dtype)
    # Return dummy scale of 1.0 for API consistency
    qxs = torch.tensor(1.0, dtype=torch.bfloat16, device=x.device)
    return qxs, qx


def quantize_act_af16(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    qx = x.to(torch.float16)
    qx = qx.to(orig_dtype)
    # Return dummy scale of 1.0 for API consistency
    qxs = torch.tensor(1.0, dtype=torch.float16, device=x.device)
    return qxs, qx


def quantize_act_ai8pt(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize linear activations using ai8ptk (int8 per-token) scheme.

    Args:
        x: Activation tensor [batch, features]
    """

    qxs = x.abs().amax() / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx


def quantize_act_ai8ptk(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize linear activations using ai8ptk (int8 per-token) scheme.

    Args:
        x: Activation tensor [batch, features]
    """
    assert (
        x.ndim == 2
    ), f"Expected 2D linear activation tensor (in_features, out_features), got {x.ndim}D"

    qxs = x.abs().max(dim=-1, keepdim=True).values / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx


def quantize_act_ai8pc(
    x: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize conv activations using ai8pc (int8 per-channel) scheme.

    Args:
        x: Activation tensor [batch, channels, height, width]
    """
    assert (
        x.ndim == 4
    ), f"Expected 4D conv activation tensor (batch, channels, height, width), got {x.ndim}D"

    qxs = x.abs().amax(dim=(2, 3), keepdim=True) / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx
