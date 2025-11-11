"""Autograd helpers for BitLab quantization schemes."""

import math
import sys
from typing import Callable, Tuple

import torch
from torch.autograd import Function

Tensor = torch.Tensor
QuantFn = Callable[[Tensor, float], Tuple[Tensor, Tensor]]

__all__ = ["QuantizerFunction", "NoQuantizer", "dequantize", "build_quantizer_class"]


def dequantize(scale: Tensor, qtensor: Tensor) -> Tensor:
    """Expand scale as needed and reconstruct the floating-point tensor.
    
    Supports various quantization schemes:
    - Per-tensor (scalar scale)
    - Per-token/channel (scale with keepdim=True dimensions)
    - Per-group (scale smaller than qtensor in last dimension)
    
    Args:
        scale: Scale tensor from quantization
        qtensor: Quantized tensor
        
    Returns:
        Dequantized tensor with same shape as qtensor
    """
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"Expected scale to be torch.Tensor, got {type(scale)}")

    if scale.dtype != qtensor.dtype:
        scale = scale.to(qtensor.dtype)

    # Scalar scale (per-tensor quantization)
    if scale.numel() == 1:
        return scale * qtensor

    # Direct broadcast (per-channel, per-token with keepdim=True)
    # This handles:
    # - 2D: [batch, 1] scale with [batch, features] qtensor
    # - 3D: [batch, seq_length, 1] scale with [batch, seq_length, hidden_dim] qtensor
    # - 4D: [batch, channels, 1, 1] scale with [batch, channels, height, width] qtensor
    try:
        return scale * qtensor
    except RuntimeError:
        pass

    # Per-group quantization (e.g. wpg) where scale < qtensor along last dimension
    if scale.ndim == 2 and qtensor.ndim == 2 and scale.shape[1] < qtensor.shape[1]:
        group_size = math.ceil(qtensor.shape[1] / scale.shape[1])
        expanded = scale.repeat_interleave(group_size, dim=1)[..., : qtensor.shape[1]]
        return expanded * qtensor
    
    # 3D per-group case: [batch, seq_length, num_groups] -> [batch, seq_length, hidden_dim]
    if scale.ndim == 3 and qtensor.ndim == 3 and scale.shape[2] < qtensor.shape[2]:
        group_size = math.ceil(qtensor.shape[2] / scale.shape[2])
        expanded = scale.repeat_interleave(group_size, dim=2)[..., : qtensor.shape[2]]
        return expanded * qtensor

    raise RuntimeError(
        f"Unable to broadcast scale shape {tuple(scale.shape)} with qtensor shape {tuple(qtensor.shape)}"
    )


class QuantizerFunction(Function):
    """Generic quantizer that accepts weight/activation functions at call time."""

    @staticmethod
    def _forward_impl(
        ctx,
        x: Tensor,
        w: Tensor,
        weight_quant_fn: QuantFn,
        act_quant_fn: QuantFn,
        eps: float,
    ) -> Tuple[Tensor, Tensor]:
        qws, qw = weight_quant_fn(w, eps)
        qxs, qx = act_quant_fn(x, eps)

        dqw = dequantize(qws, qw)
        dqx = dequantize(qxs, qx)

        ctx.save_for_backward(x, w)
        return dqx, dqw

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        w: Tensor,
        weight_quant_fn: QuantFn,
        act_quant_fn: QuantFn,
        eps: float = 1e-6,
    ) -> Tuple[Tensor, Tensor]:
        return QuantizerFunction._forward_impl(
            ctx, x, w, weight_quant_fn, act_quant_fn, eps
        )

    @staticmethod
    def _backward_impl(
        ctx, grad_dqx: Tensor, grad_dqw: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Straight-through estimator: pass gradients through.
        return grad_dqx, grad_dqw

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_dqx: Tensor,
        grad_dqw: Tensor,
    ):
        grad_x, grad_w = QuantizerFunction._backward_impl(ctx, grad_dqx, grad_dqw)
        return grad_x, grad_w, None, None, None


class NoQuantizer(Function):
    """Identity quantizer used for 'none' scheme."""

    @staticmethod
    def forward(ctx, x: Tensor, w: Tensor, eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
        return x, w

    @staticmethod
    def backward(ctx, grad_x: Tensor, grad_w: Tensor):
        return grad_x, grad_w, None


def _to_camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def build_quantizer_class(
    act_name: str,
    weight_name: str,
    weight_quant_fn: QuantFn,
    act_quant_fn: QuantFn,
):
    """Create (or retrieve) a torch.autograd.Function for a specific act/weight pair."""
    class_name = f"Quantizer{_to_camel(act_name)}{_to_camel(weight_name)}"
    module = sys.modules[__name__]

    existing = getattr(module, class_name, None)
    if existing is not None:
        return existing

    doc = f"Quantizer combining activation scheme '{act_name}' with weight scheme '{weight_name}'."

    def forward(  # type: ignore[override]
        ctx,
        x: Tensor,
        w: Tensor,
        eps: float = 1e-6,
        _wq: QuantFn = weight_quant_fn,
        _aq: QuantFn = act_quant_fn,
    ) -> Tuple[Tensor, Tensor]:
        return QuantizerFunction._forward_impl(ctx, x, w, _wq, _aq, eps)

    def backward(  # type: ignore[override]
        ctx,
        grad_dqx: Tensor,
        grad_dqw: Tensor,
    ):
        grad_x, grad_w = QuantizerFunction._backward_impl(ctx, grad_dqx, grad_dqw)
        return grad_x, grad_w, None

    quantizer_cls = type(
        class_name,
        (QuantizerFunction,),
        {
            "__doc__": doc,
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )

    setattr(module, class_name, quantizer_cls)
    if class_name not in __all__:
        __all__.append(class_name)
    return quantizer_cls
