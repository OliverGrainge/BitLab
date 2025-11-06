import torch
from torch.autograd import Function
from typing import Tuple
from .quant_fn import (
    quantize_weight_wpt,
    quantize_activation_ai8pc,
    quantize_activation_ai8pg,
)


class Quantizer_ai8pc_wpt(Function):
    """Quantizer for ai8pc_wpt quantization scheme."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        qws, qw = quantize_weight_wpt(w, eps)
        qxs, qx = quantize_activation_ai8pc(x, eps)

        dqw = qws * qw
        dqx = qxs * qx

        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return dqx, dqw

    @staticmethod
    def backward(ctx, grad_output_x, grad_output_dqw):
        grad_x = grad_output_x
        grad_w = grad_output_dqw
        grad_eps = None
        return grad_x, grad_w, grad_eps


class Quantizer_ai8pg_wpt(Function):
    """Quantizer for ai8pg_wpt quantization scheme with group-wise activation quantization."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, group_size: int = 128):
        qws, qw = quantize_weight_wpt(w, eps)
        qxs, qx = quantize_activation_ai8pg(x, eps, group_size)

        dqw = qws * qw
        dqx = qxs * qx

        ctx.save_for_backward(x, w)
        ctx.eps = eps
        ctx.group_size = group_size
        return dqx, dqw

    @staticmethod
    def backward(ctx, grad_output_x, grad_output_dqw):
        grad_x = grad_output_x
        grad_w = grad_output_dqw
        grad_eps = None
        return grad_x, grad_w, grad_eps, None

