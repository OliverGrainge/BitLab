import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple
from typing import Optional
from ..bnn import functional as BF 


def quantize_weight(w: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    qws = w.abs().mean()
    qw = (w / (qws + eps)).round().clamp(-1, 1)
    return qws, qw

def quantize_activation(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    qxs = x.abs().max(dim=-1, keepdim=True).values / 127.0
    qxs = qxs.clamp(min=eps)
    qx = (x / qxs).round().clamp(-127, 127)
    return qxs, qx


class Quantize(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):

        # Per-tensor weight abs-max quantization-dequantization
        qws, qw = quantize_weight(w, eps) 
        qxs, qx = quantize_activation(x, eps)

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




class BitQuantizer: 
    def __init__(self, eps: float = 1e-6): 
        self.eps = eps 
        self.quant_fn = Quantize 

    def __call__(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quant_fn.apply(x, w, self.eps)
