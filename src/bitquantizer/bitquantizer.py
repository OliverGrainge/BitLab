import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple
from typing import Optional
from ..bnn import functional as BF 


class Quantize(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        delta = w.abs().mean()
        dqw = delta * (w / (delta + eps)).round().clamp(-1, 1)
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return x, dqw

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
