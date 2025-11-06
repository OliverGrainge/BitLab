"""PyTorch autograd Functions for combined weight and activation quantization."""
import torch
from torch.autograd import Function
from typing import Callable, Optional
from .weight import quantize_weight_wpt
from .act import quantize_act_ai8pc, quantize_act_ai8pg128, quantize_act_ai8pg256


class QuantizerFunction(Function):
    """Generic quantizer that composes weight and activation quantization schemes."""
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        weight_quant_fn: Callable,
        act_quant_fn: Callable,
        eps: float = 1e-6,
        group_size: Optional[int] = None
    ):
        """Forward pass that quantizes weights and activations, then dequantizes.
        
        Args:
            ctx: Context object for saving tensors
            x: Activation tensor
            w: Weight tensor
            weight_quant_fn: Function to quantize weights (w, eps) -> (scale, quantized)
            act_quant_fn: Function to quantize activations (x, eps, group_size) -> (scale, quantized)
            eps: Epsilon for numerical stability
            group_size: Optional group size for group-wise activation quantization
        """
        # Quantize weights
        qws, qw = weight_quant_fn(w, eps)
        
        # Quantize activations (with optional group_size)
        if group_size is not None:
            qxs, qx = act_quant_fn(x, eps, group_size)
        else:
            qxs, qx = act_quant_fn(x, eps)
        
        # Dequantize
        dqw = qws * qw
        dqx = qxs * qx
        
        # Save for backward (straight-through estimator)
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return dqx, dqw
    
    @staticmethod
    def backward(ctx, grad_output_x, grad_output_dqw):
        """Backward pass using straight-through estimator."""
        grad_x = grad_output_x
        grad_w = grad_output_dqw
        # Return None for all non-tensor inputs: weight_quant_fn, act_quant_fn, eps, group_size
        return grad_x, grad_w, None, None, None, None


class QuantizerAi8pcWpt(QuantizerFunction):
    """Quantizer for ai8pc_wpt quantization scheme.
    
    Uses ai8pc activation quantization (per-channel) with wpt weight quantization.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        return QuantizerFunction.forward(
            ctx, x, w, quantize_weight_wpt, quantize_act_ai8pc, eps
        )

    @staticmethod
    def backward(ctx, grad_output_x, grad_output_dqw):
        grad_x, grad_w = QuantizerFunction.backward(ctx, grad_output_x, grad_output_dqw)
        return grad_x, grad_w, None


class QuantizerAi8pg128Wpt(QuantizerFunction):
    """Quantizer for ai8pg_wpt quantization scheme with group-wise activation quantization.
    
    Uses ai8pg activation quantization (per-group) with wpt weight quantization.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, group_size: int = 128):
        return QuantizerFunction.forward(
            ctx, x, w, quantize_weight_wpt, quantize_act_ai8pg128, eps, group_size
        )

    @staticmethod
    def backward(ctx, grad_output_x, grad_output_dqw):
        grad_x, grad_w = QuantizerFunction.backward(ctx, grad_output_x, grad_output_dqw)
        return grad_x, grad_w, None, None



class QuantizerAi8pg256Wpt(QuantizerFunction):
    """Quantizer for ai8pg_wpt quantization scheme with group-wise activation quantization.
    
    Uses ai8pg activation quantization (per-group) with wpt weight quantization.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, group_size: int = 128):
        return QuantizerFunction.forward(
            ctx, x, w, quantize_weight_wpt, quantize_act_ai8pg256, eps, group_size
        )

    @staticmethod
    def backward(ctx, grad_output_x, grad_output_dqw):
        grad_x, grad_w = QuantizerFunction.backward(ctx, grad_output_x, grad_output_dqw)
        return grad_x, grad_w, None, None





