# /Users/olivergrainge/github/BitLab/src/bitlab/bnn/functional/bitlinear.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from bitlab.bitquantizer import quantize_weight, quantize_activation


def dequantize(scale: torch.tensor, tensor: torch.tensor): 
    if scale.size() == 1: 
        return scale * tensor 
    elif scale.size() == len(tensor.shape(-1)):
        return scale * tensor 
    elif scale.size() > len(tensor.shape(-1)): 
        orig_shape = tensor.shape
        group_size = tensor.numel() // scale.numel() 
        tensor = tensor.reshape(-1, group_size) 
        tensor = scale * tensor 
        return tensor.reshape_like(orig_shape)


class _BitLinearFunctional:
    """Namespace + callable that mirrors the deployment API used by layers."""

    def prepare_weights(
        self,
        weight: torch.Tensor,
        eps: float = 1e-6,
        quant_type: str = "ai8pc-wpt"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return quantized weights plus scale that the layer can stash for deploy."""
        qws, qw = quantize_weight(weight, eps, quant_type)
        return qws, qw

    def __call__(
        self,
        x: torch.Tensor,
        qws: torch.Tensor,
        qw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        quant_type: str = "ai8pc-wpt"
    ) -> torch.Tensor:
        dqweight = dequantize(qws, qw)
        qxs, qx = quantize_activation(x, eps, quant_type)
        dqx = dequantize(qxs, qx)
        return F.linear(dqx, dqweight, bias)


bitlinear = _BitLinearFunctional()
