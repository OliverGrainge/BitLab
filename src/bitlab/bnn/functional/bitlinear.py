# /Users/olivergrainge/github/BitLab/src/bitlab/bnn/functional/bitlinear.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from bitlab.bitquantizer import quantize_weight, quantize_act


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
        quant_type: str = "ai8pc_wpt"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return quantized weights plus scale that the layer can stash for deploy."""
        # Parse quant_type: format is "{act_type}_{weight_type}" (e.g., "ai8pc_wpt")
        weight_quant_type = quant_type.split("_")[-1]  # Extract weight type (e.g., "wpt")
        qws, qw = quantize_weight(weight, eps, weight_quant_type)
        return qws, qw

    def __call__(
        self,
        x: torch.Tensor,
        qws: torch.Tensor,
        qw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt"
    ) -> torch.Tensor:
        dqweight = dequantize(qws, qw)
        # Parse quant_type: format is "{act_type}_{weight_type}" (e.g., "ai8pc_wpt")
        act_quant_type = quant_type.rsplit("_", 1)[0]  # Extract activation type (e.g., "ai8pc" or "ai8pg")
        qxs, qx = quantize_act(x, eps, act_quant_type)
        dqx = dequantize(qxs, qx)
        return F.linear(dqx, dqweight, bias)


bitlinear = _BitLinearFunctional()
