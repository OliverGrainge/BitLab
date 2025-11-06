import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from bitlab.bitquantizer import quantize_weight, quantize_act, dequantize


class _BitConv2dFunctional:
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
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt"
    ) -> torch.Tensor:
        dqweight = dequantize(qws, qw)
        # Parse quant_type: format is "{act_type}_{weight_type}" (e.g., "ai8pc_wpt")
        act_quant_type = quant_type.rsplit("_", 1)[0]  # Extract activation type (e.g., "ai8pc" or "ai8pg")
        qxs, qx = quantize_act(x, eps, act_quant_type)
        dqx = dequantize(qxs, qx)
        return F.conv2d(dqx, dqweight, bias, stride, padding, dilation, groups)


bitconv2d = _BitConv2dFunctional()

