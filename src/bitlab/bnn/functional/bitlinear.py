"""Functional helpers that back the deployment path for binary linear layers."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from bitlab.bitquantizer import (_parse_quant_type, dequantize, quantize_act,
                                 quantize_weight)


class _BitLinearFunctional:
    """Deployment helper for `BitLinear` that handles quantized state."""

    def prepare_weights(
        self, weight: torch.Tensor, eps: float = 1e-6, quant_type: str = "ai8pc_wpt"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return weight scale and quantized tensor for deployment buffers."""
        _, weight_quant_type = _parse_quant_type(quant_type)
        scale, qtensor = quantize_weight(weight, eps, weight_quant_type)
        return scale, qtensor

    def __call__(
        self,
        x: torch.Tensor,
        qws: torch.Tensor,
        qw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ) -> torch.Tensor:
        """
        Execute the deployment-time linear operator using packed quantized weights.

        Args:
            x: Input activation tensor expected in floating point.
            qws: Weight scale tensor produced by `prepare_weights`.
            qw: Packed quantized weight tensor produced by `prepare_weights`.
            bias: Optional bias term stored alongside deployment buffers.
            eps: Numerical stabilizer for activation quantization.
            quant_type: Identifier that selects activation/weight quantization pair.

        Returns:
            Dequantized output tensor resulting from `F.linear`.
        """
        dqweight = dequantize(qws, qw)
        act_quant_type, _ = _parse_quant_type(quant_type)
        qxs, qx = quantize_act(x, eps, act_quant_type)
        dqx = dequantize(qxs, qx)
        return F.linear(dqx, dqweight, bias)


bitlinear = _BitLinearFunctional()
