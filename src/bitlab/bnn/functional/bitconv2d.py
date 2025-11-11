"""Functional counterpart used by deployment-ready binary convolution layers."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from bitlab.bitquantizer import (_parse_quant_type, dequantize, quantize_act,
                                 quantize_weight)


class _BitConv2dFunctional:
    """Namespace + callable that mirrors the deployment API used by layers."""

    def prepare_weights(
        self, weight: torch.Tensor, eps: float = 1e-6, quant_type: str = "ai8pc_wpt"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return weight scale and quantized tensor that the layer can stash for deploy."""
        _, weight_quant_type = _parse_quant_type(quant_type)
        scale, qtensor = quantize_weight(weight, eps, weight_quant_type)
        return scale, qtensor

    def __call__(
        self,
        x: torch.Tensor,
        qws: torch.Tensor,
        qw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ) -> torch.Tensor:
        """
        Run deployment-time binary convolution leveraging quantized buffers.

        Args:
            x: Floating-point activation tensor to be quantized.
            qws: Weight scale tensor produced by `prepare_weights`.
            qw: Packed quantized weights tensor.
            bias: Optional bias buffer saved during deployment preparation.
            stride: Convolution stride applied after quantization.
            padding: Padding to apply to the input tensor.
            dilation: Kernel dilation factor for convolution.
            groups: Number of feature groups processed independently.
            eps: Numerical stabilizer for activation quantization.
            quant_type: Selector describing activation/weight quantization pairing.

        Returns:
            Tensor produced by `F.conv2d` after dequantization.
        """
        dqweight = dequantize(qws, qw)
        act_quant_type, _ = _parse_quant_type(quant_type)
        qxs, qx = quantize_act(x, eps, act_quant_type)
        dqx = dequantize(qxs, qx)
        return F.conv2d(dqx, dqweight, bias, stride, padding, dilation, groups)


bitconv2d = _BitConv2dFunctional()
