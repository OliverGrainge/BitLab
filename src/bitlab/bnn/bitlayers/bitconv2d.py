"""Binary convolutional layer implementations built on BitLab quantization."""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitquantizer import BitQuantizer
from bitlab.bnn.functional import bitconv2d


class BitConv2d(nn.Module):
    """Binary Conv2d that shares a quantization pipeline between training and deployment, supporting packed weight buffers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ):
        """
        Construct a binary convolution layer with learnable parameters and quantizer.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of convolutional filters to produce.
            kernel_size: Spatial size of the convolution kernel.
            stride: Convolution stride for height and width.
            padding: Implicit zero padding applied to the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections (as in grouped convolution).
            bias: Whether to allocate a learnable bias term.
            eps: Small constant injected for quantization stability.
            quant_type: Identifier describing the activation/weight quantization pairing.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups
        self.eps = eps
        self.quant_type = quant_type

        self.weight = nn.Parameter(
            torch.zeros(
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._init_weights()
        self.quantizer = BitQuantizer(eps=eps, quant_type=quant_type)

    def _init_weights(self) -> None:
        """Use Kaiming init for weights and zeros for bias."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode="fan_in", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _deploy(self) -> None:
        """Freeze parameters into quantized buffers and switch to deploy forward."""
        qs, qw = bitconv2d.prepare_weights(self.weight, self.eps, self.quant_type)
        bias_data = self.bias.detach().clone() if self.bias is not None else None
        del self.bias, self.weight

        self.register_buffer("qws", qs)
        self.register_buffer("qw", qw)
        self.register_buffer("bias", bias_data)

        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the packed-weight quantized convolution used during deployment."""
        return bitconv2d(
            x,
            self.qws,
            self.qw,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.eps,
            self.quant_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware convolution suitable for training loops."""
        dqx, dqw = self.quantizer(x, self.weight)
        return F.conv2d(
            dqx, dqw, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
