import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from bitlab.bnn import Module
from bitlab.bitquantizer import BitQuantizer
from bitlab.bnn.functional import bitconv2d


class BitConv2d(Module):
    """
    A binary neural network Conv2d layer that quantizes weights and activations.

    This layer supports two modes:
    1. Training mode: Uses quantized weights with gradient flow
    2. Deployed mode: Uses packed quantized weights for efficient inference

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to all four sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input channels to output channels
        bias: Whether to include a bias term
        eps: Small epsilon for numerical stability in quantization
        quant_type: Quantization scheme (e.g., "ai8pc_wpt", "ai8pg128_wpt")
    """

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
        quant_type: str = "ai8pc_wpt"
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.eps = eps
        self.quant_type = quant_type

        # Initialize parameters
        self.weight = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Initialize weights and quantizer
        self._init_weights()
        self.quantizer = BitQuantizer(eps=eps, quant_type=quant_type)

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _deploy(self) -> None:
        """
        Deploy the layer for efficient inference by:
        1. Quantizing and packing weights
        2. Removing original parameters
        3. Switching to optimized forward pass
        """
        # Quantize and pack weights for deployment
        qs, qw = bitconv2d.prepare_weights(self.weight, self.eps, self.quant_type)
        bias_data = self.bias.data if self.bias is not None else None
        del self.bias, self.weight

        # Replace parameters with quantized buffers
        self.register_buffer("qws", qs)
        self.register_buffer("qw", qw)
        self.register_buffer("bias", bias_data)

        # Switch to optimized forward pass
        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        return bitconv2d(
            x, self.qws, self.qw, self.bias,
            self.stride[0], self.padding[0], self.dilation[0], self.groups,
            self.eps, self.quant_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dqx, dqw = self.quantizer(x, self.weight)
        return F.conv2d(dqx, dqw, self.bias, self.stride, self.padding, self.dilation, self.groups)
