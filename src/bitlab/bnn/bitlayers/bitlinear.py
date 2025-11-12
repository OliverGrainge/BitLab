"""Binary linear layer implementations with shared quantization utilities."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitquantizer import BitQuantizer
from bitlab.bnn.functional import bitlinear


class BitLinear(nn.Module):
    """
    A binary neural network linear layer that quantizes weights to {-1, 0, 1}.

    This layer supports two modes:
    1. Training mode: Uses quantized weights with gradient flow
    2. Deployed mode: Uses packed quantized weights for efficient inference

    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include a bias term
        eps: Small epsilon for numerical stability in quantization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ):
        """
        Initialize a binary linear layer with learnable parameters and a quantizer.

        Args:
            in_features: Number of input activations per sample.
            out_features: Number of output activations per sample.
            bias: Whether to include a learnable bias term.
            eps: Small constant added during quantization to avoid division by zero.
            quant_type: String identifier that selects the activation/weight quantization pair.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.quant_type = quant_type

        # Initialize parameters
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Initialize weights and quantizer
        self._init_weights()
        self.quantizer = BitQuantizer(eps=eps, quant_type=quant_type)

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
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
        qs, qw = bitlinear.prepare_weights(self.weight, self.eps, self.quant_type)
        bias_data = self.bias.detach().clone() if self.bias is not None else None
        del self.bias, self.weight

        # Replace parameters with quantized buffers
        self.register_buffer("qws", qs)
        self.register_buffer("qw", qw)
        self.register_buffer("bias", bias_data)

        # Switch to optimized forward pass
        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantized inference pathway after `deploy` has packed the weights."""
        return bitlinear(x, self.qws, self.qw, self.bias, self.eps, self.quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        dqx, dqw = self.quantizer(x, self.weight)
        return F.linear(dqx, dqw, self.bias)

    def __repr__(self):
        return f"BitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, quant_type={self.quant_type})"
