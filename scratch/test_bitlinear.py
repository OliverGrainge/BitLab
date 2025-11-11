"""Binary linear layer implementations with shared quantization utilities."""

from turtle import hideturtle
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitquantizer import BitQuantizer
from bitlab.bnn.functional import bitlinear
from bitlab.bitquantizer import quantize_act, quantize_weight
import bitlab.bnn as bnn

"""Binary linear layer implementations with shared quantization utilities."""

from typing import Optional

"""Binary linear layer implementations with shared quantization utilities."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitquantizer import BitQuantizer
from bitlab.bnn.functional import bitlinear



class AutoBitLinear(nn.Module):
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

    def weight_quant(self, weight: torch.Tensor) -> torch.Tensor:
        dtype = weight.dtype
        weight = weight.float()
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        weight = (weight * scale).round().clamp(-1, 1) / scale
        return weight.to(dtype)

    def act_quant(self, activation: torch.Tensor) -> torch.Tensor:
        dtype = activation.dtype
        activation = activation.float()
        scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        activation = (activation * scale).round().clamp(-128, 127) / scale
        return activation.to(dtype)

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantized inference pathway after `deploy` has packed the weights."""
        return bitlinear(x, self.qws, self.qw, self.bias, self.eps, self.quant_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        weight = self.weight_quant(self.weight)
        x = self.act_quant(x)
        output = F.linear(x, weight, self.bias)
        return output

    def __repr__(self) -> str:
        return f"MyBitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, quant_type={self.quant_type})"





if __name__ == "__main__": 
    seq_len = 1204 
    hidden_size = 2048
    x = torch.randn(12, seq_len, hidden_size) 
    linear = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=False), nn.Linear(hidden_size, hidden_size, bias=False))
    linear_sd = linear.state_dict()
    
    # Create AutoBitLinear for fair comparison
    target = nn.Sequential(
        bnn.BitLinear(hidden_size, hidden_size, bias=False, quant_type="ai8ptk_wpt", eps=1e-5),
        bnn.BitLinear(hidden_size, hidden_size, bias=False, quant_type="ai8ptk_wpt", eps=1e-5),
    )
    reference = nn.Sequential(
        AutoBitLinear(hidden_size, hidden_size, bias=False), 
        AutoBitLinear(hidden_size, hidden_size, bias=False)
    )

    print(reference)
    print(target)

    target.load_state_dict(linear_sd)
    reference.load_state_dict(linear_sd)
    target.eval()
    reference.eval()
    
    y_hat = target(x)
    y = reference(x)
    print("y_hat", y_hat.shape, "y", y.shape)
    print("Difference:", torch.max(torch.abs(y_hat - y)))
