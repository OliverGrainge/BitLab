import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

import bitlab.bnn as bnn
from bitlab.bitquantizer import BitQuantizer


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).pow(2)


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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        weight = self.weight_quant(self.weight)
        x = self.act_quant(x)
        output = F.linear(x, weight, self.bias)
        return output

    def __repr__(self) -> str:
        return f"AutoBitLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, eps={self.eps}, quant_type={self.quant_type})"



class BitNetRMSNorm(nn.Module):
    """Reference RMSNorm used by the baseline BitNet MLP."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BitNetMLP(nn.Module):
    """Reference implementation that relies on standard nn.Linear layers."""

    def __init__(self, config: "BitNetConfig"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = AutoBitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = AutoBitLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = AutoBitLinear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = relu2
        self.ffn_sub_norm = BitNetRMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(hidden))


class RMSNorm(nn.Module):
    """Candidate RMSNorm implementation that should match BitNetRMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FFN(nn.Module):
    """Candidate implementation that uses quantized BitLinear layers."""

    def __init__(self, config: "BitNetConfig"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = bnn.BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_type="ai8ptk_wpt",
        )
        self.up_proj = bnn.BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_type="ai8ptk_wpt",
        )
        self.down_proj = bnn.BitLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_type="ai8ptk_wpt",
        )

        self.ffn_sub_norm = RMSNorm(self.intermediate_size, eps=config.rms_norm_eps)
        self.hidden_act = config.hidden_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = F.relu(self.gate_proj(x)).pow(2)
        up = self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(gate_output * up))


class BitNetConfig:
    """Minimal configuration object shared by both implementations."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "relu2",
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act


def _synchronize_parameters(reference: BitNetMLP, candidate: FFN) -> None:
    with torch.no_grad():
        candidate.gate_proj.weight.copy_(reference.gate_proj.weight)
        candidate.up_proj.weight.copy_(reference.up_proj.weight)
        candidate.down_proj.weight.copy_(reference.down_proj.weight)
        candidate.ffn_sub_norm.weight.copy_(reference.ffn_sub_norm.weight)



@pytest.fixture
def config():
    return BitNetConfig(hidden_size=32, intermediate_size=64, rms_norm_eps=1e-6)


@pytest.fixture
def inputs(config):
    torch.manual_seed(0)
    return torch.randn(4, 10, config.hidden_size, requires_grad=True)


def test_rmsnorm_equivalence(config):
    torch.manual_seed(0)
    reference = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    candidate = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    x = torch.randn(3, 5, config.hidden_size)
    assert torch.allclose(reference(x), candidate(x), atol=1e-6, rtol=1e-6)


def test_bitlinear_equivalence(config, inputs): 
    bitlinear = bnn.BitLinear(config.hidden_size, config.intermediate_size, bias=False, quant_type="ai8ptk_wpt")
    reference = AutoBitLinear(config.hidden_size, config.intermediate_size, bias=False)
    reference.load_state_dict(bitlinear.state_dict())
    assert torch.allclose(bitlinear(inputs), reference(inputs), atol=1e-6, rtol=1e-6)



def test_ffn_equivalence(config, inputs):
    torch.manual_seed(0)
    reference = BitNetMLP(config)
    candidate = FFN(config)
    _synchronize_parameters(reference, candidate)
    assert torch.allclose(reference(inputs), candidate(inputs), atol=1e-6, rtol=1e-6)

