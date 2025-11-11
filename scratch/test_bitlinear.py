from curses import def_prog_mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitlab.bnn as bnn
from bitlab.bitquantizer import BitQuantizer


from transformers import AutoModelForCausalLM, AutoTokenizer


class WeightQuant(torch.autograd.Function):
    """
    Implements a custom autograd function for weight quantization.
    This performs ternary quantization (-1, 0, 1) based on scaling by the
    mean absolute value of the weights. It uses the Straight-Through Estimator
    (STE) for the backward pass.
    """

    @staticmethod
    @torch.compile
    def forward(ctx, weight):
        dtype = weight.dtype
        weight = weight.float()
        scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        weight = (weight * scale).round().clamp(-1, 1) / scale
        return weight.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActQuant(torch.autograd.Function):
    """
    Implements a custom autograd function for activation quantization.
    This performs symmetric 8-bit quantization (to the range [-128, 127])
    based on the maximum absolute value along the last dimension (per-token/row scaling).
    It uses the Straight-Through Estimator (STE) for the backward pass.
    """

    @staticmethod
    @torch.compile
    def forward(ctx, activation):
        dtype = activation.dtype
        activation = activation.float()
        scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        activation = (activation * scale).round().clamp(-128, 127) / scale
        return activation.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class AutoBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        online_quant: bool = True,
        use_rms_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__(in_features, out_features, bias)
        self.online_quant = online_quant
        # Optional RMSNorm
        self.rms_norm = None


    def forward(self, input):
        weight = WeightQuant.apply(self.weight)
        input = ActQuant.apply(input)
        output = F.linear(input, weight, self.bias)
        return output


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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware linear transformation suitable for training."""
        dqx, dqw = self.quantizer(x, self.weight)
        return F.linear(dqx, dqw, self.bias)

    def __repr__(self): 
        return "BBitLinear"



if __name__ == "__main__": 
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T-bf16")
    model_autobitlinear = hf_model.model.layers[0].self_attn.q_proj 
    in_features = model_autobitlinear.in_features
    out_features = model_autobitlinear.out_features
    bias = model_autobitlinear.bias is not None
    model_autobitlinear.float()
    
    autobitlinear = AutoBitLinear(in_features, out_features, bias)
    autobitlinear.load_state_dict(model_autobitlinear.state_dict())
    autobitlinear.eval()
    autobitlinear.float()
    
    bitlinear = bnn.BitLinear(in_features, out_features, bias, eps=1e-5, quant_type="ai8ptk_wpt")
    bitlinear.float()
    bitlinear.load_state_dict(autobitlinear.state_dict())
    bitlinear.eval() 
    autobitlinear.eval()
    
    # Test input
    x = torch.randn(1, 100, in_features)
    
    # Check if weights are identical
    print("Weight difference:", torch.max(torch.abs(bitlinear.weight - autobitlinear.weight)))
    
    # Check quantized weights
    
    with torch.no_grad():
        auto_qw = WeightQuant.apply(autobitlinear.weight)
        bit_qa, bit_qw = bitlinear.quantizer(x, bitlinear.weight)
        print("Quantized weight difference:", torch.max(torch.abs(auto_qw - bit_qw)))
        
        # Check quantized activations
        auto_qa = ActQuant.apply(x)
        print("Quantized activation difference:", torch.max(torch.abs(auto_qa - bit_qa)))
        
        # Check linear operation
        auto_out_manual = F.linear(auto_qa, auto_qw, autobitlinear.bias)
        bit_out_manual = F.linear(bit_qa, bit_qw, bitlinear.bias)
        print("Manual linear difference:", torch.max(torch.abs(auto_out_manual - bit_out_manual)))
    
    # Full forward pass
    y_auto = autobitlinear(x)
    y_bit = bitlinear(x)
    
    print("\nFinal output difference:")
    print("Max:", torch.max(torch.abs(y_bit - y_auto)))
    print("Mean:", torch.mean(torch.abs(y_bit - y_auto)))
    print("Relative error:", torch.max(torch.abs(y_bit - y_auto)) / torch.max(torch.abs(y_auto)))


