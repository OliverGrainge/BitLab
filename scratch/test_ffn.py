import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM

import bitlab.bnn as bnn
from bitlab.bitquantizer import BitQuantizer


HF_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).pow(2)


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = relu2
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x):
        down_proj = self.down_proj(
            self.ffn_sub_norm(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )
        return down_proj


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
    """Copy weights from the dense reference implementation into the quantized one."""
    with torch.no_grad():
        candidate.gate_proj.weight.copy_(reference.gate_proj.weight)
        candidate.up_proj.weight.copy_(reference.up_proj.weight)
        candidate.down_proj.weight.copy_(reference.down_proj.weight)
        candidate.ffn_sub_norm.weight.copy_(reference.ffn_sub_norm.weight)


def run_synthetic_equivalence(hidden_size: int = 32, intermediate_size: int = 64) -> None:
    """
    Quick sanity check on small random tensors to ensure the dense and quantized
    implementations match when weights are manually synchronized.
    """
    print("\n=== Synthetic Sanity Check ===")
    config = BitNetConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    torch.manual_seed(0)
    inputs = torch.randn(4, 10, hidden_size)

    reference = BitNetMLP(config)
    candidate = FFN(config)
    _synchronize_parameters(reference, candidate)

    with torch.no_grad():
        ref_out = reference(inputs)
        cand_out = candidate(inputs)

    diff = (ref_out - cand_out).abs()
    print(f"Max diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")


def load_hf_ffn(model_id: str, layer_index: int = 0) -> tuple[BitNetConfig, BitNetMLP, FFN]:
    """
    Load the specified layer's FFN block from the official BitNet model and mirror
    its weights into both the dense reference implementation and our quantized candidate.
    """
    print(f"\nLoading HuggingFace model '{model_id}'...")
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    try:
        hf_mlp = hf_model.model.layers[layer_index].mlp
    except AttributeError as exc:
        raise RuntimeError("Unexpected model structure; couldn't locate MLP block.") from exc

    config = BitNetConfig(
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        hidden_act=hf_config.hidden_act,
    )

    reference = BitNetMLP(config)
    candidate = FFN(config)

    with torch.no_grad():
        reference.gate_proj.weight.copy_(hf_mlp.gate_proj.weight)
        reference.up_proj.weight.copy_(hf_mlp.up_proj.weight)
        reference.down_proj.weight.copy_(hf_mlp.down_proj.weight)
        reference.ffn_sub_norm.weight.copy_(hf_mlp.ffn_sub_norm.weight)

    _synchronize_parameters(reference, candidate)

    return config, reference.eval(), candidate.eval()


def report(name: str, ref_tensor: torch.Tensor, cand_tensor: torch.Tensor) -> None:
    diff = (ref_tensor - cand_tensor).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"{name:>16}: max diff={max_diff:.6e}, mean diff={mean_diff:.6e}")


def run_hf_comparison(
    model_id: str,
    layer_index: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> None:
    """
    Compare the HuggingFace FFN block against the custom quantized implementation
    and print intermediate statistics so we can inspect numerical discrepancies.
    """
    config, reference, candidate = load_hf_ffn(model_id, layer_index)

    torch.manual_seed(seed)
    inputs = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        ref_gate = reference.gate_proj(inputs)
        cand_gate = candidate.gate_proj(inputs)

        ref_up = reference.up_proj(inputs)
        cand_up = candidate.up_proj(inputs)

        ref_act = reference.act_fn(ref_gate)
        cand_act = F.relu(cand_gate).pow(2)

        ref_norm = reference.ffn_sub_norm(ref_act * ref_up)
        cand_norm = candidate.ffn_sub_norm(cand_act * cand_up)

        ref_out = reference.down_proj(ref_norm)
        cand_out = candidate.down_proj(cand_norm)

    print("\n=== HuggingFace BitNet FFN vs Custom BitLinear FFN ===")
    report("gate projection", ref_gate, cand_gate)
    report("up projection", ref_up, cand_up)
    report("activation", ref_act, cand_act)
    report("post sub-norm", ref_norm, cand_norm)
    report("down projection", ref_out, cand_out)

    if not torch.isfinite(ref_out).all() or not torch.isfinite(cand_out).all():
        print("⚠ Non-finite values detected in outputs!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FFN comparison utility.")
    parser.add_argument(
        "--hf-model",
        default=HF_MODEL_ID,
        help="HuggingFace model identifier or local path.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Decoder layer index to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the random test inputs.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length for the random test inputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the small synthetic equivalence check.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.skip_sanity:
        run_synthetic_equivalence()

    run_hf_comparison(
        model_id=args.hf_model,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
 