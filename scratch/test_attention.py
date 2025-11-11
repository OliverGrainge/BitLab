import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest

import bitlab.bnn as bnn
from bitlab.bitquantizer import BitQuantizer
from typing import Optional, Callable, Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class BitNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = AutoBitLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = AutoBitLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = AutoBitLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = AutoBitLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward


        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_sub_norm(attn_output)  # diff with Llama
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights






# ============================================================


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

def create_causal_mask(batch_size, seq_length, kv_seq_length, dtype, device):
    """
    FIX: Create causal mask that matches the original implementation
    Returns a 4D mask of shape (batch, 1, seq_length, kv_seq_length)
    """
    # Create causal mask (upper triangular mask)
    mask = torch.full((seq_length, kv_seq_length), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    
    # Expand to (batch, 1, seq_length, kv_seq_length) to match attention weights shape
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return mask

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.quant_type = config.quant_type 
        
        # Match original projection layers exactly
        self.q_proj = bnn.BitLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, quant_type=self.quant_type)
        self.k_proj = bnn.BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, quant_type=self.quant_type)
        self.v_proj = bnn.BitLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, quant_type=self.quant_type)
        self.o_proj = bnn.BitLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False, quant_type=self.quant_type)
        
        # BitNet-specific sub-normalization
        self.attn_sub_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project queries, keys, values - EXACT match to original
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat KV for grouped-query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores - EXACT match to eager_attention_forward
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # FIX: Simplified causal masking that matches the original behavior
        kv_seq_length = key_states.shape[-2]
        
        if attention_mask is not None:
            # Use the provided attention mask (which should already include causal masking)
            causal_mask = attention_mask[:, :, :, :kv_seq_length]
            attn_weights = attn_weights + causal_mask
        else:
            # Create causal mask only if no mask is provided
            causal_mask = create_causal_mask(
                batch_size, seq_length, kv_seq_length, 
                attn_weights.dtype, attn_weights.device
            )
            attn_weights = attn_weights + causal_mask
        
        # Softmax and dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        
        # Apply sub-normalization (BitNet-specific)
        attn_output = self.attn_sub_norm(attn_output)
        attn_output = self.o_proj(attn_output)
        
        return attn_output



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
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.attention_dropout = 0.0
        self.attention_bias = False
        self.attention_implementation = "eager"
        self.quant_type = "ai8ptk_wpt"





@pytest.fixture
def config():
    return BitNetConfig(hidden_size=32, intermediate_size=64, rms_norm_eps=1e-6)


@pytest.fixture
def inputs(config):
    torch.manual_seed(0)
    return torch.randn(4, 10, config.hidden_size, requires_grad=True)


def _make_position_embeddings(batch: int, seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    cos = torch.ones(batch, seq_len, head_dim, device=device, dtype=dtype)
    sin = torch.zeros(batch, seq_len, head_dim, device=device, dtype=dtype)
    return cos, sin


def _make_attention_mask(batch: int, seq_len: int, device: torch.device, dtype: torch.dtype):
    mask = torch.zeros(batch, 1, seq_len, seq_len, device=device, dtype=dtype)
    mask.triu_(diagonal=1).masked_fill_(mask.bool(), float("-inf"))
    return mask


def test_attention_forward(config, inputs):
    attention = BitNetAttention(config, layer_idx=0)
    batch, seq_len, _ = inputs.shape

    cos, sin = _make_position_embeddings(batch, seq_len, attention.head_dim, inputs.device, inputs.dtype)
    mask = _make_attention_mask(batch, seq_len, inputs.device, inputs.dtype)

    output, attn_weights = attention(inputs, (cos, sin), mask)

    assert output.shape == inputs.shape
    assert attn_weights.shape[:3] == (batch, config.num_attention_heads, seq_len)


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


def test_attention_equivalence(config, inputs):
    torch.manual_seed(42)
    reference = BitNetAttention(config, layer_idx=0)

    torch.manual_seed(42)
    candidate = Attention(config)
    candidate.load_state_dict(reference.state_dict())

    batch, seq_len, _ = inputs.shape
    cos, sin = _make_position_embeddings(batch, seq_len, reference.head_dim, inputs.device, inputs.dtype)
    mask = _make_attention_mask(batch, seq_len, inputs.device, inputs.dtype)

    ref_output, _ = reference(inputs, (cos, sin), mask)
    cand_output = candidate(inputs, (cos, sin), mask)

    assert torch.allclose(ref_output, cand_output, atol=1e-5, rtol=1e-5)

