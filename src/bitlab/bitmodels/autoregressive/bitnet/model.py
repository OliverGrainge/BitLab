from __future__ import annotations

from typing import Any, ClassVar, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlab.bitmodels.auto import register_bitmodel
from bitlab.bitmodels.autoregressive.bitnet.config import BitNetConfig
from bitlab.bitmodels.base import BaseBitModel
from bitlab.bnn.bitlayers import BitLinear


def _resolve_head_dim(config: BitNetConfig) -> int:
    """Return the per-head dimensionality, deriving it if absent."""
    if config.head_dim is not None:
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization matching the BitNet reference."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype=input_dtype)

    def __repr__(self):
        return f"RMSNorm(hidden_size={self.weight.shape[0]}, eps={self.variance_epsilon})"


class BitNetFeedForward(nn.Module):
    """Feed-forward sub-block built from quantized BitLinear layers."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.gate_proj = BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_type=config.quant_type,
        )
        self.up_proj = BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_type=config.quant_type,
        )
        self.down_proj = BitLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_type=config.quant_type,
        )
        self.ffn_sub_norm = RMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "relu2":
            gate = F.relu(self.gate_proj(hidden_states)).pow(2)
        elif self.hidden_act in {"silu", "swish"}:
            gate = F.silu(self.gate_proj(hidden_states))
        elif self.hidden_act == "gelu":
            gate = F.gelu(self.gate_proj(hidden_states))
        else:
            gate = F.relu(self.gate_proj(hidden_states)).pow(2)

        up = self.up_proj(hidden_states)
        fused = self.ffn_sub_norm(gate * up)
        return self.down_proj(fused)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimensions (used by RoPE)."""
    half = tensor.shape[-1] // 2
    x1 = tensor[..., :half]
    x2 = tensor[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query/key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


def repeat_kv(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped KV tensors to the full attention head count."""
    batch, num_heads, seq_len, head_dim = tensor.shape
    if n_rep == 1:
        return tensor
    tensor = tensor[:, :, None, :, :].expand(batch, num_heads, n_rep, seq_len, head_dim)
    return tensor.reshape(batch, num_heads * n_rep, seq_len, head_dim)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding helper."""

    def __init__(
        self,
        dim: int,
        *,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        value: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_ids = position_ids[:, None, :].float()

        device_type = value.device.type if value.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ pos_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=value.dtype), sin.to(dtype=value.dtype)


class BitNetAttention(nn.Module):
    """Grouped-query self-attention with BitLinear projections."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = _resolve_head_dim(config)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = BitLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            quant_type=config.quant_type,
        )
        self.k_proj = BitLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            quant_type=config.quant_type,
        )
        self.v_proj = BitLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            quant_type=config.quant_type,
        )
        self.o_proj = BitLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_type=config.quant_type,
        )
        self.attn_sub_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present = (key, value) if use_cache else None

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        attn_output = self.attn_sub_norm(attn_output)
        attn_output = self.o_proj(attn_output)

        return attn_output, present


class BitNetDecoderLayer(nn.Module):
    """Single transformer decoder block with pre-norm layout."""

    def __init__(self, config: BitNetConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetFeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class BitNetDecoder(nn.Module):
    """BitNet transformer backbone with rotary embeddings and KV cache support."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.head_dim = _resolve_head_dim(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            BitNetDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_length = input_ids.shape
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                seq_length + past_length,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if attention_mask is None:
            if past_length > 0:
                causal_mask = torch.zeros(
                    (batch_size, 1, seq_length, seq_length + past_length),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                if seq_length > 1:
                    causal_mask[:, :, :, past_length:] = torch.triu(
                        torch.full(
                            (seq_length, seq_length),
                            float("-inf"),
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        ),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0)
            else:
                causal_mask = torch.triu(
                    torch.full(
                        (seq_length, seq_length),
                        float("-inf"),
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                    diagonal=1,
                )
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length)
        else:
            causal_mask = attention_mask

        present_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, layer_present = decoder_layer(
                hidden_states,
                position_embeddings,
                attention_mask=causal_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present_key_values is not None:
                present_key_values.append(layer_present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_key_values


class BitNetLM(nn.Module):
    """BitNet causal language model head on top of the decoder backbone."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BitNetDecoder(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        hidden_states, present_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "past_key_values": present_key_values,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> torch.LongTensor:
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

        for step in range(max_length - input_ids.shape[1]):
            model_inputs = generated if step == 0 else next_token
            outputs = self.forward(model_inputs, past_key_values=past_key_values, use_cache=use_cache)
            if use_cache:
                past_key_values = outputs["past_key_values"]

            logits = outputs["logits"][:, -1, :] / temperature

            if top_k is not None:
                kth_values = torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(logits < kth_values, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break

        return generated


@register_bitmodel("bitnet")
class BitNetForCausalLM(BaseBitModel):
    """BitNet causal language model with registry + config integration."""

    config_cls: ClassVar[type[BitNetConfig]] = BitNetConfig

    def __init__(self, config: Optional[BitNetConfig] = None, **overrides: Any) -> None:
        super().__init__(config=config, **overrides)

    def build_model(self, config: BitNetConfig) -> nn.Module:
        return BitNetLM(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        lm: BitNetLM = self.model  # type: ignore[attr-defined]
        return lm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
        )


def load_weights_from_hf(model: BitNetForCausalLM, hf_model_path: str) -> BitNetForCausalLM:
    """
    Load weights from a Hugging Face BitNet checkpoint into this implementation.

    Any leading ``model.`` prefix on Hugging Face parameter names is stripped before
    loading to match the local module hierarchy.
    """
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    hf_state_dict = hf_model.state_dict()

    mapped_state_dict: dict[str, torch.Tensor] = {}
    for key, value in hf_state_dict.items():
        new_key = key.removeprefix("model.")
        mapped_state_dict[new_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)

    print(f"Loaded weights from {hf_model_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model
