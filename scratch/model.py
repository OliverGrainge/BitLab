import torch 
import torch.nn as nn 
import torch.nn.functional as F
import bitlab.bnn as bnn
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class BitNetConfig:
    """Configuration class for BitNet model"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = None  # Will be computed if None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    quant_type: str = "ai8ptk_wpt"
    hidden_act: str = "relu2"
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - EXACT match to original"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps  # Match original attribute name

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FFN(nn.Module):
    """Feed-forward network with quantized BitLinear layers."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.gate_proj = bnn.BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_type=config.quant_type,
        )
        self.up_proj = bnn.BitLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            quant_type=config.quant_type,
        )
        self.down_proj = bnn.BitLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_type=config.quant_type,
        )

        self.ffn_sub_norm = RMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match HF implementation: act_fn(gate) * up1
        if self.hidden_act == "relu2":
            # For ReLU, BitNet uses squared ReLU
            gate_output = F.relu(self.gate_proj(x)).pow(2)
        elif self.hidden_act == "silu" or self.hidden_act == "swish":
            gate_output = F.silu(self.gate_proj(x))
        elif self.hidden_act == "gelu":
            gate_output = F.gelu(self.gate_proj(x))
        else:
            # Fallback to ReLU squared
            gate_output = F.relu(self.gate_proj(x)).pow(2)
        
        up = self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(gate_output * up))


def create_causal_mask(batch_size, seq_length, kv_seq_length, dtype, device):
    """
    Create causal mask that matches the original implementation
    Returns a 4D mask of shape (batch, 1, seq_length, kv_seq_length)
    """
    # Create causal mask (upper triangular mask)
    mask = torch.full((seq_length, kv_seq_length), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    
    # Expand to (batch, 1, seq_length, kv_seq_length) to match attention weights shape
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return mask


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors for grouped-query attention.
    Goes from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [batch_size, num_attention_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Force float32 for numerical stability
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    """Multi-headed attention with BitLinear projections"""
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
        
        # Quantized projection layers
        self.q_proj = bnn.BitLinear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=False, 
            quant_type=self.quant_type
        )
        self.k_proj = bnn.BitLinear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=False, 
            quant_type=self.quant_type
        )
        self.v_proj = bnn.BitLinear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=False, 
            quant_type=self.quant_type
        )
        self.o_proj = bnn.BitLinear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=False, 
            quant_type=self.quant_type
        )
        
        # BitNet-specific sub-normalization
        self.attn_sub_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project queries, keys, values
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
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        
        # Apply sub-normalization (BitNet-specific)
        attn_output = self.attn_sub_norm(attn_output)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class DecoderLayer(nn.Module):
    """Transformer decoder layer with pre-normalization"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.self_attn = Attention(config)
        self.mlp = FFN(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class BitNetModel(nn.Module):
    """Complete BitNet Transformer Model"""
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Compute rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Create causal mask matching HF implementation
        # HF creates a 4D mask: (batch, 1, seq_len, seq_len) with -inf for masked positions
        if attention_mask is None:
            # Create causal mask (lower triangular)
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), dtype=hidden_states.dtype, device=hidden_states.device),
                diagonal=1
            )
            # Expand to (batch, 1, seq_len, seq_len)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_length, seq_length)
        else:
            causal_mask = attention_mask
        
        # Pass through transformer layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class BitNetForCausalLM(nn.Module):
    """BitNet model for causal language modeling"""
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.model = BitNetModel(config)
        self.vocab_size = config.vocab_size
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        # Get model outputs
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.LongTensor:
        """Simple greedy/sampling generation"""
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated


def load_weights_from_hf(model: BitNetForCausalLM, hf_model_path: str):
    """
    Load weights from a HuggingFace BitNet model.
    This function maps the HuggingFace state dict to your custom model.
    """
    from transformers import AutoModelForCausalLM
    
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
    hf_state_dict = hf_model.state_dict()
    
    # Create mapping from HF keys to your model keys
    state_dict_mapping = {}
    for key in hf_state_dict.keys():
        new_key = key
        # Remove 'model.' prefix if it exists in HF model
        if new_key.startswith('model.'):
            new_key = new_key[6:]  # Remove 'model.'
        
        state_dict_mapping[key] = new_key
    
    # Load weights with mapping
    new_state_dict = {}
    for hf_key, value in hf_state_dict.items():
        new_key = state_dict_mapping.get(hf_key, hf_key)
        new_state_dict[new_key] = value
    
    # Load into your model
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"Loaded weights from {hf_model_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return model


# Example usage
if __name__ == "__main__":
    # Create config
    config = BitNetConfig(
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        hidden_act="relu2",
    )
    
    # Create model
    model = BitNetForCausalLM(config)
    
    # Example forward pass
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Example generation
    generated = model.generate(input_ids[:1], max_length=20, do_sample=False)
    print(f"Generated shape: {generated.shape}")