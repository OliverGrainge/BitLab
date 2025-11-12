
from typing import Literal

from pydantic import Field

from bitlab.bitmodels.config import BaseBitModelConfig, register_bitconfig


@register_bitconfig("bitnet")
class BitNetConfig(BaseBitModelConfig):
    """
    vocab_size: Size of the tokenizer vocabulary backing the embedding table.
    hidden_size: Transformer hidden dimension used for embeddings and MLPs.
    intermediate_size: Width of the feed-forward hidden projection.
    num_hidden_layers: Number of decoder layers in the transformer stack.
    num_attention_heads: Attention head count for multi-headed self-attention.
    num_key_value_heads: Distinct KV heads used for grouped-query attention.
    head_dim: Per-head dimensionality. Defaults to hidden_size // num_attention_heads.
    hidden_act: Activation used inside the gated feed-forward (e.g. relu2, silu).
    max_position_embeddings: Maximum context length supported by rotary embeddings.
    rms_norm_eps: Epsilon value for RMSNorm layers to ensure numerical stability.
    pad_token_id: ID of the padding token in the tokenizer vocabulary.
    bos_token_id: ID of the beginning-of-sequence token.
    eos_token_id: ID of the end-of-sequence token.
    tie_word_embeddings: Whether to share weights between input and LM head embeddings.
    rope_theta: Rotary positional embedding base frequency.
    attention_bias: Whether to include additive attention biases.
    attention_dropout: Dropout probability applied to attention weights.
    quant_type: Identifier of the Bit quantization scheme for BitLinear layers.
    """

    model_type: Literal["bitnet"] = Field(default="bitnet", frozen=True)
    vocab_size: int = Field(default=128256)
    hidden_size: int = Field(default=2560)
    intermediate_size: int = Field(default=6912)
    num_hidden_layers: int = Field(default=30)
    num_attention_heads: int = Field(default=20)
    num_key_value_heads: int = Field(default=5)
    head_dim: int | None = Field(default=None)
    hidden_act: str = Field(default="relu2")
    max_position_embeddings: int = Field(default=4096)
    rms_norm_eps: float = Field(default=1e-5)
    pad_token_id: int = Field(default=0)
    bos_token_id: int = Field(default=1)
    eos_token_id: int = Field(default=2)
    tie_word_embeddings: bool = Field(default=True)
    rope_theta: float = Field(default=500000.0)
    attention_bias: bool = Field(default=False)
    attention_dropout: float = Field(default=0.0)
    quant_type: str = Field(default="ai8ptk_wpt")