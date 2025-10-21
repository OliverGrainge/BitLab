from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BitQuantConfig:
    """
    Configuration for quantization settings of bit models.
    
    Args:
        weight_granularity: How weights are quantized, either per tensor (global) or per channel (row-wise).
        activation_dtype: Data type for quantized activations.
        activation_granularity: How activations are quantized, either per tensor or per channel.
    """
    weight_granularity: Literal["per_tensor", "per_channel"] = "per_tensor"
    # Weights are always ternary (-1, 0, 1) in this framework.

    activation_dtype: Literal["float32"] = "float32"
    activation_granularity: Literal["per_tensor", "per_channel"] = "per_tensor"
