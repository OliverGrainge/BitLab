"""BitQuantizer: Quantization utilities for binary neural networks."""
from .quantizers import (
    QuantizerFunction,
    QuantizerAi8pcWpt,
    QuantizerAi8pg128Wpt,
    QuantizerAi8pg256Wpt,
    dequantize,
)
from .weight import quantize_weight_wpt
from .act import quantize_act_ai8pc, quantize_act_ai8pg, quantize_act_ai8pg128, quantize_act_ai8pg256
from .registry import (
    QUANTIZER_REGISTRY,
    WEIGHT_QUANT_REGISTRY,
    ACT_QUANT_REGISTRY,
    get_quantizer,
    get_weight_quant_fn,
    get_act_quant_fn,
)
import torch
import re
from typing import Tuple, Optional

# Backward compatibility aliases
Quantizer_ai8pc_wpt = QuantizerAi8pcWpt
Quantizer_ai8pg_wpt = QuantizerAi8pg128Wpt  # Default to 128 for backward compatibility


def _parse_quant_type(quant_type: str) -> Tuple[str, str, Optional[int]]:
    """Parse quantizer type string to extract activation type, weight type, and group size.
    
    Examples:
        "ai8pc_wpt" -> ("ai8pc", "wpt", None)
        "ai8pg128_wpt" -> ("ai8pg128", "wpt", 128)
        "ai8pg256_wpt" -> ("ai8pg256", "wpt", 256)
    """
    quant_type = quant_type.lower()
    parts = quant_type.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid quant_type format: {quant_type}. Expected format: '{{act_quant_type}}_{{weight_quant_type}}' (e.g., 'ai8pc_wpt' or 'ai8pg128_wpt')")
    act_quant_type, weight_quant_type = parts
    return act_quant_type, weight_quant_type

    
class BitQuantizer:
    """Main quantizer class that wraps quantization schemes."""
    def __init__(
        self,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt"
    ):
        """Initialize quantizer.
        
        Args:
            eps: Epsilon for numerical stability
            quant_type: Quantization scheme in format "{act_quant_type}_{weight_quant_type}" or 
                       "{act_quant_type}{group_size}_{weight_quant_type}" for pg schemes.
                       Examples: "ai8pc_wpt", "ai8pg128_wpt", "ai8pg256_wpt"
        """
        self.eps = eps
        
        # Parse quant_type to extract act_quant_type, weight_quant_type, and group_size
        self.act_quant_type, self.weight_quant_type = _parse_quant_type(quant_type)
        self.quant_type = quant_type
        
        # Get quantization functions
        self.weight_quant_fn = get_weight_quant_fn(self.weight_quant_type)
        self.act_quant_fn = get_act_quant_fn(self.act_quant_type)

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize activations and weights, then return dequantized tensors.
        
        Args:
            x: Activation tensor
            w: Weight tensor
            
        Returns:
            Tuple of (dequantized_activation, dequantized_weight)
        """
        # Use the generic quantizer function with group_size if needed
        return QuantizerFunction.apply(
            x, w, self.weight_quant_fn, self.act_quant_fn, self.eps
        )


def quantize_weight(w: torch.Tensor, eps: float = 1e-6, weight_quant_type: str = "wpt"): 
    """Quantize weights using the specified scheme."""
    quant_fn = get_weight_quant_fn(weight_quant_type)
    return quant_fn(w, eps)


def quantize_act(x: torch.Tensor, eps: float = 1e-6, act_quant_type: str = "ai8pc"): 
    """Quantize activations using the specified scheme."""
    quant_fn = get_act_quant_fn(act_quant_type)
    return quant_fn(x, eps)


__all__ = [
    "BitQuantizer",
    "quantize_weight",
    "quantize_act",
    "dequantize",
    "QuantizerFunction",
    "QuantizerAi8pcWpt",
    "QuantizerAi8pg128Wpt",
    "QuantizerAi8pg256Wpt",
    "Quantizer_ai8pc_wpt",
    "Quantizer_ai8pg_wpt",
    "quantize_weight_wpt",
    "quantize_act_ai8pc",
    "quantize_act_ai8pg",
    "quantize_act_ai8pg128",
    "quantize_act_ai8pg256",
    "QUANTIZER_REGISTRY",
    "WEIGHT_QUANT_REGISTRY",
    "ACT_QUANT_REGISTRY",
]