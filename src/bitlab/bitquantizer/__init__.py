"""BitQuantizer: Quantization utilities for binary neural networks."""
from .quantizers import (
    QuantizerFunction,
    QuantizerAi8pcWpt,
    QuantizerAi8pgWpt,
    # Backward compatibility aliases
    Quantizer_ai8pc_wpt,
    Quantizer_ai8pg_wpt,
)
from .weight import quantize_weight_wpt
from .act import quantize_act_ai8pc, quantize_act_ai8pg
from .registry import (
    QUANTIZER_REGISTRY,
    WEIGHT_QUANT_REGISTRY,
    ACT_QUANT_REGISTRY,
    get_quantizer,
    get_weight_quant_fn,
    get_act_quant_fn,
)
import torch
from typing import Tuple, Optional


class BitQuantizer:
    """Main quantizer class that wraps quantization schemes."""
    def __init__(
        self,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
        group_size: Optional[int] = None
    ):
        """Initialize quantizer.
        
        Args:
            eps: Epsilon for numerical stability
            quant_type: Quantization scheme in format "{act_type}_{weight_type}" (e.g., "ai8pc_wpt")
            group_size: Group size for group-wise activation quantization (only used for ai8pg schemes)
        """
        self.eps = eps
        self.group_size = group_size
        
        # Parse quant_type: "{act_type}_{weight_type}"
        parts = quant_type.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid quant_type format: {quant_type}. "
                f"Expected format: '{{act_type}}_{{weight_type}}' (e.g., 'ai8pc_wpt')"
            )
        
        act_type, weight_type = parts
        self.act_type = act_type
        self.weight_type = weight_type
        self.quant_type = quant_type
        
        # Get quantization functions
        self.weight_quant_fn = get_weight_quant_fn(weight_type)
        self.act_quant_fn = get_act_quant_fn(act_type)
        
        # For backward compatibility, also store the old-style quantizer class
        # This allows using the registry-based approach if needed
        self._legacy_quant_fn = get_quantizer(quant_type) if quant_type in QUANTIZER_REGISTRY else None

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
        # Prepare group_size for activation quantization
        group_size = None
        if self.act_type == "ai8pg":
            group_size = self.group_size if self.group_size is not None else 128
        
        # Use the generic quantizer function
        return QuantizerFunction.apply(
            x, w, self.weight_quant_fn, self.act_quant_fn, self.eps, group_size
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
    "QuantizerFunction",
    "QuantizerAi8pcWpt",
    "QuantizerAi8pgWpt",
    # Backward compatibility
    "Quantizer_ai8pc_wpt",
    "Quantizer_ai8pg_wpt",
    "quantize_weight_wpt",
    "quantize_act_ai8pc",
    "quantize_act_ai8pg",
    "QUANTIZER_REGISTRY",
    "WEIGHT_QUANT_REGISTRY",
    "ACT_QUANT_REGISTRY",
]
