"""BitQuantizer: Quantization utilities for binary neural networks."""

from typing import Tuple, Union

import torch

from .act import (quantize_act_abf16, quantize_act_af16, quantize_act_ai8pc,
                  quantize_act_ai8pt, quantize_act_ai8ptk)
from .quantizers import (NoQuantizer, QuantizerFunction, build_quantizer_class,
                         dequantize)
from .registry import (ACT_QUANT_REGISTRY, QUANTIZER_REGISTRY,
                       WEIGHT_QUANT_REGISTRY, get_act_quant_fn, get_quantizer,
                       get_weight_quant_fn)
from .weight import (quantize_weight_wbf16, quantize_weight_wf16,
                     quantize_weight_wpc, quantize_weight_wpg,
                     quantize_weight_wpt)

# Canonical quantizer classes for common schemes
QuantizerAi8pcWpt = build_quantizer_class(
    "ai8pc", "wpt", quantize_weight_wpt, quantize_act_ai8pc
)
QuantizerAi8ptWpt = build_quantizer_class(
    "ai8pt", "wpt", quantize_weight_wpt, quantize_act_ai8pt
)
QuantizerAi8ptkWpc = build_quantizer_class(
    "ai8ptk", "wpc", quantize_weight_wpc, quantize_act_ai8ptk
)

# Backward compatibility aliases
Quantizer_ai8pc_wpt = QuantizerAi8pcWpt
Quantizer_ai8pg_wpt = QuantizerAi8ptWpt  # Legacy name retained for compatibility


def _parse_quant_type(quant_type: Union[str, None]) -> Tuple[str, str]:
    """Parse quantizer type string to extract activation and weight identifiers.

    Examples:
        "ai8pc_wpt" -> ("ai8pc", "wpt")
        "ai8pt_wpc" -> ("ai8pt", "wpc")
    """
    if quant_type is None or quant_type.lower() == "none":
        return "none", "none"

    quant_type = quant_type.lower()
    parts = quant_type.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid quant_type format: {quant_type}. Expected format: '{{act_quant_type}}_{{weight_quant_type}}' (e.g., 'ai8pc_wpt' or 'ai8pg128_wpt')"
        )
    act_quant_type, weight_quant_type = parts
    return act_quant_type, weight_quant_type


class BitQuantizer:
    """Main quantizer class that wraps quantization schemes."""

    def __init__(self, eps: float = 1e-6, quant_type: str = "ai8pc_wpt"):
        """Initialize quantizer.

        Args:
            eps: Epsilon for numerical stability
            quant_type: Quantization scheme in format "{act_quant_type}_{weight_quant_type}".
                       Example: "ai8pc_wpt"
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
    "NoQuantizer",
    "build_quantizer_class",
    "QuantizerAi8pcWpt",
    "QuantizerAi8ptWpt",
    "QuantizerAi8ptkWpc",
    "Quantizer_ai8pc_wpt",
    "Quantizer_ai8pg_wpt",
    "quantize_weight_wpt",
    "quantize_weight_wpc",
    "quantize_weight_wpg",
    "quantize_weight_wbf16",
    "quantize_weight_wf16",
    "quantize_act_ai8pc",
    "quantize_act_ai8pt",
    "quantize_act_ai8ptk",
    "quantize_act_abf16",
    "quantize_act_af16",
    "QUANTIZER_REGISTRY",
    "WEIGHT_QUANT_REGISTRY",
    "ACT_QUANT_REGISTRY",
]
