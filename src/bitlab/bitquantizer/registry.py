"""Registry for quantization schemes and functions."""
from typing import Dict, Type, Callable
import torch
from .quantizers import QuantizerAi8pcWpt, QuantizerAi8pg128Wpt, QuantizerAi8pg256Wpt, NoQuantizer
from .weight import quantize_weight_wpt
from .act import quantize_act_ai8pc, quantize_act_ai8pg128, quantize_act_ai8pg256

# Quantization scheme registry (combines weight + activation)
QUANTIZER_REGISTRY: Dict[str, Type] = {
    "ai8pc_wpt": QuantizerAi8pcWpt,
    "ai8pg128_wpt": QuantizerAi8pg128Wpt,
    "ai8pg256_wpt": QuantizerAi8pg256Wpt,
    "none": NoQuantizer,
}

WEIGHT_QUANT_REGISTRY: Dict[str, Callable] = {
    "wpt": quantize_weight_wpt,
    "none": lambda x, eps: (
        torch.ones(1, device=x.device, dtype=x.dtype),
        x,
    ),
}

# Activation quantization function registry  
ACT_QUANT_REGISTRY: Dict[str, Callable] = {
    "ai8pc": quantize_act_ai8pc,
    "ai8pg128": quantize_act_ai8pg128,
    "ai8pg256": quantize_act_ai8pg256,
    "none": lambda x, eps: (
        torch.ones(1, device=x.device, dtype=x.dtype),
        x,
    ),
}

def get_quantizer(quant_type: str):
    """Get quantizer class by type."""
    if quant_type not in QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quantizer type: {quant_type}. Available: {list(QUANTIZER_REGISTRY.keys())}")
    return QUANTIZER_REGISTRY[quant_type]

def get_weight_quant_fn(weight_quant_type: str):
    """Get weight quantization function by type."""
    if weight_quant_type not in WEIGHT_QUANT_REGISTRY:
        raise ValueError(f"Unknown weight quant type: {weight_quant_type}. Available: {list(WEIGHT_QUANT_REGISTRY.keys())}")
    return WEIGHT_QUANT_REGISTRY[weight_quant_type]

def get_act_quant_fn(act_quant_type: str):
    """Get activation quantization function by type."""
    if act_quant_type not in ACT_QUANT_REGISTRY:
        raise ValueError(f"Unknown activation quant type: {act_quant_type}. Available: {list(ACT_QUANT_REGISTRY.keys())}")
    return ACT_QUANT_REGISTRY[act_quant_type]

