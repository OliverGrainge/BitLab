"""Registry for BitLab quantization schemes."""

from functools import partial
from typing import Callable, Dict, Type

import torch

from .act import (quantize_act_abf16, quantize_act_af16, quantize_act_ai8pc,
                  quantize_act_ai8pt, quantize_act_ai8ptk)
from .quantizers import NoQuantizer, build_quantizer_class
from .weight import (quantize_weight_wbf16, quantize_weight_wf16,
                     quantize_weight_wpc, quantize_weight_wpg,
                     quantize_weight_wpt)

Tensor = torch.Tensor
QuantFn = Callable[[Tensor, float], tuple[Tensor, Tensor]]


def _identity_weight(w: Tensor, eps: float = 1e-6):
    return torch.ones(1, device=w.device, dtype=w.dtype), w


def _identity_act(x: Tensor, eps: float = 1e-6):
    return torch.ones(1, device=x.device, dtype=x.dtype), x


WEIGHT_QUANT_REGISTRY: Dict[str, QuantFn] = {
    "wpt": quantize_weight_wpt,
    "wpc": quantize_weight_wpc,
    "wpg": quantize_weight_wpg,
    "wpg64": partial(quantize_weight_wpg, group_size=64),
    "wpg128": partial(quantize_weight_wpg, group_size=128),
    "wpg256": partial(quantize_weight_wpg, group_size=256),
    "wbf16": quantize_weight_wbf16,
    "wf16": quantize_weight_wf16,
    "none": _identity_weight,
}

ACT_QUANT_REGISTRY: Dict[str, QuantFn] = {
    "ai8pt": quantize_act_ai8pt,
    "ai8ptk": quantize_act_ai8ptk,
    "ai8pc": quantize_act_ai8pc,
    "abf16": quantize_act_abf16,
    "af16": quantize_act_af16,
    "none": _identity_act,
}


def _build_quantizer_registry() -> Dict[str, Type]:
    registry: Dict[str, Type] = {"none": NoQuantizer}
    for act_name, act_fn in ACT_QUANT_REGISTRY.items():
        for weight_name, weight_fn in WEIGHT_QUANT_REGISTRY.items():
            key = f"{act_name}_{weight_name}"
            quantizer_cls = build_quantizer_class(
                act_name, weight_name, weight_fn, act_fn
            )
            registry[key] = quantizer_cls
    return registry


QUANTIZER_REGISTRY: Dict[str, Type] = _build_quantizer_registry()


def get_quantizer(quant_type: str):
    if quant_type not in QUANTIZER_REGISTRY:
        raise ValueError(
            f"Unknown quantizer type '{quant_type}'. Available: {sorted(QUANTIZER_REGISTRY.keys())}"
        )
    return QUANTIZER_REGISTRY[quant_type]


def get_weight_quant_fn(weight_quant_type: str) -> QuantFn:
    if weight_quant_type not in WEIGHT_QUANT_REGISTRY:
        raise ValueError(
            f"Unknown weight quant type '{weight_quant_type}'. Available: {sorted(WEIGHT_QUANT_REGISTRY.keys())}"
        )
    return WEIGHT_QUANT_REGISTRY[weight_quant_type]


def get_act_quant_fn(act_quant_type: str) -> QuantFn:
    if act_quant_type not in ACT_QUANT_REGISTRY:
        raise ValueError(
            f"Unknown activation quant type '{act_quant_type}'. Available: {sorted(ACT_QUANT_REGISTRY.keys())}"
        )
    return ACT_QUANT_REGISTRY[act_quant_type]


__all__ = [
    "QUANTIZER_REGISTRY",
    "WEIGHT_QUANT_REGISTRY",
    "ACT_QUANT_REGISTRY",
    "get_quantizer",
    "get_weight_quant_fn",
    "get_act_quant_fn",
]
