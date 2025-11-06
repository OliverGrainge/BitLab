from .quantizer import Quantizer_ai8pc_wpt, Quantizer_ai8pg_wpt
from .quant_fn import (
    quantize_weight_wpt,
    quantize_activation_ai8pc,
    quantize_activation_ai8pg,
)
import torch 
from typing import Tuple

# Alias for backward compatibility
quantize_weight_ai8pc_wpt = quantize_weight_wpt 

__all__ = ["BitQuantizer", "Quantize", "quantize_weight", "quantize_activation"]


QUANTIZER_REGISTRY = {
    "ai8pc_wpt": Quantizer_ai8pc_wpt,
    "ai8pg_wpt": Quantizer_ai8pg_wpt,
}

QUANT_WEIGHT_FN_REGISTRY = {
    "ai8pc_wpt": quantize_weight_wpt,
    "ai8pg_wpt": quantize_weight_wpt,  # ai8pg uses same weight quantization
}

QUANT_ACT_FN_REGSITRY = {
    "ai8pc": quantize_activation_ai8pc,
    "ai8pg": quantize_activation_ai8pg,
}


class BitQuantizer:
    def __init__(self, eps: float = 1e-6, quant_type: str = "ai8pc_wpt"):
        self.eps = eps
        self.quant_fn = QUANTIZER_REGISTRY[quant_type]

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.quant_fn.apply(x, w, self.eps)


def quantize_weight(w: torch.Tensor, eps: float=1e-6, quant_type: str="ai8pc_wpt"): 
    return QUANT_WEIGHT_FN_REGISTRY[quant_type](w, eps)


def quantize_activation(x: torch.Tensor, eps: float=1e-6, quant_type: str="ai8pc_wpt"): 
    # Extract activation type from quant_type (remove _wpt suffix if present)
    act_type = quant_type.replace("_wpt", "").replace("-wpt", "")
    return QUANT_ACT_FN_REGSITRY[act_type](x, eps)
    