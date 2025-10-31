# /Users/olivergrainge/github/BitLab/src/bitlab/bnn/functional/bitlinear.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from bitlab.bitquantizer import quantize_weight, quantize_activation


class _BitLinearFunctional:
    """Namespace + callable that mirrors the deployment API used by layers."""

    def prepare_weights(
        self,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return quantized weights plus scale that the layer can stash for deploy."""
        qws, qw = quantize_weight(weight, eps)
        return qws, qw

    def __call__(
        self,
        x: torch.Tensor,
        qws: torch.Tensor,
        qw: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        dqweight = qws * qw
        qxs, qx = quantize_activation(x, eps)
        dqx = qxs * qx
        return F.linear(dqx, dqweight, bias)


bitlinear = _BitLinearFunctional()
