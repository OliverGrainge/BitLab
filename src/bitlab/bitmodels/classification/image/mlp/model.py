from __future__ import annotations

from functools import partial
from typing import Any, ClassVar, Iterable, Optional

import torch
import torch.nn as nn

import bitlab.bnn as bnn
from bitlab.bitmodels.auto import register_bitmodel
from bitlab.bitmodels.base import BaseBitModel
from bitlab.bitmodels.classification.image.mlp.config import BitMLPConfig
from bitlab.bnn.bitlayers import BitLinear


class BitMLP(nn.Module):
    """Multi-layer perceptron supporting quantized BitLinear hidden layers."""

    def __init__(
        self,
        input_size: int,
        hidden_dims: Iterable[int],
        num_classes: int,
        quant_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        self.quant_type = quant_type
        if not hidden_dims:
            hidden_dims = [256]

        layers: list[nn.Module] = []
        prev_dim = input_size

        linear_factory = partial(BitLinear, quant_type=quant_type)

        for hidden_dim in hidden_dims:
            layers.append(linear_factory(prev_dim, hidden_dim, bias=True))
            layers.append(nn.RMSNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


@register_bitmodel("bitmlp")
class BitMLPModel(BaseBitModel):
    """BitMLPModel wraps the MLP architecture with registry + config support."""

    config_cls: ClassVar[type[BitMLPConfig]] = BitMLPConfig

    def __init__(self, config: Optional[BitMLPConfig] = None, **overrides: Any) -> None:
        super().__init__(config=config, **overrides)

    def build_model(self, config: BitMLPConfig) -> nn.Module:
        return BitMLP(
            input_size=config.input_size,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            quant_type=config.quant_type,
        )
