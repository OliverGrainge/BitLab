from __future__ import annotations

from functools import partial
from typing import Iterable, Optional

import torch
import torch.nn as nn

from bitlab.bnn import Module
from bitlab.bnn.bitlayers import BitLinear
from bitlab.bitmodels.auto import register_bitmodel
from bitlab.bitmodels.mlp.config import BitMLPConfig


class BitMLP(Module):
    """Multi-layer perceptron supporting quantized BitLinear hidden layers."""

    def __init__(
        self,
        input_size: int,
        hidden_dims: Iterable[int],
        num_classes: int,
        use_bitlinear: bool,
        quant_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            hidden_dims = [256]

        if use_bitlinear:
            if quant_type is None:
                raise ValueError("quant_type must be specified when use_bitlinear=True")
            linear_factory = partial(BitLinear, quant_type=quant_type)
        else:
            linear_factory = nn.Linear

        layers: list[nn.Module] = []
        prev_dim = input_size

        for hidden_dim in hidden_dims:
            layers.append(linear_factory(prev_dim, hidden_dim, bias=True))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


@register_bitmodel("bitmlp")
class BitMLPModel(Module):
    """BitMLPModel wraps the MLP architecture with registry + config support."""

    def __init__(self, config: Optional[BitMLPConfig] = None, **overrides) -> None:
        super().__init__()

        if config is None:
            config = BitMLPConfig(**overrides)
        else:
            if not isinstance(config, BitMLPConfig):
                raise TypeError("config must be a BitMLPConfig instance or None")
            if overrides:
                config = config.with_overrides(**overrides)

        if config.use_bitlinear and config.quant_type is None:
            raise ValueError("quant_type must be provided when use_bitlinear=True")

        self.config = config

        self.model = BitMLP(
            input_size=config.input_size,
            hidden_dims=config.hidden_dims,
            num_classes=config.num_classes,
            use_bitlinear=config.use_bitlinear,
            quant_type=config.quant_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

