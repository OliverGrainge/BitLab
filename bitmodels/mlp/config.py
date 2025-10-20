from dataclasses import asdict, dataclass
from typing import List

from bitmodels import register_config


@register_config("BitMLPConfig")
@dataclass
class BitMLPConfig:
    n_layers: int
    in_channels: int
    hidden_dim: int
    out_channels: int
    dropout: float = 0.0
