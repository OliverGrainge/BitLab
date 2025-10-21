from dataclasses import asdict, dataclass
from typing import List

from bitmodels import register_config


@register_config("BitMLPConfig")
@dataclass
class BitMLPConfig:
    n_layers: int = 3
    in_channels: int = 256
    hidden_dim: int = 128
    out_channels: int = 10
    dropout: float = 0.0
