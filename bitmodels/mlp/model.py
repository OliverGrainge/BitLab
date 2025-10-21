import torch.nn as nn

from bitmodels import register_model
from bitmodels.base import BitModelBase

from .config import BitMLPConfig
from bitlayers import LAYER_REGISTRY



@register_model("BitMLPModel")
class BitMLPModel(BitModelBase):
    def __init__(self, config: BitMLPConfig, quant_config=None):
        super().__init__(config, quant_config)

        # Get BitLinear from registry
        BitLinear = LAYER_REGISTRY['BitLinear']

        # Build layers
        layers = []
        in_dim = config.in_channels
        for _ in range(config.n_layers - 1):
            layers.append(BitLinear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim

        layers.append(BitLinear(in_dim, config.out_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
