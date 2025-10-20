from bitmodels import register_model
from .config import BitMLPConfig
import torch.nn as nn

@register_model("BitMLPModel")
class BitMLPModel(nn.Module):
    def __init__(self, config: BitMLPConfig, quant_config=None):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        
        # Build layers
        layers = []
        in_dim = config.in_channels
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        
        layers.append(nn.Linear(in_dim, config.out_channels))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)