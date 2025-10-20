from bitlayers import register_layer 
import torch.nn as nn
import torch
import math

from bitcore.ops import bitlinear


@register_layer("BitLinear") 
class BitLinear(nn.Module): 
    def __init__(self, in_features, out_features, bias=True, init_method='xavier_uniform'):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        self.use_bias = bias 

        # Initialize weight with proper scaling for quantized layers
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weight(init_method)
        
        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def _init_weight(self, init_method):
        """Initialize weight with appropriate method for quantized layers"""
        with torch.no_grad():
            if init_method == 'xavier_uniform':
                # Xavier/Glorot uniform initialization
                bound = math.sqrt(6.0 / (self.in_features + self.out_features))
                self.weight.uniform_(-bound, bound)
            elif init_method == 'xavier_normal':
                # Xavier/Glorot normal initialization
                std = math.sqrt(2.0 / (self.in_features + self.out_features))
                self.weight.normal_(0, std)
            elif init_method == 'kaiming_uniform':
                # He initialization (uniform)
                bound = math.sqrt(3.0) * math.sqrt(2.0 / self.in_features)
                self.weight.uniform_(-bound, bound)
            elif init_method == 'kaiming_normal':
                # He initialization (normal)
                std = math.sqrt(2.0 / self.in_features)
                self.weight.normal_(0, std)
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")

    def forward(self, input): 
        return bitlinear(input, self.weight, self.bias)
