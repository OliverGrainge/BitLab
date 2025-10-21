from bitlayers import register_layer 
import torch.nn as nn
import torch
import math

from bitcore.ops import bitlinear
from bitlayers.base import BitLayerBase
from bitcore.config import BitQuantConfig


@register_layer("BitLinear") 
class BitLinear(BitLayerBase): 
    def __init__(self, in_features, out_features, bias=True, init_method='xavier_uniform', quant_config: BitQuantConfig = None):
        super().__init__(quant_config)
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
        
        # Initialize quantization parameters based on config
        self._init_quantization_params()

    def _init_weight(self, init_method):
        """Initialize weight with appropriate method for quantized layers"""
        with torch.no_grad():
            if init_method == 'kaiming_uniform':
                # He initialization (uniform)
                bound = math.sqrt(3.0) * math.sqrt(2.0 / self.in_features)
                self.weight.uniform_(-bound, bound)
            elif init_method == 'xavier_uniform':
                # Xavier/Glorot initialization (uniform)
                bound = math.sqrt(6.0 / (self.in_features + self.out_features))
                self.weight.uniform_(-bound, bound)
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")

    def _init_quantization_params(self):
        """Initialize quantization parameters based on quant_config"""
        # Weight quantization parameters
        if self.quant_config.weight_granularity == "per_tensor":
            self.weight_scale = nn.Parameter(torch.ones(1))
        elif self.quant_config.weight_granularity == "per_channel":
            self.weight_scale = nn.Parameter(torch.ones(self.out_features))
        
        # Activation quantization parameters
        if self.quant_config.activation_granularity == "per_tensor":
            self.activation_scale = nn.Parameter(torch.ones(1))
        elif self.quant_config.activation_granularity == "per_channel":
            self.activation_scale = nn.Parameter(torch.ones(self.in_features))

    def _train_forward(self, input): 
        return bitlinear(input, self.weight, self.bias, self.quant_config) 

    def _eval_forward(self, input): 
        return bitlinear(input, self.weight, self.bias, self.quant_config)

    def _on_enter_training_mode(self):
        """Called when entering training mode - override in subclasses"""
        pass

    def _on_enter_eval_mode(self):
        """Called when entering evaluation mode - override in subclasses"""
        pass