import quopri
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


    def _train_forward(self, x: torch.Tensor): 
        """Training forward pass - uses full precision weights"""
        if self.is_deployed:
            # If deployed, we no longer have original weights, use eval forward
            raise RuntimeError("Deployed layers cannot enter training mode")
        return bitlinear.forward(x=x, weight=self.weight, bias=self.bias, quant_config=self.quant_config, training=True)

    def _eval_forward(self, x: torch.Tensor): 
        """Evaluation forward pass - uses quantized weights"""
        return bitlinear.forward(x=x, qweight_scale=self.qweight_scale, qweight=self.qweight, bias=self.bias, quant_config=self.quant_config, training=False)

    def _on_enter_training_mode(self):
        """Called when entering training mode - remove quantized weights and scales"""
        if self.is_deployed:
            # Deployed layers can't enter training mode
            raise RuntimeError("Deployed layers cannot enter training mode")
        
        if 'qweight' in self._buffers:
            delattr(self, 'qweight')
        if 'qweight_scale' in self._buffers:
            delattr(self, 'qweight_scale')
        if 'qx_scale' in self._buffers:
            delattr(self, 'qx_scale')

    def _on_enter_eval_mode(self):
        """Called when entering evaluation mode - create quantized weights and scales"""
        if self.is_deployed:
            # Deployed layers already have quantized weights, no need to recreate
            return 
        
        qweight_scale, qweight = bitlinear.prepare_weights(self.weight, self.quant_config)
        
        # Store quantization parameters as buffers since they're computed values, not learnable parameters
        self.register_buffer('qweight_scale', qweight_scale)
        self.register_buffer('qweight', qweight)

    def _perform_deployment(self): 
        """Perform deployment by permanently quantizing the layer and removing latent weights"""
        # Ensure we're in eval mode with quantized weights
        if not hasattr(self, 'qweight_scale'):
            self._on_enter_eval_mode()
        
        # Keep only quantized weights and scales, remove original weights
        self.qweight_scale = self.qweight_scale.detach()
        self.qweight = self.qweight.detach()
        
        # Keep bias as a regular tensor (remove gradients)
        if self.bias is not None:
            # Convert bias parameter to buffer to remove gradients
            bias_tensor = self.bias.detach()
            # Remove the parameter first, then register as buffer
            delattr(self, 'bias')
            self.register_buffer('bias', bias_tensor)
        
        # Remove original full-precision weights - they're no longer needed
        delattr(self, 'weight')