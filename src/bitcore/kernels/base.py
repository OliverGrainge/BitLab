# bitcore/kernels/base.py
import torch
from abc import ABC, abstractmethod
from typing import Tuple
from bitcore.config import BitQuantConfig
from bitcore.ops.utils import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
import torch.nn.functional as F

class BitKernelBase(ABC):
    """Base class for all bitlinear kernels"""
    
    def __init__(self, quant_config: BitQuantConfig):
        self.quant_config = quant_config
    
    @classmethod
    @abstractmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Check if this kernel is suitable for the given config"""
        pass
    
    @abstractmethod
    def quantize_weights(self, weight: torch.Tensor, quant_config: BitQuantConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Offline weight quantization - prepares weights for this kernel's requirements
        
        Returns:
            Tuple of (qweight_scale, qweight) - the quantized weight scale and quantized weights
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, qweight_scale: torch.Tensor, 
                qweight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """Online forward pass using pre-quantized weights
        
        Args:
            x: Input activations
            qweight_scale: Pre-computed weight scale from quantize_weights
            qweight: Pre-quantized weights from quantize_weights  
            bias: Optional bias tensor
        """
        pass


class ReferenceKernel(BitKernelBase):
    """Reference implementation kernel (fallback)"""
    
    def __init__(self, quant_config: BitQuantConfig):
        super().__init__(quant_config)
        self.quantize_weight = quantize_weight
        self.quantize_x = quantize_x
        self.compute_weight_scale = compute_weight_scale
        self.compute_x_scale = compute_x_scale
        self.dequantize_x = dequantize_x
        self.dequantize_weight = dequantize_weight
        self.F = F
    
    @classmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Reference kernel is suitable for all configs"""
        return True
    
    def quantize_weights(self, weight: torch.Tensor, quant_config: BitQuantConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference weight preparation"""
        qweight_scale = self.compute_weight_scale(weight, quant_config)
        qweight = self.quantize_weight(weight, qweight_scale, quant_config)
        return qweight_scale, qweight
    
    def forward(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None) -> torch.Tensor:
        """Reference forward pass"""
        qx_scale = self.compute_x_scale(x, self.quant_config)
        qx = self.quantize_x(x, qx_scale, self.quant_config)
        dqx = self.dequantize_x(qx, qx_scale, self.quant_config)
        dqweight = self.dequantize_weight(qweight, qweight_scale, self.quant_config)
        return self.F.linear(dqx, dqweight, bias)