# bitcore/kernels/base.py
import torch
from abc import ABC, abstractmethod
from bitcore.config import BitQuantConfig
from bitcore.ops.utils import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
import torch.nn.functional as F

class BitKernelBase(ABC):
    """Base class for all bitlinear kernels"""
    
    def __init__(self, quant_config: BitQuantConfig):
        self.quant_config = quant_config
    
    @abstractmethod
    def quantize_weights(self, weight: torch.Tensor, quant_config: BitQuantConfig):
        """Prepare weights for this kernel's specific requirements"""
        pass
    
    @abstractmethod
    def __call__(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
        """Execute the kernel-specific forward pass"""
        pass
    
    @abstractmethod
    def is_compatible_with(self, quant_config: BitQuantConfig) -> bool:
        """Check if this kernel is compatible with the given config"""
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
    
    def quantize_weights(self, weight: torch.Tensor, quant_config: BitQuantConfig):
        """Reference weight preparation"""
        qweight_scale = self.compute_weight_scale(weight, quant_config)
        qweight = self.quantize_weight(weight, qweight_scale, quant_config)
        return qweight_scale, qweight
    
    def __call__(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
        """Reference forward pass"""
        qx_scale = self.compute_x_scale(x, quant_config)
        qx = self.quantize_x(x, qx_scale, quant_config)
        dqx = self.dequantize_x(qx, qx_scale, quant_config)
        dqweight = self.dequantize_weight(qweight, qweight_scale, quant_config)
        return self.F.linear(dqx, dqweight, bias)
    
    def is_compatible_with(self, quant_config: BitQuantConfig) -> bool:
        """Reference kernel is compatible with all configs"""
        return True