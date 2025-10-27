# bitcore/kernels/base.py
import torch
from abc import ABC, abstractmethod
from typing import Tuple
from bitcore.config import BitQuantConfig
from bitcore.ops.utils import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
import torch.nn.functional as F
from typing import Optional


class BitKernelBase(ABC):
    """Base class for all bitlinear kernels"""
    
    def __init__(self):
        pass
    
    @classmethod
    @abstractmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Check if this kernel is suitable for the given config"""
        pass
    
    @abstractmethod
    def quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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




class OptimizedKernelBase(BitKernelBase):
    """
    Base class for kernels with C++ optimization and automatic PyTorch fallback.
    
    Subclasses only need to:
    1. Set cpp_module in __init__
    2. Set quant_config in __init__
    3. Implement is_suitable_for() classmethod
    """
    
    def __init__(self, cpp_module: Optional[object], quant_config: BitQuantConfig):
        """
        Args:
            cpp_module: The C++ module with quantize_weights() and forward() methods, or None
            quant_config: The quantization configuration for this kernel
        """
        super().__init__()
        self.has_cpp_backend = cpp_module is not None
        self.cpp_module = cpp_module
        self.quant_config = quant_config
    
    @classmethod
    @abstractmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Check if this kernel is suitable for the given config"""
        pass
    
    def quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize weights using C++ if available, otherwise PyTorch fallback"""
        if self.has_cpp_backend:
            return self.cpp_module.quantize_weights(weight)
        else:
            return self._pytorch_quantize_weights_fallback(weight)
    
    def forward(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None) -> torch.Tensor:
        """Forward pass using C++ if available, otherwise PyTorch fallback"""
        if self.has_cpp_backend:
            return self.cpp_module.forward(x, qweight_scale, qweight, bias)
        else:
            return self._pytorch_forward_fallback(x, qweight_scale, qweight, bias)
    
    def _pytorch_quantize_weights_fallback(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default PyTorch implementation of weight quantization"""
        qweight_scale = compute_weight_scale(weight, self.quant_config)
        qweight = quantize_weight(weight, qweight_scale, self.quant_config)
        return qweight_scale, qweight
    
    def _pytorch_forward_fallback(self, x: torch.Tensor, qweight_scale: torch.Tensor, 
                                   qweight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Default PyTorch implementation of forward pass"""
        qx_scale = compute_x_scale(x, self.quant_config)
        qx = quantize_x(x, qx_scale, self.quant_config)
        
        # Dequantize and compute
        dqx = dequantize_x(qx, qx_scale, self.quant_config)
        dqweight = dequantize_weight(qweight, qweight_scale, self.quant_config)
        return F.linear(dqx, dqweight, bias)



class ReferenceKernel(BitKernelBase):
    """Reference implementation kernel (fallback)"""
    
    def __init__(self, quant_config: BitQuantConfig):
        super().__init__()
        self.quantize_weight = quantize_weight
        self.quantize_x = quantize_x
        self.compute_weight_scale = compute_weight_scale
        self.compute_x_scale = compute_x_scale
        self.dequantize_x = dequantize_x
        self.dequantize_weight = dequantize_weight
        self.F = F
        self.quant_config = quant_config
    
    @classmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Reference kernel is suitable for all configs"""
        return True
    
    def quantize_weights(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference weight preparation"""
        qweight_scale = self.compute_weight_scale(weight, self.quant_config)
        qweight = self.quantize_weight(weight, qweight_scale, self.quant_config)
        return qweight_scale, qweight
    
    def forward(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None) -> torch.Tensor:
        """Reference forward pass"""
        qx_scale = self.compute_x_scale(x, self.quant_config)
        qx = self.quantize_x(x, qx_scale, self.quant_config)
        dqx = self.dequantize_x(qx, qx_scale, self.quant_config)
        dqweight = self.dequantize_weight(qweight, qweight_scale, self.quant_config)
        return self.F.linear(dqx, dqweight, bias)