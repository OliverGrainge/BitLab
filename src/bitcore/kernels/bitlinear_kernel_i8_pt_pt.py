# bitcore/kernels/bitlinear_kernel_i8_pt_pt.py
import torch
import torch.nn.functional as F
from typing import Tuple
from bitcore.kernels.base import BitKernelBase
from bitcore.config import BitQuantConfig
from bitcore.ops.utils import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
from bitcore.kernels.registry import KernelRegistry

# Import the optimized C++ bindings directly
try:
    from bitcore.kernels.bindings import bitlinear_int8_pt_pt_cpp
    HAS_OPTIMIZED_BACKEND = True
except ImportError:
    HAS_OPTIMIZED_BACKEND = False


@KernelRegistry.register_kernel(
    name="int8_per_tensor",
    priority=100,
    activation_dtype="int8",
    activation_granularity="per_tensor",
    weight_granularity="per_tensor"
)
class bitlinear_kernel_i8_pt_pt(BitKernelBase):
    """Optimized kernel for INT8 per-tensor quantization with C++ bindings"""
    
    def __init__(self, quant_config: BitQuantConfig):
        super().__init__(quant_config)
        self.has_cpp_backend = HAS_OPTIMIZED_BACKEND
    
    @classmethod
    def is_suitable_for(cls, config: BitQuantConfig) -> bool:
        """Check if compatible with INT8 per-tensor config"""
        return (
            config.activation_dtype == "int8" and
            config.activation_granularity == "per_tensor" and
            config.weight_granularity == "per_tensor"
        )
    
    def quantize_weights(self, weight: torch.Tensor, quant_config: BitQuantConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare weights using optimized C++ implementation"""
        if self.has_cpp_backend:
            # Use optimized C++ implementation
            return bitlinear_int8_pt_pt_cpp.quantize_weights(weight)
        else:
            # Fallback to pure PyTorch
            return self._pytorch_quantize_weights_fallback(weight, quant_config)
    
    def forward(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None) -> torch.Tensor:
        """Forward pass using optimized C++ implementation"""
        if self.has_cpp_backend:
            # Use optimized C++ implementation
            return bitlinear_int8_pt_pt_cpp.forward(x, qweight_scale, qweight, bias)
        else:
            # Fallback to pure PyTorch
            return self._pytorch_forward_fallback(x, qweight_scale, qweight, bias, self.quant_config)
    
    def _pytorch_quantize_weights_fallback(self, weight: torch.Tensor, quant_config: BitQuantConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback quantize_weights using pure PyTorch"""
        qweight_scale = compute_weight_scale(weight, quant_config)
        qweight = quantize_weight(weight, qweight_scale, quant_config)
        return qweight_scale, qweight
    
    def _pytorch_forward_fallback(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                                bias: torch.Tensor, quant_config: BitQuantConfig) -> torch.Tensor:
        """Fallback forward using pure PyTorch"""
        qx_scale = compute_x_scale(x, quant_config)
        qx = quantize_x(x, qx_scale, quant_config)
        
        # TODO: Replace with optimized kernel implementation
        dqx = dequantize_x(qx, qx_scale, quant_config)
        dqweight = dequantize_weight(qweight, qweight_scale, quant_config)
        return F.linear(dqx, dqweight, bias)