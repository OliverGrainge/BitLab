# bitcore/kernels/BitLinear_Ai8pt_Wpt.py
import torch
import torch.nn.functional as F
from bitcore.kernels.base import BitKernelBase
from bitcore.config import BitQuantConfig
from bitcore.ops.base import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
from bitcore.kernels.registry import KernelRegistry


@KernelRegistry.register(
    lambda config: (
        config.activation_dtype == "int8" and
        config.activation_granularity == "per_tensor" and
        config.weight_granularity == "per_tensor"
    )
)
class bitlinear_kernel_i8_pt_pt(BitKernelBase):
    """Optimized kernel for INT8 per-tensor quantization"""
    
    def prepare_weights(self, weight: torch.Tensor, quant_config: BitQuantConfig):
        """Prepare weights for INT8 per-tensor kernel"""
        qweight_scale = compute_weight_scale(weight, quant_config)
        qweight = quantize_weight(weight, qweight_scale, quant_config)
        return qweight_scale, qweight
    
    def __call__(self, x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, 
                bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
        """Optimized forward pass for INT8 per-tensor"""
        qx_scale = compute_x_scale(x, quant_config)
        qx = quantize_x(x, qx_scale, quant_config)
        
        # TODO: Replace with optimized kernel implementation
        dqx = dequantize_x(qx, qx_scale, quant_config)
        dqweight = dequantize_weight(qweight, qweight_scale, quant_config)
        return F.linear(dqx, dqweight, bias)
    
    def is_compatible_with(self, quant_config: BitQuantConfig) -> bool:
        """Check if compatible with INT8 per-tensor config"""
        return (
            quant_config.activation_dtype == "int8" and
            quant_config.activation_granularity == "per_tensor" and
            quant_config.weight_granularity == "per_tensor"
        )