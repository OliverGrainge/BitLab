# bitcore/ops/bitlinear.py
import torch
import torch.nn.functional as F
from bitcore.config import BitQuantConfig
from bitcore.ops.utils import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
from bitcore.kernels import KernelRegistry
from typing import Tuple



def train_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
    """Training forward pass using the appropriate kernel: 
    
    Args:
        x: The input tensor
        weight: The weight tensor
        bias: The bias tensor
        quant_config: The quantization configuration
    """
    qweight_scale = compute_weight_scale(weight, quant_config) 
    qx_scale = compute_x_scale(x, quant_config) 
    qweight = quantize_weight(weight, qweight_scale, quant_config) 
    qx = quantize_x(x, qx_scale, quant_config) 
    dqx = dequantize_x(qx, qx_scale, quant_config) 
    dqweight = dequantize_weight(qweight, qweight_scale, quant_config) 
    return F.linear(dqx, dqweight, bias) 

def eval_forward(x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
    """Forward pass using the appropriate kernel: 
    
    Args:
        x: The input tensor
        qweight_scale: The quantized weight scale
        qweight: The quantized weights
        bias: The bias tensor
        quant_config: The quantization configuration
    """
    kernel = KernelRegistry.get_kernel_from_config(quant_config)
    return kernel.forward(x, qweight_scale, qweight, bias)

def quantize_weights(weight: torch.Tensor, quant_config: BitQuantConfig) -> Tuple[torch.Tensor, torch.Tensor]: 
    """Quantize weights using the appropriate kernel: 
    
    Args:
        weight: The weight tensor to quantize
        quant_config: The quantization configuration
    
    Returns:
        A tuple of (qweight_scale, qweight) - the quantized weight scale and quantized weights
    """
    kernel = KernelRegistry.get_kernel_from_config(quant_config)
    return kernel.quantize_weights(weight)