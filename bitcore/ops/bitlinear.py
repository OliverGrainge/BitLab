# bitcore/ops/bitlinear.py
import torch
import torch.nn.functional as F
from bitcore.config import BitQuantConfig
from bitcore.ops.base import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
from bitcore.kernels import KernelRegistry

def forward(x: torch.Tensor, training: bool = True, **kwargs):
    """
    Flexible bitlinear operation that can handle different argument patterns
    
    Args:
        x: (N, *, in_features)
        **kwargs: Can include:
            - weight: (out_features, in_features) - for training
            - qweight: (out_features, in_features) - for evaluation  
            - qweight_scale: scaling factors
            - bias: (out_features) optional
            - quant_config: BitQuantConfig instance
            - training: bool (True if performing training forward pass or False if performing evaluation forward pass)
    
    Returns:
        output: (N, *, out_features)
    """
    
    if training:
        return _train_bitlinear(x, **kwargs)
    else:
        return _eval_bitlinear(x, **kwargs)

def _train_bitlinear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, quant_config: BitQuantConfig = None, **kwargs):
    """Training forward - expects 'weight' (full precision)"""
    # For training, use standard linear operation
    qweight_scale = compute_weight_scale(weight, quant_config) 
    qx_scale = compute_x_scale(x, quant_config) 
    qweight = quantize_weight(weight, qweight_scale, quant_config) 
    qx = quantize_x(x, qx_scale, quant_config) 
    dqx = dequantize_x(qx, qx_scale, quant_config) 
    dqweight = dequantize_weight(qweight, qweight_scale, quant_config) 
    return F.linear(dqx, dqweight, bias) 

def _eval_bitlinear(x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, bias: torch.Tensor = None, quant_config: BitQuantConfig = None, **kwargs):
    """Evaluation forward with kernel dispatch"""
    
    # Get the appropriate kernel for this config
    kernel = KernelRegistry.get_kernel_from_config(quant_config)
    
    # Execute the kernel's __call__ method
    return kernel(x, qweight_scale, qweight, bias, quant_config)

def prepare_weights(weight: torch.Tensor, quant_config: BitQuantConfig): 
    """Prepare weights using the appropriate kernel"""
    
    # Get the appropriate kernel for this config
    kernel = KernelRegistry.get_kernel_from_config(quant_config)
    
    # Execute the kernel's prepare_weights method
    return kernel.prepare_weights(weight, quant_config)