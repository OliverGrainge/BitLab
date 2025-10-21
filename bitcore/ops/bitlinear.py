# bitcore/ops/bitlinear.py
import torch
import torch.nn.functional as F
from bitcore.config import BitQuantConfig
from bitcore.ops.utils import quantize_weight, quantize_x, compute_weight_scale, compute_x_scale, dequantize_x, dequantize_weight
from bitcore.kernels import KernelRegistry


def train_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
    qweight_scale = compute_weight_scale(weight, quant_config) 
    qx_scale = compute_x_scale(x, quant_config) 
    qweight = quantize_weight(weight, qweight_scale, quant_config) 
    qx = quantize_x(x, qx_scale, quant_config) 
    dqx = dequantize_x(qx, qx_scale, quant_config) 
    dqweight = dequantize_weight(qweight, qweight_scale, quant_config) 
    return F.linear(dqx, dqweight, bias) 

def eval_forward(x: torch.Tensor, qweight_scale: torch.Tensor, qweight: torch.Tensor, bias: torch.Tensor = None, quant_config: BitQuantConfig = None):
    kernel = KernelRegistry.get_kernel_from_config(quant_config)
    return kernel(x, qweight_scale, qweight, bias, quant_config)

def quantize_weights(weight: torch.Tensor, quant_config: BitQuantConfig): 
    kernel = KernelRegistry.get_kernel_from_config(quant_config)
    return kernel.quantize_weights(weight, quant_config)