import torch
import torch.nn.functional as F
from .config import BitQuantConfig

def bitlinear(input, weight, bias=None, quant_config: BitQuantConfig = None):
    """
    Ternary linear operation - fallback to torch.nn.functional.linear for now

    Args:
        input: (N, *, in_features)
        weight: (out_features, in_features)
        bias: (out_features) optional
        quant_config: BitQuantConfig instance, contains quantization settings

    Returns:
        output: (N, *, out_features)
    """
    # TODO: Add ternary quantization and use quant_config in custom kernel
    return F.linear(input, weight, bias)


def bitconv2d(input, weight, bias=None, stride=1, padding=0, quant_config: BitQuantConfig = None):
    """
    Ternary 2D convolution - fallback to torch for now

    Args:
        input: (N, C_in, H, W)
        weight: (C_out, C_in, kH, kW)
        bias: (C_out) optional
        stride: int or tuple
        padding: int or tuple
        quant_config: BitQuantConfig instance, contains quantization settings

    Returns:
        output: (N, C_out, H_out, W_out)
    """
    # TODO: Add ternary quantization and use quant_config in custom kernel
    return F.conv2d(input, weight, bias, stride, padding)


def bitmatmul(input, other, quant_config: BitQuantConfig = None):
    """
    Ternary matrix multiplication

    Args:
        input: (..., m, n)
        other: (..., n, p)
        quant_config: BitQuantConfig instance, contains quantization settings

    Returns:
        output: (..., m, p)
    """
    # TODO: Add ternary quantization and use quant_config in custom kernel
    return torch.matmul(input, other)