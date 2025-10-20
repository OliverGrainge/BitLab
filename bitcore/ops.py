import torch
import torch.nn.functional as F

def bitlinear(input, weight, bias=None):
    """
    Ternary linear operation - fallback to torch.nn.functional.linear for now
    
    Args:
        input: (N, *, in_features)
        weight: (out_features, in_features)
        bias: (out_features) optional
    
    Returns:
        output: (N, *, out_features)
    """
    # TODO: Add ternary quantization
    return F.linear(input, weight, bias)


def bitconv2d(input, weight, bias=None, stride=1, padding=0):
    """Ternary 2D convolution - fallback to torch for now"""
    # TODO: Add ternary quantization
    return F.conv2d(input, weight, bias, stride, padding)


def bitmatmul(input, other):
    """Ternary matrix multiplication"""
    # TODO: Add ternary quantization
    return torch.matmul(input, other)