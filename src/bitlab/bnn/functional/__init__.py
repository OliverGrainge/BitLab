"""Deployment-friendly functional interfaces for quantized layers."""

from .bitlinear import bitlinear
from .bitconv2d import bitconv2d

__all__ = ["bitlinear", "bitconv2d"]
