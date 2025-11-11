"""Deployment-friendly functional interfaces for quantized layers."""

from .bitconv2d import bitconv2d
from .bitlinear import bitlinear

__all__ = ["bitlinear", "bitconv2d"]
