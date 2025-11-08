"""Unit tests for activation quantization functions."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestLinearActivationQuantizationOutput:
    """Unit tests for activation quantization output format."""

    def test_output_shape(self, linear_act_quantizer, sample_linear_activations):
        act_quant_name, act_quant_fn = linear_act_quantizer
        x = sample_linear_activations

        scale, quantized = act_quant_fn(x)

        assert isinstance(scale, torch.Tensor), f"Scale should be torch.Tensor for {act_quant_name}"
        assert quantized.shape == x.shape, f"Quantized should have same shape as input for {act_quant_name}"

    def test_output_type(self, linear_act_quantizer, sample_linear_activations):
        act_quant_name, act_quant_fn = linear_act_quantizer
        x = sample_linear_activations

        scale, quantized = act_quant_fn(x)

        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)


@pytest.mark.unit
class TestConvActivationQuantizationOutput:
    """Unit tests for conv2d activation quantization (4D tensors) output format."""

    def test_output_shape(self, conv_act_quantizer, sample_conv_activations):
        act_quant_name, act_quant_fn = conv_act_quantizer
        x = sample_conv_activations

        scale, quantized = act_quant_fn(x)

        assert isinstance(scale, torch.Tensor), f"Scale should be torch.Tensor for {act_quant_name}"
        assert quantized.shape == x.shape, f"Quantized should have same shape as input for {act_quant_name}"

    def test_output_type(self, conv_act_quantizer, sample_conv_activations):
        act_quant_name, act_quant_fn = conv_act_quantizer
        x = sample_conv_activations

        scale, quantized = act_quant_fn(x)

        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)
