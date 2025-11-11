"""Unit tests for weight quantization functions."""

from __future__ import annotations

import pytest
import torch

from .conftest import expected_linear_scale_shape


@pytest.mark.unit
class TestWeightQuantizationOutput:
    """Unit tests for weight quantization output format."""

    def test_output_shape(self, linear_weight_quantizer, sample_linear_weights):
        weight_quant_name, weight_quant_fn = linear_weight_quantizer
        w = sample_linear_weights

        scale, quantized = weight_quant_fn(w)

        expected_shape = expected_linear_scale_shape(weight_quant_name, w)
        assert scale.shape == expected_shape, (
            f"Scale shape mismatch for linear weights with {weight_quant_name}: "
            f"expected {expected_shape}, got {tuple(scale.shape)}"
        )
        assert quantized.shape == w.shape

    def test_output_type(self, linear_weight_quantizer, sample_linear_weights):
        weight_quant_name, weight_quant_fn = linear_weight_quantizer
        w = sample_linear_weights

        scale, quantized = weight_quant_fn(w)

        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)


@pytest.mark.unit
class TestWeightQuantizationRange:
    """Unit tests for weight quantization value ranges."""

    def test_quantized_values_in_range(
        self, linear_range_weight_quantizer, sample_linear_weights
    ):
        weight_quant_name, weight_quant_fn = linear_range_weight_quantizer
        w = sample_linear_weights

        _, quantized = weight_quant_fn(w)

        assert (
            quantized >= -1
        ).all(), f"Quantized values should be >= -1 for {weight_quant_name}"
        assert (
            quantized <= 1
        ).all(), f"Quantized values should be <= 1 for {weight_quant_name}"


@pytest.mark.unit
class TestConvWeightQuantizationOutput:
    """Unit tests for conv2d weight quantization (4D tensors) output format."""

    def test_output_shape(self, conv_weight_quantizer, sample_conv_weights):
        weight_quant_name, weight_quant_fn = conv_weight_quantizer
        w = sample_conv_weights

        scale, quantized = weight_quant_fn(w)

        assert isinstance(scale, torch.Tensor)
        assert quantized.shape == w.shape

    def test_output_type(self, conv_weight_quantizer, sample_conv_weights):
        weight_quant_name, weight_quant_fn = conv_weight_quantizer
        w = sample_conv_weights

        scale, quantized = weight_quant_fn(w)

        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)
