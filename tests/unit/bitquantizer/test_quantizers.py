"""Unit tests for Quantizer classes."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestQuantizerForward:
    """Unit tests for quantizer forward pass on linear (2D) tensors."""

    def test_forward_output_shape(
        self, linear_quantizer, sample_activations, sample_weights
    ):
        quantizer_name, quantizer_class = linear_quantizer

        dqx, dqw = quantizer_class.apply(sample_activations, sample_weights)

        assert dqx.shape == sample_activations.shape
        assert dqw.shape == sample_weights.shape

    def test_forward_output_type(
        self, linear_quantizer, sample_activations, sample_weights
    ):
        quantizer_name, quantizer_class = linear_quantizer

        dqx, dqw = quantizer_class.apply(sample_activations, sample_weights)

        assert isinstance(dqx, torch.Tensor)
        assert isinstance(dqw, torch.Tensor)


@pytest.mark.unit
class TestQuantizerOutput:
    """Output validation for linear quantizer invocations."""

    def test_output_is_tensor(
        self, linear_quantizer, sample_activations, sample_weights
    ):
        quantizer_name, quantizer_class = linear_quantizer

        dqx, dqw = quantizer_class.apply(sample_activations, sample_weights)

        assert isinstance(dqx, torch.Tensor)
        assert isinstance(dqw, torch.Tensor)

    def test_output_preserves_shape(
        self, linear_quantizer, sample_activations, sample_weights
    ):
        quantizer_name, quantizer_class = linear_quantizer

        dqx, dqw = quantizer_class.apply(sample_activations, sample_weights)

        assert dqx.shape == sample_activations.shape
        assert dqw.shape == sample_weights.shape


@pytest.mark.unit
class TestConvQuantizerForward:
    """Unit tests for quantizer forward pass with conv2d (4D) tensors."""

    def test_forward_output_shape(
        self, conv_quantizer, sample_conv_activations, sample_conv_weights
    ):
        quantizer_name, quantizer_class = conv_quantizer

        dqx, dqw = quantizer_class.apply(sample_conv_activations, sample_conv_weights)

        assert dqx.shape == sample_conv_activations.shape
        assert dqw.shape == sample_conv_weights.shape

    def test_forward_output_type(
        self, conv_quantizer, sample_conv_activations, sample_conv_weights
    ):
        quantizer_name, quantizer_class = conv_quantizer

        dqx, dqw = quantizer_class.apply(sample_conv_activations, sample_conv_weights)

        assert isinstance(dqx, torch.Tensor)
        assert isinstance(dqw, torch.Tensor)


@pytest.mark.unit
class TestConvQuantizerOutput:
    """Output validation for conv quantizer invocations."""

    def test_output_is_tensor(
        self, conv_quantizer, sample_conv_activations, sample_conv_weights
    ):
        quantizer_name, quantizer_class = conv_quantizer

        dqx, dqw = quantizer_class.apply(sample_conv_activations, sample_conv_weights)

        assert isinstance(dqx, torch.Tensor)
        assert isinstance(dqw, torch.Tensor)

    def test_output_preserves_shape(
        self, conv_quantizer, sample_conv_activations, sample_conv_weights
    ):
        quantizer_name, quantizer_class = conv_quantizer

        dqx, dqw = quantizer_class.apply(sample_conv_activations, sample_conv_weights)

        assert dqx.shape == sample_conv_activations.shape
        assert dqw.shape == sample_conv_weights.shape
