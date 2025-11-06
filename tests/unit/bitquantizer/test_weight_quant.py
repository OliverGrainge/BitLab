"""Unit tests for weight quantization functions."""
import pytest
import torch
from bitlab.bitquantizer.registry import WEIGHT_QUANT_REGISTRY

WEIGHT_QUANT_FUNCTIONS = list(WEIGHT_QUANT_REGISTRY.values())
WEIGHT_QUANT_NAMES = list(WEIGHT_QUANT_REGISTRY.keys())


# Fixtures
@pytest.fixture
def sample_weights():
    """Fixture providing sample weight tensors for linear layers (2D)."""
    return torch.randn(4, 8)


@pytest.fixture
def sample_conv_weights():
    """Fixture providing sample weight tensors for conv layers (4D)."""
    return torch.randn(64, 32, 3, 3)


@pytest.fixture(params=zip(WEIGHT_QUANT_NAMES, WEIGHT_QUANT_FUNCTIONS), ids=WEIGHT_QUANT_NAMES)
def weight_quantizer(request):
    """Fixture providing weight quantizer name and function."""
    return request.param


@pytest.mark.unit
class TestWeightQuantizationOutput:
    """Unit tests for weight quantization output format."""
    
    def test_output_shape(self, weight_quantizer, sample_weights):
        """Test that weight quantization returns correct output shapes for linear weights."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_weights
        
        scale, quantized = weight_quant_fn(w)
        
        # For 2D linear weights, scale should be scalar (per-tensor)
        assert scale.shape == (), f"Scale should be a scalar for linear weights with {weight_quant_name}"
        assert quantized.shape == w.shape, f"Quantized weight should have same shape as input for {weight_quant_name}"
    
    def test_output_type(self, weight_quantizer, sample_weights):
        """Test that weight quantization returns torch.Tensor."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_weights
        
        scale, quantized = weight_quant_fn(w)
        
        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)


@pytest.mark.unit
class TestWeightQuantizationRange:
    """Unit tests for weight quantization value ranges."""
    
    def test_quantized_values_in_range(self, weight_quantizer, sample_weights):
        """Test that quantized values are in [-1, 1] range."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_weights
        
        _, quantized = weight_quant_fn(w)
        
        assert (quantized >= -1).all(), f"Quantized values should be >= -1 for {weight_quant_name}"
        assert (quantized <= 1).all(), f"Quantized values should be <= 1 for {weight_quant_name}"


@pytest.mark.unit
class TestConvWeightQuantizationOutput:
    """Unit tests for conv2d weight quantization (4D tensors) output format."""
    
    def test_output_shape(self, weight_quantizer, sample_conv_weights):
        """Test that weight quantization returns correct output shapes for conv2d weights."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_conv_weights
        
        scale, quantized = weight_quant_fn(w)
        
        assert isinstance(scale, torch.Tensor), f"Scale should be torch.Tensor for {weight_quant_name}"
        assert quantized.shape == w.shape, f"Quantized weight should have same shape as input for {weight_quant_name}"
    
    def test_output_type(self, weight_quantizer, sample_conv_weights):
        """Test that weight quantization returns torch.Tensor for conv2d weights."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_conv_weights
        
        scale, quantized = weight_quant_fn(w)
        
        assert isinstance(scale, torch.Tensor)
        assert isinstance(quantized, torch.Tensor)
    
    def test_scale_shape_for_per_channel(self, sample_conv_weights):
        """Test that wpt produces correct scale shape for conv2d (per-output-channel)."""
        from bitlab.bitquantizer.weight import quantize_weight_wpt
        w = sample_conv_weights  # [out_channels, in_channels, kernel_h, kernel_w]
        
        scale, quantized = quantize_weight_wpt(w)
        
        # For conv2d, wpt should produce per-output-channel scales
        expected_scale_shape = (w.shape[0], 1, 1, 1)
        assert scale.shape == expected_scale_shape, f"Scale shape should be {expected_scale_shape} for conv2d, got {scale.shape}"
        assert quantized.shape == w.shape


@pytest.mark.unit
class TestConvWeightQuantizationRange:
    """Unit tests for conv2d weight quantization value ranges."""
    
    def test_quantized_values_in_range(self, weight_quantizer, sample_conv_weights):
        """Test that quantized values are in [-1, 1] range for conv2d weights."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_conv_weights
        
        _, quantized = weight_quant_fn(w)
        
        assert (quantized >= -1).all(), f"Quantized values should be >= -1 for {weight_quant_name}"
        assert (quantized <= 1).all(), f"Quantized values should be <= 1 for {weight_quant_name}"

