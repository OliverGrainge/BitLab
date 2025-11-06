"""Unit tests for weight quantization functions."""
import pytest
import torch
from bitlab.bitquantizer.registry import WEIGHT_QUANT_REGISTRY

WEIGHT_QUANT_FUNCTIONS = list(WEIGHT_QUANT_REGISTRY.values())
WEIGHT_QUANT_NAMES = list(WEIGHT_QUANT_REGISTRY.keys())


# Fixtures
@pytest.fixture
def sample_weights():
    """Fixture providing sample weight tensors."""
    return torch.randn(4, 8)


@pytest.fixture(params=zip(WEIGHT_QUANT_NAMES, WEIGHT_QUANT_FUNCTIONS), ids=WEIGHT_QUANT_NAMES)
def weight_quantizer(request):
    """Fixture providing weight quantizer name and function."""
    return request.param


@pytest.mark.unit
class TestWeightQuantizationOutput:
    """Unit tests for weight quantization output format."""
    
    def test_output_shape(self, weight_quantizer, sample_weights):
        """Test that weight quantization returns correct output shapes."""
        weight_quant_name, weight_quant_fn = weight_quantizer
        w = sample_weights
        
        scale, quantized = weight_quant_fn(w)
        
        assert scale.shape == (), f"Scale should be a scalar for {weight_quant_name}"
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

