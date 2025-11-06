"""Unit tests for Quantizer classes."""
import pytest
import torch
from bitlab.bitquantizer.registry import QUANTIZER_REGISTRY

QUANTIZER_CLASSES = list(QUANTIZER_REGISTRY.values())
QUANTIZER_NAMES = list(QUANTIZER_REGISTRY.keys())


# Fixtures
@pytest.fixture
def sample_activations():
    """Fixture providing sample activation tensors."""
    return torch.randn(4, 256)


@pytest.fixture
def sample_weights():
    """Fixture providing sample weight tensors."""
    return torch.randn(8, 256)


@pytest.fixture(params=zip(QUANTIZER_NAMES, QUANTIZER_CLASSES), ids=QUANTIZER_NAMES)
def quantizer(request):
    """Fixture providing quantizer name and class."""
    return request.param


@pytest.mark.unit
class TestQuantizerForward:
    """Unit tests for quantizer forward pass."""
    
    def test_forward_output_shape(self, quantizer, sample_activations, sample_weights):
        """Test that forward pass returns correct output shapes."""
        quantizer_name, quantizer_class = quantizer
        x, w = sample_activations, sample_weights
        
        dqx, dqw = quantizer_class.apply(x, w)
        
        assert dqx.shape == x.shape, f"Dequantized activation shape mismatch for {quantizer_name}"
        assert dqw.shape == w.shape, f"Dequantized weight shape mismatch for {quantizer_name}"
    
    def test_forward_output_type(self, quantizer, sample_activations, sample_weights):
        """Test that forward pass returns torch.Tensor."""
        quantizer_name, quantizer_class = quantizer
        x, w = sample_activations, sample_weights
        
        dqx, dqw = quantizer_class.apply(x, w)
        
        assert isinstance(dqx, torch.Tensor), f"Output should be torch.Tensor for {quantizer_name}"
        assert isinstance(dqw, torch.Tensor), f"Output should be torch.Tensor for {quantizer_name}"


@pytest.mark.unit
class TestQuantizerOutput:
    """Unit tests for quantizer output validation."""
    
    def test_output_is_tensor(self, quantizer, sample_activations, sample_weights):
        """Test that quantizer outputs are torch.Tensor instances."""
        quantizer_name, quantizer_class = quantizer
        x, w = sample_activations, sample_weights
        
        dqx, dqw = quantizer_class.apply(x, w)
        
        assert isinstance(dqx, torch.Tensor)
        assert isinstance(dqw, torch.Tensor)
    
    def test_output_preserves_shape(self, quantizer, sample_activations, sample_weights):
        """Test that quantizer outputs preserve input shapes."""
        quantizer_name, quantizer_class = quantizer
        x, w = sample_activations, sample_weights
        
        dqx, dqw = quantizer_class.apply(x, w)
        
        assert dqx.shape == x.shape
        assert dqw.shape == w.shape

