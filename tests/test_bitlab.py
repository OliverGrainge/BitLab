import pytest
import torch
import bitlab.bnn as bnn
from bitlab.bnn import BitLinear, Module
from bitlab.bitquantizer import BitQuantizer


def test_bitlinear_creation():
    """Test BitLinear layer creation."""
    layer = BitLinear(10, 5)
    assert layer.in_features == 10
    assert layer.out_features == 5
    assert layer.weight.shape == (5, 10)


def test_bitlinear_forward():
    """Test BitLinear forward pass."""
    layer = BitLinear(10, 5)
    x = torch.randn(3, 10)
    output = layer(x)
    assert output.shape == (3, 5)


def test_bitquantizer():
    """Test BitQuantizer functionality."""
    quantizer = BitQuantizer()
    x = torch.randn(3, 10)
    w = torch.randn(5, 10)
    
    qx, qw = quantizer(x, w)
    assert qx.shape == x.shape
    assert qw.shape == w.shape


def test_module_deploy():
    """Test Module deployment functionality."""
    class TestModel(Module):
        def __init__(self):
            super().__init__()
            self.layer = BitLinear(10, 5)
        
        def forward(self, x):
            return self.layer(x)
    
    model = TestModel()
    x = torch.randn(3, 10)
    
    # Test before deployment
    output1 = model(x)
    assert output1.shape == (3, 5)
    
    # Test after deployment
    deployed_model = model.deploy()
    output2 = deployed_model(x)
    assert output2.shape == (3, 5)


def test_package_imports():
    """Test that package imports work correctly."""
    import bitlab
    import bitlab.bnn
    import bitlab.bitquantizer
    
    assert hasattr(bitlab.bnn, 'BitLinear')
    assert hasattr(bitlab.bnn, 'Module')
    assert hasattr(bitlab.bitquantizer, 'BitQuantizer')


if __name__ == "__main__":
    pytest.main([__file__])
