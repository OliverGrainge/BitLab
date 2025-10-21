import pytest
import torch
from bitlayers import LAYER_REGISTRY, register_layer, BitLinear


class TestLayerRegistry:
    """Test cases for layer registration system"""
    
    def test_bitlinear_is_registered(self):
        """Test that BitLinear is registered in the registry"""
        assert "BitLinear" in LAYER_REGISTRY
        assert LAYER_REGISTRY["BitLinear"] == BitLinear

    
    def test_registered_layer_instantiation(self):
        """Test that registered layers can be instantiated"""
        # Test BitLinear instantiation
        layer = LAYER_REGISTRY["BitLinear"](in_features=10, out_features=5)
        assert isinstance(layer, BitLinear)
        assert layer.in_features == 10
        assert layer.out_features == 5
    
    
    def test_registry_is_mutable(self):
        """Test that the registry can be modified"""
        original_size = len(LAYER_REGISTRY)
        
        # Add a test entry
        LAYER_REGISTRY["TestEntry"] = "test_value"
        assert len(LAYER_REGISTRY) == original_size + 1
        assert LAYER_REGISTRY["TestEntry"] == "test_value"
        
        # Remove the test entry
        del LAYER_REGISTRY["TestEntry"]
        assert len(LAYER_REGISTRY) == original_size
        assert "TestEntry" not in LAYER_REGISTRY
