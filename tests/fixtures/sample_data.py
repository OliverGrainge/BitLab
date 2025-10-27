"""
Shared test data and fixtures for BitLab tests
"""
import torch


def get_sample_2d_tensor(batch_size=32, features=10):
    """Get sample 2D tensor"""
    return torch.randn(batch_size, features)


def get_sample_3d_tensor(batch_size=2, seq_len=3, features=10):
    """Get sample 3D tensor"""
    return torch.randn(batch_size, seq_len, features)


def get_sample_targets(batch_size=32, output_features=5):
    """Get sample target tensor"""
    return torch.randn(batch_size, output_features)


def get_sample_weights(in_features=10, out_features=5):
    """Get sample weight tensor"""
    return torch.randn(out_features, in_features)


def get_sample_bias(out_features=5):
    """Get sample bias tensor"""
    return torch.randn(out_features)
