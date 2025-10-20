LAYER_REGISTRY = {}


def register_layer(name: str):
    """Register a model class"""

    def decorator(cls):
        LAYER_REGISTRY[name] = cls
        return cls

    return decorator

from .bitlinear import BitLinear 

__ALL__ = [
    "LAYER_REGISTRY",
    "BitLinear",
]
