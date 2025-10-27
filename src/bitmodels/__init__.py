# bitmodels/__init__.py

MODEL_REGISTRY = {}
CONFIG_REGISTRY = {}


def register_model(name: str):
    """Register a model class"""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_config(name: str):
    """Register a config class"""

    def decorator(cls):
        CONFIG_REGISTRY[name] = cls
        return cls

    return decorator


class AutoBitModel:
    """Auto class for building models from configs"""

    @staticmethod
    def from_config(config, quant_config=None):
        """Build model automatically from config type"""
        config_name = config.__class__.__name__

        # Map config name to model name (remove 'Config' suffix and add 'Model')
        if config_name.endswith("Config"):
            base_name = config_name[:-6]  # Remove 'Config'
            model_name = base_name + "Model"
        else:
            model_name = config_name + "Model"

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"No model found for config {config_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        return MODEL_REGISTRY[model_name](config, quant_config)

    @staticmethod
    def from_pretrained(path: str):
        raise NotImplementedError("Not implemented yet")


def build_model(name: str, config, quant_config=None):
    """Build model from registry by name"""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {name} not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](config, quant_config)
