# bitlab/bitmodels/auto.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Type
import torch
from bitlab.bitmodels.config import BaseBitModelConfig

# Minimal registry for models
MODEL_REGISTRY: dict[str, type] = {}


def register_bitmodel(model_type: str):
    """Decorator to register a model class under its `model_type`."""

    def deco(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls

    return deco




class BitAutoModel:
    @classmethod
    def from_config(
        cls,
        config: BaseBitModelConfig | str | Path,
        *,
        eval_mode: bool = True,
        **config_overrides,
    ):
        """Instantiate a registered model directly from a config object."""

        if isinstance(config, (str, Path)):
            cfg = BaseBitModelConfig.load(config)
        else:
            cfg = config

        if config_overrides:
            cfg = cfg.with_overrides(**config_overrides)

        model_cls: Optional[Type] = MODEL_REGISTRY.get(cfg.model_type)
        if model_cls is None:
            raise ValueError(
                f"No model registered for type '{cfg.model_type}'. "
                f"Registered: {list(MODEL_REGISTRY.keys())}"
            )

        model = model_cls(cfg)
        if eval_mode and hasattr(model, "eval"):
            model.eval()
        return model


