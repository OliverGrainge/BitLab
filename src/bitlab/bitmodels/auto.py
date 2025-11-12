# bitlab/bitmodels/auto.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

from bitlab.bitmodels.config import BaseBitModelConfig
from bitlab.bitmodels.base import BaseBitModel

# Minimal registry for models
MODEL_REGISTRY: dict[str, type] = {}


def register_bitmodel(model_type: str):
    """Decorator to register a model class under its `model_type`."""

    def deco(cls):
        MODEL_REGISTRY[model_type] = cls
        MODEL_REGISTRY[model_type.lower()] = cls
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

        model_cls = cls._resolve_model_cls(cfg.model_type)

        model = model_cls(cfg)
        if eval_mode and hasattr(model, "eval"):
            model.eval()
        return model

    @classmethod
    def _resolve_model_cls(cls, model_type: str) -> Type[BaseBitModel]:
        candidates = (
            MODEL_REGISTRY.get(model_type),
            MODEL_REGISTRY.get(model_type.lower()) if isinstance(model_type, str) else None,
        )
        for candidate in candidates:
            if candidate is not None:
                return candidate
        raise ValueError(
            f"No model registered for type '{model_type}'. "
            f"Registered: {sorted(set(MODEL_REGISTRY.keys()))}"
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        eval_mode: bool = True,
    ) -> BaseBitModel:
        """
        Instantiate a registered model and load pretrained weights.

        The `name` can be either a bare model type (e.g. `"bitnet"`) or a
        composite string of the form `"bitnet:base"` where the part after the
        colon selects a specific weight alias exposed by the model.
        """

        alias: Optional[str] = None
        if ":" in name:
            model_type, alias = name.split(":", 1)
            alias = alias or None
        else:
            model_type = name

        model_cls = cls._resolve_model_cls(model_type)
        loader = getattr(model_cls, "_load_weights", None)
        if not callable(loader):
            raise TypeError(
                f"{model_cls.__name__} does not expose a callable `_load_weights` loader."
            )

        if alias is not None:
            model = loader(alias)
        else:
            model = loader()
        if not isinstance(model, BaseBitModel):
            raise TypeError(
                f"Expected `_load_weights` of {model_cls.__name__} to return a BaseBitModel, "
                f"got {type(model)}."
            )

        if eval_mode and hasattr(model, "eval"):
            model.eval()
        return model
