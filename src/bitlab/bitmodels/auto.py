# bitlab/bitmodels/auto.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Optional, Type

from bitlab.bitmodels.base import BaseBitModel
from bitlab.bitmodels.config import BaseBitModelConfig
from bitlab.bitmodels.tasks import ModelTask

# Minimal registry for models
MODEL_REGISTRY: dict[str, type[BaseBitModel]] = {}
TASK_REGISTRY: dict[str, dict[str, type[BaseBitModel]]] = defaultdict(dict)


def register_bitmodel(model_type: str, *, task: ModelTask | str | None = None):
    """Decorator to register a model class under its `model_type`."""

    def deco(cls):
        MODEL_REGISTRY[model_type] = cls
        MODEL_REGISTRY[model_type.lower()] = cls

        resolved_task = task or getattr(cls, "task", None)
        if resolved_task is not None:
            task_key = (
                resolved_task.value
                if isinstance(resolved_task, ModelTask)
                else str(resolved_task)
            )
            TASK_REGISTRY[task_key][model_type] = cls
            TASK_REGISTRY[task_key][model_type.lower()] = cls
        return cls

    return deco


class BitAutoModel:
    task: ClassVar[ModelTask | None] = None

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
    def _resolve_model_cls(cls, model_type: Optional[str]) -> Type[BaseBitModel]:
        if model_type:
            candidates = (
                MODEL_REGISTRY.get(model_type),
                (
                    MODEL_REGISTRY.get(model_type.lower())
                    if isinstance(model_type, str)
                    else None
                ),
            )
            for candidate in candidates:
                if candidate is not None:
                    if cls.task is not None:
                        task_key = (
                            cls.task.value
                            if isinstance(cls.task, ModelTask)
                            else str(cls.task)
                        )
                        registered_task = getattr(candidate, "task", None)
                        if registered_task is None:
                            registered_task = next(
                                (
                                    key
                                    for key, registry in TASK_REGISTRY.items()
                                    if candidate in registry.values()
                                ),
                                None,
                            )
                        if registered_task is not None:
                            registered_task = (
                                registered_task.value
                                if isinstance(registered_task, ModelTask)
                                else str(registered_task)
                            )
                        if registered_task is not None and registered_task != task_key:
                            raise ValueError(
                                f"Model '{model_type}' is registered for task '{registered_task}', "
                                f"but '{task_key}' was requested."
                            )
                    return candidate

            raise ValueError(
                f"No model registered for type '{model_type}'. "
                f"Registered: {sorted(set(MODEL_REGISTRY.keys()))}"
            )

        if cls.task is None:
            raise ValueError(
                "model_type must be provided for BitAutoModel without a predefined task."
            )

        task_key = cls.task.value if isinstance(cls.task, ModelTask) else str(cls.task)
        task_registry = TASK_REGISTRY.get(task_key, {})
        if not task_registry:
            raise ValueError(f"No models registered for task '{task_key}'.")
        if len(task_registry) == 1:
            return next(iter(task_registry.values()))

        raise ValueError(
            f"Multiple models registered for task '{task_key}'. "
            "Please specify `model_type` explicitly."
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


class BitAutoModelForCausalLM(BitAutoModel):
    """Auto loader for causal language models."""

    task: ClassVar[ModelTask] = ModelTask.CAUSAL_LM


class BitAutoModelForImageClassification(BitAutoModel):
    """Auto loader for image classification models."""

    task: ClassVar[ModelTask] = ModelTask.IMAGE_CLASSIFICATION


class BitAutoModelForImageGeneration(BitAutoModel):
    """Auto loader for unconditional image generation models."""

    task: ClassVar[ModelTask] = ModelTask.IMAGE_GENERATION
