from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar

import torch
import torch.nn as nn

from bitlab.bitmodels.config import BaseBitModelConfig

ConfigT = TypeVar("ConfigT", bound=BaseBitModelConfig)


class BaseBitModel(nn.Module):
    """
    Lightweight foundation for Bit models.

    Responsibilities:
      * Normalize config construction + validation
      * Expose convenience accessors for quantization choices
      * Provide a shared `deploy` implementation that cooperates with Bit layers
      * Optionally bootstrap a `self.model` backbone via `build_model`
    """

    #: Concrete subclasses can set this to their config type to enable the
    #: automatic config lifecycle management.
    config_cls: ClassVar[Optional[Type[BaseBitModelConfig]]] = None

    #: Toggle automatic invocation of `build_model` during __init__.
    auto_build: ClassVar[bool] = True

    def __init__(
        self,
        config: Optional[BaseBitModelConfig] = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        self.config = self._resolve_config(config, overrides)
        self.quant_type = self._infer_quant_type(self.config)

        if self.auto_build and self.config is not None and self._has_custom_build():
            self.model = self.build_model(self.config)  # type: ignore[attr-defined]

    # --------------------------------------------------------------------- #
    # Hooks for subclasses
    # --------------------------------------------------------------------- #
    def build_model(self, config: BaseBitModelConfig) -> nn.Module:
        raise NotImplementedError(
            f"{self.__class__.__name__} must override `build_model` or provide a custom "
            "`forward` implementation."
        )

    # --------------------------------------------------------------------- #
    # Shared helpers
    # --------------------------------------------------------------------- #
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Default forward delegates to an auto-built backbone if one exists.
        Subclasses remain free to override for custom signatures (e.g., UNet).
        """
        if hasattr(self, "model"):
            model = getattr(self, "model")
            return model(*args, **kwargs)
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define `forward` and no `self.model` "
            "backbone is available."
        )

    def deploy(self) -> nn.Module:
        """Invoke `_deploy` on child modules (where available) and return self."""
        for module in self.modules():
            if module is self:
                continue
            deploy_fn = getattr(module, "_deploy", None)
            if callable(deploy_fn):
                deploy_fn()
        return self

    # Quantization -------------------------------------------------------- #
    def uses_quantization(self) -> bool:
        """Whether the resolved config enables Bit quantization."""
        return self.quant_type is not None or self.quant_type != "none"

    def get_quant_type(self, default: Optional[str] = None) -> Optional[str]:
        """Return the configured quantization type, falling back to `default`."""
        return self.quant_type if self.quant_type is not None else default

    # Internals ----------------------------------------------------------- #
    def _resolve_config(
        self,
        config: Optional[BaseBitModelConfig],
        overrides: Dict[str, Any],
    ) -> Optional[BaseBitModelConfig]:
        config_cls = self.config_cls

        if config_cls is None:
            if config is not None and overrides:
                config = config.with_overrides(**overrides)
            elif overrides:
                raise TypeError(
                    f"{self.__class__.__name__} does not declare `config_cls` but "
                    "received config overrides."
                )
            return config

        if config is None:
            return config_cls(**overrides)

        if not isinstance(config, config_cls):
            raise TypeError(f"config must be an instance of {config_cls.__name__}")

        if overrides:
            config = config.with_overrides(**overrides)

        return config

    def _infer_quant_type(self, config: Optional[BaseBitModelConfig]) -> Optional[str]:
        if config is None:
            return None
        return getattr(config, "quant_type", None)

    def _has_custom_build(self) -> bool:
        return type(self).build_model is not BaseBitModel.build_model

    @classmethod
    def _load_weights(
        cls,
        *,
        weights: Optional[str] = None,
        **kwargs: Any,
    ) -> "BaseBitModel":
        raise NotImplementedError(f"{cls.__name__} must override `_load_weights`")
