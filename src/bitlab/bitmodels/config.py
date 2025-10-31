from typing import Any
from pathlib import Path
import json

from pydantic import BaseModel, ConfigDict


CONFIG_REGISTRY: dict[str, type["BaseBitModelConfig"]] = {}


def register_bitconfig(model_type: str):
    def deco(cls):
        CONFIG_REGISTRY[model_type] = cls
        return cls

    return deco


class BaseBitModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model_type: str
    name_or_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())

    def with_overrides(self, **updates: Any) -> "BaseBitModelConfig":
        return self.model_copy(update=updates)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseBitModelConfig":
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> "BaseBitModelConfig":
        """Load from <path>/config.json (path may be a dir or a .json file)."""
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p / "config.json"
        data = json.loads(p.read_text())
        model_type = data.get("model_type")
        if model_type is None:
            raise ValueError("Config missing required field 'model_type'.")
        config_cls = CONFIG_REGISTRY.get(model_type, cls)
        return config_cls.from_dict(data)


# Backwards compatibility alias for older imports
BaseModelConfig = BaseBitModelConfig
