import pytest

import bitlab.bitmodels  # noqa: F401 - ensures registry population via side effects
from bitlab.bitmodels.auto import (
    BitAutoModel,
    BitAutoModelForCausalLM,
    BitAutoModelForImageClassification,
    BitAutoModelForImageGeneration,
    MODEL_REGISTRY,
    TASK_REGISTRY,
)
from bitlab.bitmodels.tasks import ModelTask

MODEL_OVERRIDES = {
    "bitnet": {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 128,
    },
    "bitmlp": {
        "input_size": 16,
        "hidden_dims": (8,),
        "num_classes": 2,
        "quant_type": None,
    },
    "bitresnet": {
        "num_classes": 2,
        "in_channels": 3,
        "base_channels": 8,
        "block_layers": (1, 1, 1, 1),
        "quant_type": None,
    },
    "bitunet": {
        "image_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "model_channels": 32,
        "num_res_blocks": 1,
        "attention_resolutions": (4,),
        "channel_mult": (1, 2),
        "num_heads": 1,
    },
}


def _iter_unique_models():
    seen = set()
    for model_type, model_cls in MODEL_REGISTRY.items():
        if model_cls in seen:
            continue
        seen.add(model_cls)
        yield model_type, model_cls


TASK_AUTO_MAP = {
    ModelTask.CAUSAL_LM.value: BitAutoModelForCausalLM,
    ModelTask.IMAGE_CLASSIFICATION.value: BitAutoModelForImageClassification,
    ModelTask.IMAGE_GENERATION.value: BitAutoModelForImageGeneration,
}


def _iter_task_auto_cases():
    seen = set()
    for task_key, registry in TASK_REGISTRY.items():
        auto_cls = TASK_AUTO_MAP.get(task_key)
        if auto_cls is None:
            continue
        for model_type, model_cls in registry.items():
            if model_cls in seen:
                continue
            seen.add(model_cls)
            yield auto_cls, model_type, model_cls


UNIQUE_MODELS = sorted(_iter_unique_models(), key=lambda item: item[0])
AUTO_MODEL_CASES = sorted(
    _iter_task_auto_cases(), key=lambda item: (item[0].__name__, item[1])
)


@pytest.mark.parametrize("model_type, model_cls", UNIQUE_MODELS)
def test_bitmodel_registry_instantiation(model_type, model_cls):
    config_cls = getattr(model_cls, "config_cls", None)
    assert (
        config_cls is not None
    ), f"{model_cls.__name__} must define `config_cls` for this test."

    overrides = MODEL_OVERRIDES.get(model_type, {})
    config = config_cls(**overrides) if overrides else config_cls()

    model = model_cls(config)

    assert isinstance(model, model_cls)
    assert model.config == config


@pytest.mark.parametrize("auto_cls, model_type, model_cls", AUTO_MODEL_CASES)
def test_auto_model_task_from_config(auto_cls, model_type, model_cls):
    config_cls = getattr(model_cls, "config_cls")
    overrides = MODEL_OVERRIDES.get(model_type, {})
    config = config_cls(**overrides) if overrides else config_cls()

    auto_model = auto_cls.from_config(config)

    assert isinstance(auto_model, model_cls)
    assert auto_model.config == config
    assert not auto_model.training


@pytest.mark.parametrize("model_type, model_cls", UNIQUE_MODELS)
def test_bitautomodel_from_config(model_type, model_cls):
    config_cls = getattr(model_cls, "config_cls")
    overrides = MODEL_OVERRIDES.get(model_type, {})
    config = config_cls(**overrides) if overrides else config_cls()

    model = BitAutoModel.from_config(config)

    assert isinstance(model, model_cls)
    assert model.config == config
    assert not model.training

