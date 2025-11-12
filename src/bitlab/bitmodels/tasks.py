from __future__ import annotations

from enum import Enum


class ModelTask(str, Enum):
    """Enumeration of supported BitLab model tasks."""

    CAUSAL_LM = "causal-lm"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_GENERATION = "image-generation"


