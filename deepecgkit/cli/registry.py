"""Registry helpers and constants for the CLI."""

import deepecgkit.datasets
import deepecgkit.models  # noqa: F401
from deepecgkit.models.pretrained import model_name_for_weights
from deepecgkit.registry import (
    _DATASET_REGISTRY,
    _MODEL_REGISTRY,
    get_dataset_info,
    get_dataset_names,
    get_model_names,
)
from deepecgkit.utils.weights import WEIGHTS_REGISTRY

MODEL_NAMES = get_model_names()
DATASET_NAMES = get_dataset_names()
DATASET_NUM_CLASSES = {n: get_dataset_info(n)["num_classes"] for n in DATASET_NAMES}
DATASET_INPUT_CHANNELS = {n: info["input_channels"] for n, info in _DATASET_REGISTRY.items()}
MODEL_DESCRIPTIONS = {n: info["description"] for n, info in _MODEL_REGISTRY.items()}


def _build_model_weights() -> dict[str, list[str]]:
    """Map each registered model to the pretrained weights that target it."""
    mapping: dict[str, list[str]] = {}
    for weight_name in WEIGHTS_REGISTRY:
        model_name = model_name_for_weights(weight_name)
        if model_name is not None:
            mapping.setdefault(model_name, []).append(weight_name)
    return mapping


MODEL_WEIGHTS = _build_model_weights()


def _get_model_registry() -> dict[str, type]:
    """Return model registry mapping names to classes."""
    return {name: info["class"] for name, info in _MODEL_REGISTRY.items()}


def _get_dataset_registry() -> dict[str, type]:
    """Return dataset registry mapping names to classes."""
    return {name: info["class"] for name, info in _DATASET_REGISTRY.items()}
