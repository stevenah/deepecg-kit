"""Registry helpers and constants for the CLI."""

import deepecgkit.datasets
import deepecgkit.models  # noqa: F401
from deepecgkit.registry import (
    _DATASET_REGISTRY,
    _MODEL_REGISTRY,
    get_dataset_info,
    get_dataset_names,
    get_model_names,
)

MODEL_NAMES = get_model_names()
DATASET_NAMES = get_dataset_names()
DATASET_NUM_CLASSES = {n: get_dataset_info(n)["num_classes"] for n in DATASET_NAMES}
DATASET_INPUT_CHANNELS = {n: info["input_channels"] for n, info in _DATASET_REGISTRY.items()}
MODEL_DESCRIPTIONS = {n: info["description"] for n, info in _MODEL_REGISTRY.items()}

MODEL_WEIGHTS = {
    "afmodel": ["afmodel-30s"],
    "kanres": ["kanres-af-30s"],
}


def _get_model_registry():
    """Return model registry mapping names to classes."""
    return {name: info["class"] for name, info in _MODEL_REGISTRY.items()}


def _get_dataset_registry():
    """Return dataset registry mapping names to classes."""
    return {name: info["class"] for name, info in _DATASET_REGISTRY.items()}
