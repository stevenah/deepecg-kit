"""
Model and dataset registry for deepecg-kit.

Provides decorator-based registration so new models and datasets are
automatically discoverable by the CLI and other consumers.

Usage:
    from deepecgkit.registry import register_model, register_dataset

    @register_model("my-model", description="My custom model")
    class MyModel(nn.Module):
        ...

    @register_dataset("my-dataset", input_channels=1, description="My dataset")
    class MyDataset(BaseECGDataset):
        ...
"""

import importlib
from collections.abc import Callable
from typing import Any

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}
_DATASET_REGISTRY: dict[str, dict[str, Any]] = {}

#: The packages whose modules register the built-in models and datasets when
#: imported. Nothing imports them up front, so that `import deepecgkit` stays
#: cheap; the first lookup does.
_BUILTIN_PACKAGES = {"models": "deepecgkit.models", "datasets": "deepecgkit.datasets"}
_loaded_builtins: set[str] = set()


def _load_builtins(kind: str) -> None:
    """Import the built-in models or datasets so that they are registered.

    Args:
        kind: "models" or "datasets"
    """
    if kind in _loaded_builtins:
        return
    # Marked first: importing the package calls back into this module.
    _loaded_builtins.add(kind)
    try:
        importlib.import_module(_BUILTIN_PACKAGES[kind])
    except BaseException:
        _loaded_builtins.discard(kind)
        raise


def register_model(
    name: str,
    description: str = "",
    default_kwargs: dict[str, Any] | None = None,
    min_input_length: int | None = None,
    max_input_length: int | None = None,
) -> Callable:
    """Register a model class under a short CLI/registry name.

    Args:
        name: Registry name ("resnet", "kanres", ...)
        description: One-line description shown by list-models
        default_kwargs: Constructor arguments applied by build_model unless
            overridden by the caller
        min_input_length: Smallest signal length (samples) the architecture
            accepts, when it has a floor
        max_input_length: Largest signal length it accepts, when it has a
            ceiling (positional-encoding limits)

    Returns:
        A class decorator that records the class and returns it unchanged.
    """

    def decorator(cls: type) -> type:
        _MODEL_REGISTRY[name] = {
            "class": cls,
            "description": description,
            "default_kwargs": default_kwargs or {},
            "head_attr": getattr(cls, "HEAD_ATTR", None),
            "min_input_length": min_input_length,
            "max_input_length": max_input_length,
        }
        cls._registry_name = name
        return cls

    return decorator


def register_dataset(
    name: str,
    input_channels: int,
    num_classes: int | None = None,
    description: str = "",
) -> Callable:
    """Register a dataset class under a short CLI/registry name.

    Args:
        name: Registry name ("ptbxl", "ltafdb", ...)
        input_channels: Lead count the dataset yields by default
        num_classes: Class count; derived from CLASS_LABELS when omitted
        description: One-line description shown by list-datasets

    Returns:
        A class decorator that records the class and returns it unchanged.
    """

    def decorator(cls: type) -> type:
        _DATASET_REGISTRY[name] = {
            "class": cls,
            "input_channels": input_channels,
            "num_classes": num_classes,
            "description": description,
        }
        cls._registry_name = name
        return cls

    return decorator


def get_model(name: str) -> type:
    """Return the registered model class for a name.

    Raises:
        KeyError: If the name is not registered.
    """
    _load_builtins("models")
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model: {name}. Available: {available}")
    return _MODEL_REGISTRY[name]["class"]


def get_dataset(name: str) -> type:
    """Return the registered dataset class for a name.

    Raises:
        KeyError: If the name is not registered.
    """
    _load_builtins("datasets")
    if name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASET_REGISTRY[name]["class"]


def get_model_names() -> list[str]:
    """Return every registered model name, sorted."""
    _load_builtins("models")
    return sorted(_MODEL_REGISTRY.keys())


def get_dataset_names() -> list[str]:
    """Return every registered dataset name, sorted."""
    _load_builtins("datasets")
    return sorted(_DATASET_REGISTRY.keys())


def get_model_info(name: str) -> dict[str, Any]:
    """Return a model's registry entry: class, description, default_kwargs,
    head_attr, and input-length bounds.

    Raises:
        KeyError: If the name is not registered.
    """
    _load_builtins("models")
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return dict(_MODEL_REGISTRY[name])


def get_dataset_info(name: str) -> dict[str, Any]:
    """Return a dataset's registry entry: class, input_channels, num_classes,
    and description; num_classes falls back to len(CLASS_LABELS).

    Raises:
        KeyError: If the name is not registered.
    """
    _load_builtins("datasets")
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    entry = dict(_DATASET_REGISTRY[name])
    if entry["num_classes"] is None:
        cls = entry["class"]
        if hasattr(cls, "CLASS_LABELS"):
            entry["num_classes"] = len(cls.CLASS_LABELS)
    return entry
