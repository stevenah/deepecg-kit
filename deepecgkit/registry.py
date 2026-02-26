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

from typing import Any, Callable, Dict, List, Optional, Type

_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_model(
    name: str,
    description: str = "",
    default_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable:
    def decorator(cls: Type) -> Type:
        _MODEL_REGISTRY[name] = {
            "class": cls,
            "description": description,
            "default_kwargs": default_kwargs or {},
        }
        cls._registry_name = name
        return cls

    return decorator


def register_dataset(
    name: str,
    input_channels: int,
    num_classes: Optional[int] = None,
    description: str = "",
) -> Callable:
    def decorator(cls: Type) -> Type:
        _DATASET_REGISTRY[name] = {
            "class": cls,
            "input_channels": input_channels,
            "num_classes": num_classes,
            "description": description,
        }
        cls._registry_name = name
        return cls

    return decorator


def get_model(name: str) -> Type:
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model: {name}. Available: {available}")
    return _MODEL_REGISTRY[name]["class"]


def get_dataset(name: str) -> Type:
    if name not in _DATASET_REGISTRY:
        available = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASET_REGISTRY[name]["class"]


def get_model_names() -> List[str]:
    return sorted(_MODEL_REGISTRY.keys())


def get_dataset_names() -> List[str]:
    return sorted(_DATASET_REGISTRY.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return dict(_MODEL_REGISTRY[name])


def get_dataset_info(name: str) -> Dict[str, Any]:
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    entry = dict(_DATASET_REGISTRY[name])
    if entry["num_classes"] is None:
        cls = entry["class"]
        if hasattr(cls, "CLASS_LABELS"):
            entry["num_classes"] = len(cls.CLASS_LABELS)
    return entry
