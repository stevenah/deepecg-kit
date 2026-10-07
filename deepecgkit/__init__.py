"""DeepECG-Kit: Deep learning library for ECG analysis and arrhythmia classification."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepecgkit.datasets import ECGDataModule
    from deepecgkit.evaluation import ECGEvaluator
    from deepecgkit.explainability import Attribution, ECGExplainer, explain
    from deepecgkit.features import FeatureExtractor, FeatureSet, extract_features
    from deepecgkit.models import KanResWideX
    from deepecgkit.models.factory import build_model, build_model_from_checkpoint
    from deepecgkit.registry import (
        get_dataset_names,
        get_model_names,
        register_dataset,
        register_model,
    )
    from deepecgkit.training import ECGTrainer
    from deepecgkit.utils import read_csv
    from deepecgkit.utils.device import resolve_device

try:
    __version__ = version("deepecgkit")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__author__ = "DeepECG Kit Contributors"
__license__ = "Apache-2.0"

__all__ = [
    "Attribution",
    "ECGDataModule",
    "ECGEvaluator",
    "ECGExplainer",
    "ECGTrainer",
    "FeatureExtractor",
    "FeatureSet",
    "KanResWideX",
    "build_model",
    "build_model_from_checkpoint",
    "explain",
    "extract_features",
    "get_dataset_names",
    "get_model_names",
    "read_csv",
    "register_dataset",
    "register_model",
    "resolve_device",
]

#: Where each public name lives. A name is imported when it is first asked
#: for, so `import deepecgkit` and the commands that need little of the
#: package do not load PyTorch and the rest of the scientific stack.
_EXPORTS = {
    "Attribution": "deepecgkit.explainability",
    "ECGDataModule": "deepecgkit.datasets",
    "ECGEvaluator": "deepecgkit.evaluation",
    "ECGExplainer": "deepecgkit.explainability",
    "ECGTrainer": "deepecgkit.training",
    "FeatureExtractor": "deepecgkit.features",
    "FeatureSet": "deepecgkit.features",
    "KanResWideX": "deepecgkit.models",
    "build_model": "deepecgkit.models.factory",
    "build_model_from_checkpoint": "deepecgkit.models.factory",
    "explain": "deepecgkit.explainability",
    "extract_features": "deepecgkit.features",
    "get_dataset_names": "deepecgkit.registry",
    "get_model_names": "deepecgkit.registry",
    "read_csv": "deepecgkit.utils",
    "register_dataset": "deepecgkit.registry",
    "register_model": "deepecgkit.registry",
    "resolve_device": "deepecgkit.utils.device",
}
_SUBMODULES = frozenset(
    {
        "analysis",
        "cli",
        "datasets",
        "evaluation",
        "explainability",
        "features",
        "models",
        "registry",
        "training",
        "utils",
    }
)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_SUBMODULES})
