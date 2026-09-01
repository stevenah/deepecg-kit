"""DeepECG-Kit: Deep learning library for ECG analysis and arrhythmia classification."""

from importlib.metadata import PackageNotFoundError, version

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
