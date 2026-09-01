"""DeepECG-Kit: Deep learning library for ECG analysis and arrhythmia classification."""

from importlib.metadata import PackageNotFoundError, version

from deepecgkit.datasets import ECGDataModule
from deepecgkit.explainability import Attribution, ECGExplainer, explain
from deepecgkit.features import FeatureExtractor, FeatureSet, extract_features
from deepecgkit.models import KanResWideX
from deepecgkit.training import ECGTrainer
from deepecgkit.utils import read_csv

try:
    __version__ = version("deepecgkit")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__author__ = "DeepECG Kit Contributors"
__license__ = "Apache-2.0"

__all__ = [
    "Attribution",
    "ECGDataModule",
    "ECGExplainer",
    "ECGTrainer",
    "FeatureExtractor",
    "FeatureSet",
    "KanResWideX",
    "explain",
    "extract_features",
    "read_csv",
]
