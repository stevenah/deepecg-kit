"""DeepECG-Kit: Deep learning library for ECG analysis and arrhythmia classification."""

from deepecgkit.datasets import ECGDataModule
from deepecgkit.models import KanResWideX
from deepecgkit.training import ECGTrainer
from deepecgkit.utils import read_csv

__version__ = "0.1.0"
__author__ = "DeepECG Kit Contributors"
__license__ = "MIT"

__all__ = [
    "ECGDataModule",
    "ECGTrainer",
    "KanResWideX",
    "read_csv",
]
