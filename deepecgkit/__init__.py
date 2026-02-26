"""
DeepECG-Kit: A comprehensive Python package for ECG analysis using deep learning.

This package provides tools for:
- Loading and preprocessing ECG data
- Training deep learning models on ECG signals
- Evaluating model performance
- Traditional ECG analysis methods
- Visualization and reporting

Basic usage:
    import deepecgkit as ecg


    data = ecg.load_ecg_data("path/to/data")


    preprocessed = ecg.preprocess(data)


    model = ecg.models.KanResWideX(input_channels=12, output_size=1)
    trainer = ecg.train_model(model, preprocessed)


    results = ecg.evaluate_model(model, test_data)
"""

from deepecgkit.datasets import ECGDataModule
from deepecgkit.models import KanResWideX
from deepecgkit.training import ECGLitModel
from deepecgkit.utils import read_csv

__version__ = "0.1.0"
__author__ = "DeepECG Kit Contributors"
__license__ = "MIT"

__all__ = [
    "ECGDataModule",
    "ECGLitModel",
    "KanResWideX",
    "read_csv",
]
