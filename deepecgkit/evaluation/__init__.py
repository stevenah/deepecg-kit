"""
Model evaluation and metrics module for DeepECG-Kit.

This module provides comprehensive evaluation functionality including:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Regression metrics (MSE, MAE, R²)
- ECG-specific metrics
- Visualization tools
- Statistical analysis
"""

from deepecgkit.evaluation.evaluator import ECGEvaluator
from deepecgkit.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    confusion_matrix_analysis,
    roc_auc_score,
)
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_ecg_signals,
    plot_predictions,
    plot_roc_curve,
    plot_training_curves,
)

__all__ = [
    "ECGEvaluator",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "confusion_matrix_analysis",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_ecg_signals",
    "plot_predictions",
    "plot_roc_curve",
    "plot_training_curves",
    "roc_auc_score",
]
