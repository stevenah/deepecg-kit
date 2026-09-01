"""
Model evaluation and metrics module for DeepECG-Kit.

This module provides comprehensive evaluation functionality including:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Regression metrics (MSE, MAE, R²)
- ECG-specific metrics
- Visualization tools
- Statistical analysis
"""

from deepecgkit.evaluation.calibration import (
    TemperatureScaler,
    apply_temperature,
    calibration_report,
    fit_temperature,
    probabilities_to_logits,
    resolve_calibration_task,
    temperature_nll,
)
from deepecgkit.evaluation.evaluator import ECGEvaluator
from deepecgkit.evaluation.metrics import (
    DEFAULT_CALIBRATION_BINS,
    TASK_BINARY,
    TASK_MULTICLASS,
    TASK_MULTILABEL,
    adaptive_calibration_error,
    brier_score,
    calculate_calibration_metrics,
    calculate_classification_metrics,
    calculate_regression_metrics,
    confusion_matrix_analysis,
    expected_calibration_error,
    maximum_calibration_error,
    resolve_average,
    resolve_task_shape,
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
    "DEFAULT_CALIBRATION_BINS",
    "TASK_BINARY",
    "TASK_MULTICLASS",
    "TASK_MULTILABEL",
    "ECGEvaluator",
    "TemperatureScaler",
    "adaptive_calibration_error",
    "apply_temperature",
    "brier_score",
    "calculate_calibration_metrics",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calibration_report",
    "confusion_matrix_analysis",
    "expected_calibration_error",
    "fit_temperature",
    "maximum_calibration_error",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_ecg_signals",
    "plot_predictions",
    "plot_roc_curve",
    "plot_training_curves",
    "probabilities_to_logits",
    "resolve_average",
    "resolve_calibration_task",
    "resolve_task_shape",
    "roc_auc_score",
    "temperature_nll",
]
