"""
Model evaluation and metrics module for DeepECG-Kit.

This module provides comprehensive evaluation functionality including:
- Classification metrics (accuracy, precision, recall, F1, AUC)
- Regression metrics (MSE, MAE, R²)
- ECG-specific metrics
- Representation quality (linear probing, cross-dataset transfer)
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
from deepecgkit.evaluation.representation import (
    DEFAULT_PROBE_METRIC,
    PROBE_DATASETS,
    PROBE_INPUT_CHANNELS,
    PROBE_METRICS,
    PROBE_SAMPLING_RATE,
    PROTOCOL_DATASET_KWARGS,
    RANDOM_INIT,
    ProbeResult,
    TransferMatrix,
    build_probe,
    cross_dataset_transfer,
    embed_2d,
    linear_probe,
    probe_encoder,
    protocol_data_module,
)
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_ecg_signals,
    plot_embedding,
    plot_predictions,
    plot_roc_curve,
    plot_training_curves,
    plot_transfer_matrix,
)

__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_PROBE_METRIC",
    "PROBE_DATASETS",
    "PROBE_INPUT_CHANNELS",
    "PROBE_METRICS",
    "PROBE_SAMPLING_RATE",
    "PROTOCOL_DATASET_KWARGS",
    "RANDOM_INIT",
    "TASK_BINARY",
    "TASK_MULTICLASS",
    "TASK_MULTILABEL",
    "ECGEvaluator",
    "ProbeResult",
    "TemperatureScaler",
    "TransferMatrix",
    "adaptive_calibration_error",
    "apply_temperature",
    "brier_score",
    "build_probe",
    "calculate_calibration_metrics",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calibration_report",
    "confusion_matrix_analysis",
    "cross_dataset_transfer",
    "embed_2d",
    "expected_calibration_error",
    "fit_temperature",
    "linear_probe",
    "maximum_calibration_error",
    "plot_calibration_curve",
    "plot_confusion_matrix",
    "plot_ecg_signals",
    "plot_embedding",
    "plot_predictions",
    "plot_roc_curve",
    "plot_training_curves",
    "plot_transfer_matrix",
    "probabilities_to_logits",
    "probe_encoder",
    "protocol_data_module",
    "resolve_average",
    "resolve_calibration_task",
    "resolve_task_shape",
    "roc_auc_score",
    "temperature_nll",
]
