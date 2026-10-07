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

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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

#: The module each name comes from. A name is imported when first asked for, so
#: that using the metrics alone, as the analysis commands do, does not load PyTorch
#: and matplotlib along with them.
_EXPORTS = {
    "DEFAULT_CALIBRATION_BINS": "deepecgkit.evaluation.metrics",
    "DEFAULT_PROBE_METRIC": "deepecgkit.evaluation.representation",
    "ECGEvaluator": "deepecgkit.evaluation.evaluator",
    "PROBE_DATASETS": "deepecgkit.evaluation.representation",
    "PROBE_INPUT_CHANNELS": "deepecgkit.evaluation.representation",
    "PROBE_METRICS": "deepecgkit.evaluation.representation",
    "PROBE_SAMPLING_RATE": "deepecgkit.evaluation.representation",
    "PROTOCOL_DATASET_KWARGS": "deepecgkit.evaluation.representation",
    "ProbeResult": "deepecgkit.evaluation.representation",
    "RANDOM_INIT": "deepecgkit.evaluation.representation",
    "TASK_BINARY": "deepecgkit.evaluation.metrics",
    "TASK_MULTICLASS": "deepecgkit.evaluation.metrics",
    "TASK_MULTILABEL": "deepecgkit.evaluation.metrics",
    "TemperatureScaler": "deepecgkit.evaluation.calibration",
    "TransferMatrix": "deepecgkit.evaluation.representation",
    "adaptive_calibration_error": "deepecgkit.evaluation.metrics",
    "apply_temperature": "deepecgkit.evaluation.calibration",
    "brier_score": "deepecgkit.evaluation.metrics",
    "build_probe": "deepecgkit.evaluation.representation",
    "calculate_calibration_metrics": "deepecgkit.evaluation.metrics",
    "calculate_classification_metrics": "deepecgkit.evaluation.metrics",
    "calculate_regression_metrics": "deepecgkit.evaluation.metrics",
    "calibration_report": "deepecgkit.evaluation.calibration",
    "confusion_matrix_analysis": "deepecgkit.evaluation.metrics",
    "cross_dataset_transfer": "deepecgkit.evaluation.representation",
    "embed_2d": "deepecgkit.evaluation.representation",
    "expected_calibration_error": "deepecgkit.evaluation.metrics",
    "fit_temperature": "deepecgkit.evaluation.calibration",
    "linear_probe": "deepecgkit.evaluation.representation",
    "maximum_calibration_error": "deepecgkit.evaluation.metrics",
    "plot_calibration_curve": "deepecgkit.evaluation.visualization",
    "plot_confusion_matrix": "deepecgkit.evaluation.visualization",
    "plot_ecg_signals": "deepecgkit.evaluation.visualization",
    "plot_embedding": "deepecgkit.evaluation.visualization",
    "plot_predictions": "deepecgkit.evaluation.visualization",
    "plot_roc_curve": "deepecgkit.evaluation.visualization",
    "plot_training_curves": "deepecgkit.evaluation.visualization",
    "plot_transfer_matrix": "deepecgkit.evaluation.visualization",
    "probabilities_to_logits": "deepecgkit.evaluation.calibration",
    "probe_encoder": "deepecgkit.evaluation.representation",
    "protocol_data_module": "deepecgkit.evaluation.representation",
    "resolve_average": "deepecgkit.evaluation.metrics",
    "resolve_calibration_task": "deepecgkit.evaluation.calibration",
    "resolve_task_shape": "deepecgkit.evaluation.metrics",
    "roc_auc_score": "deepecgkit.evaluation.metrics",
    "temperature_nll": "deepecgkit.evaluation.calibration",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
