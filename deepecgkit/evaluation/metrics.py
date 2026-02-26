"""
Evaluation metrics for ECG models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

BINARY_THRESHOLD = 0.5
MULTICLASS_THRESHOLD = 2


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        metrics: List of metrics to compute

    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]

    results = {}

    if y_pred.ndim > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_scores = y_pred
    elif (
        y_pred.dtype == float
        and np.max(y_pred) <= 1.0
        and len(np.unique(y_true)) <= MULTICLASS_THRESHOLD
    ):
        y_pred_labels = (y_pred > BINARY_THRESHOLD).astype(int)
        y_scores = y_pred
    else:
        y_pred_labels = y_pred.astype(int)
        y_scores = y_pred

    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes <= MULTICLASS_THRESHOLD else "macro"

    def _compute_metric(metric):
        result = np.nan
        if metric == "accuracy":
            result = accuracy_score(y_true, y_pred_labels)
        elif metric in ["precision", "recall", "f1"]:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred_labels, average=avg, zero_division=0
            )
            if metric == "precision":
                result = precision
            elif metric == "recall":
                result = recall
            elif metric == "f1":
                result = f1
        elif metric == "auc":
            if n_classes > MULTICLASS_THRESHOLD:
                result = roc_auc_score(y_true, y_scores, multi_class="ovr")
            else:
                result = roc_auc_score(y_true, y_scores)
        elif metric == "mcc":
            result = matthews_corrcoef(y_true, y_pred_labels)
        return result

    for metric in metrics:
        try:
            results[metric] = _compute_metric(metric)
        except Exception as e:
            print(f"Warning: Could not compute {metric}: {e}")
            results[metric] = np.nan

    return results


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metrics to compute

    Returns:
        Dictionary of computed metrics
    """
    if metrics is None:
        metrics = ["mse", "mae", "r2"]

    results = {}

    for metric in metrics:
        try:
            if metric in ["mse", "mean_squared_error"]:
                results[metric] = mean_squared_error(y_true, y_pred)
            elif metric in ["mae", "mean_absolute_error"]:
                results[metric] = mean_absolute_error(y_true, y_pred)
            elif metric in ["r2", "r2_score"]:
                results[metric] = r2_score(y_true, y_pred)
            elif metric == "rmse":
                results[metric] = np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception as e:
            print(f"Warning: Could not compute {metric}: {e}")
            results[metric] = np.nan

    return results


def confusion_matrix_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Detailed confusion matrix analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    cm = confusion_matrix(y_true, y_pred)

    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)

    for i in range(n_classes):
        if cm[:, i].sum() > 0:
            precision[i] = cm[i, i] / cm[:, i].sum()
        if cm[i, :].sum() > 0:
            recall[i] = cm[i, i] / cm[i, :].sum()
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return {
        "confusion_matrix": cm,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "macro_precision": np.mean(precision),
        "macro_recall": np.mean(recall),
        "macro_f1": np.mean(f1),
    }
