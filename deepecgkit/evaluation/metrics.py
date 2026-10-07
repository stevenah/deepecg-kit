"""
Evaluation metrics for ECG models.
"""

import logging
import warnings
from typing import Any

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

logger = logging.getLogger(__name__)

BINARY_THRESHOLD = 0.5
MULTICLASS_THRESHOLD = 2

#: Names under which accuracy can be asked for. They differ for multi-hot
#: targets only: per label ("accuracy", "label_accuracy") or per sample.
ACCURACY_METRICS = ("accuracy", "label_accuracy", "exact_match_accuracy")

TASK_BINARY = "binary"
TASK_MULTICLASS = "multiclass"
TASK_MULTILABEL = "multilabel"

BINNING_UNIFORM = "uniform"
BINNING_QUANTILE = "quantile"

DEFAULT_CALIBRATION_BINS = 15
CALIBRATION_METRIC_NAMES = ("ece", "adaptive_ece", "mce", "brier")

#: Keys emitted by calibration_report: the four binned metrics, the fitted
#: temperature, and the rescaled variants. The evaluator's report rows and the
#: analysis pipeline's expected columns both derive from this.
CALIBRATION_REPORT_KEYS = (
    "ece",
    "adaptive_ece",
    "mce",
    "brier",
    "temperature",
    "ece_temp",
    "adaptive_ece_temp",
    "mce_temp",
    "brier_temp",
)


def resolve_average(y_true: np.ndarray, average: str = "auto") -> str:
    """Resolve the averaging mode applied to precision, recall and F1.

    Precision, recall and F1 are not comparable across averaging modes, so the
    mode in force must be recorded alongside the values rather than inferred.
    Under "auto" a two-class problem yields positive-class ("binary") values,
    not macro-averaged ones. Multi-hot targets are macro-averaged over their
    labels.

    Args:
        y_true: True labels, used only to count classes under "auto"
        average: One of "auto", "binary", "macro", "micro", "weighted"

    Returns:
        The concrete averaging mode to pass to scikit-learn.
    """
    if average != "auto":
        return average
    if np.ndim(y_true) > 1:
        return "macro"
    return "binary" if len(np.unique(y_true)) <= MULTICLASS_THRESHOLD else "macro"


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list[str] | None = None,
    average: str = "auto",
) -> dict[str, float]:
    """
    Calculate classification metrics.

    Multi-hot targets of shape (n_samples, n_labels) make the problem
    multi-label, and the metrics are defined as `deepecg evaluate` reports
    them: predictions are probabilities thresholded at 0.5 (or 0/1 labels),
    "accuracy" is the share of individual labels predicted correctly, as
    ECGTrainer reports it, "exact_match_accuracy" the share of samples with
    every label right, and precision, recall, F1 and AUC are macro averages
    over the labels. MCC is not defined for them.

    Args:
        y_true: True labels, or multi-hot targets
        y_pred: Predicted labels or probabilities
        metrics: List of metrics to compute, from accuracy, label_accuracy,
            exact_match_accuracy, precision, recall, f1, auc and mcc. The two
            longer accuracy names equal accuracy for single-label targets.
        average: Averaging for precision/recall/f1. "auto" selects positive-class
            values for two-class problems and macro averaging otherwise. Pass an
            explicit mode to make the choice independent of the label set. Use
            resolve_average to obtain the mode actually applied.

    Returns:
        Dictionary of computed metrics
    """
    multilabel = np.ndim(y_true) > 1
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        metrics.append("exact_match_accuracy" if multilabel else "mcc")

    results = {}

    if multilabel:
        y_true = np.asarray(y_true).astype(int)
        is_score = np.issubdtype(y_pred.dtype, np.floating)
        y_pred_labels = (y_pred > BINARY_THRESHOLD).astype(int) if is_score else y_pred
        y_scores = y_pred
    elif y_pred.ndim > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_scores = y_pred
    elif (
        np.issubdtype(y_pred.dtype, np.floating)
        and np.max(y_pred) <= 1.0
        and len(np.unique(y_true)) <= MULTICLASS_THRESHOLD
    ):
        y_pred_labels = (y_pred > BINARY_THRESHOLD).astype(int)
        y_scores = y_pred
    else:
        y_pred_labels = y_pred.astype(int)
        y_scores = y_pred

    n_classes = len(np.unique(y_true))
    avg = resolve_average(y_true, average)

    def _compute_metric(metric: str) -> Any:
        result = np.nan
        if metric in ACCURACY_METRICS:
            if multilabel:
                correct = y_pred_labels == y_true
                exact = metric == "exact_match_accuracy"
                result = correct.all(axis=1).mean() if exact else correct.mean()
            else:
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
            if multilabel:
                result = roc_auc_score(y_true, y_scores, average="macro")
            elif n_classes > MULTICLASS_THRESHOLD:
                result = roc_auc_score(y_true, y_scores, multi_class="ovr")
            else:
                # Binary AUC expects 1D scores (positive class probability)
                scores = y_scores[:, 1] if y_scores.ndim > 1 else y_scores
                result = roc_auc_score(y_true, scores)
        elif metric == "mcc":
            if multilabel:
                raise ValueError("MCC is not defined for multi-label targets")
            result = matthews_corrcoef(y_true, y_pred_labels)
        return result

    for metric in metrics:
        try:
            results[metric] = _compute_metric(metric)
        except Exception as e:
            logger.warning(f"Could not compute {metric}: {e}")
            results[metric] = np.nan

    return results


def resolve_task_shape(y_true: np.ndarray, y_prob: np.ndarray, task: str | None = None) -> str:
    """Resolve which of the three prediction shapes produced these arrays.

    ECGTrainer emits exactly three shapes and the correct calibration
    definition differs between them, so the shape is resolved once and carried
    down rather than re-sniffed at each step. A two-column probability matrix
    is reported as "binary" whether it came from a single sigmoid logit stacked
    into [1 - p, p] or from a two-class softmax: the two are indistinguishable
    from the arrays alone and yield identical top-label statistics anyway.

    Args:
        y_true: True labels, (n_samples,) for single-label problems or
            (n_samples, n_classes) multi-hot for multi-label ones
        y_prob: Predicted probabilities, (n_samples,) or (n_samples, n_classes)
        task: Explicit override, one of "binary", "multiclass", "multilabel"

    Returns:
        One of TASK_BINARY, TASK_MULTICLASS or TASK_MULTILABEL.

    Raises:
        ValueError: If task is supplied but is not a recognised shape.
    """
    if task is not None:
        if task not in (TASK_BINARY, TASK_MULTICLASS, TASK_MULTILABEL):
            raise ValueError(f"Unknown task shape: {task}")
        return task

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        return TASK_MULTILABEL
    if y_prob.ndim == 1 or y_prob.shape[-1] <= MULTICLASS_THRESHOLD:
        return TASK_BINARY
    return TASK_MULTICLASS


def _as_probability_matrix(y_prob: np.ndarray) -> np.ndarray:
    """Widen bare positive-class probabilities into a two-column matrix.

    Args:
        y_prob: Probabilities of shape (n_samples,) or (n_samples, n_classes)

    Returns:
        A float array of shape (n_samples, n_classes).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        return np.column_stack([1.0 - y_prob, y_prob])
    return y_prob


def _binned_gaps(
    confidence: np.ndarray, correctness: np.ndarray, n_bins: int, strategy: str
) -> tuple:
    """Bin populations and their absolute accuracy-minus-confidence gaps.

    Equal-mass bins are formed by splitting the confidence-sorted order rather
    than by cutting at empirical quantiles. Quantile edges collapse onto each
    other whenever confidences tie, which silently yields fewer bins than were
    asked for and quietly changes the reported error.

    Args:
        confidence: Predicted probability attached to each decision
        correctness: 1.0 where that decision was right, 0.0 otherwise
        n_bins: Number of bins
        strategy: BINNING_UNIFORM for equal width, BINNING_QUANTILE for
            equal mass

    Returns:
        Tuple of (weights, gaps) over non-empty bins, both 1D float arrays.
    """
    n_samples = confidence.shape[0]
    if n_samples == 0 or n_bins < 1:
        return np.empty(0), np.empty(0)

    if strategy == BINNING_QUANTILE:
        order = np.argsort(confidence, kind="stable")
        groups = np.array_split(order, min(n_bins, n_samples))
    else:
        index = np.clip((confidence * n_bins).astype(int), 0, n_bins - 1)
        groups = [np.flatnonzero(index == edge) for edge in range(n_bins)]

    weights = []
    gaps = []
    for group in groups:
        if group.size == 0:
            continue
        weights.append(group.size / n_samples)
        gaps.append(abs(float(correctness[group].mean()) - float(confidence[group].mean())))
    return np.asarray(weights, dtype=float), np.asarray(gaps, dtype=float)


def _calibration_error(
    confidence: np.ndarray, correctness: np.ndarray, n_bins: int, strategy: str
) -> tuple:
    """Weighted and maximum calibration error for one binary decision stream.

    Args:
        confidence: Predicted probability attached to each decision
        correctness: 1.0 where that decision was right, 0.0 otherwise
        n_bins: Number of bins
        strategy: BINNING_UNIFORM or BINNING_QUANTILE

    Returns:
        Tuple of (expected error, maximum error). Both are NaN when no bin is
        populated. Empty bins contribute nothing rather than contributing a
        zero gap, which would otherwise flatter a model that abstains from a
        region of the probability range.
    """
    weights, gaps = _binned_gaps(confidence, correctness, n_bins, strategy)
    if gaps.size == 0:
        return np.nan, np.nan
    return float(np.sum(weights * gaps)), float(np.max(gaps))


def _decision_streams(y_true: np.ndarray, y_prob: np.ndarray, shape: str) -> list[tuple]:
    """Split predictions into the binary decision streams calibration scores.

    For multi-class and binary problems this is the single top-label stream of
    Guo et al. (2017): confidence is the largest predicted probability and the
    decision is right when the argmax matches the label. For multi-label
    problems it is one stream per class, each covering the full [0, 1] range,
    which is why multi-label results are averaged over classes rather than
    pooled: pooling would let a well-calibrated common class hide a badly
    calibrated rare one.

    Args:
        y_true: True labels
        y_prob: Predicted probability matrix
        shape: Task shape from resolve_task_shape

    Returns:
        List of (confidence, correctness) pairs, one per stream.
    """
    if shape == TASK_MULTILABEL:
        targets = np.asarray(y_true, dtype=float)
        return [(y_prob[:, c], targets[:, c]) for c in range(y_prob.shape[1])]

    labels = np.asarray(y_true).astype(int).ravel()
    confidence = y_prob.max(axis=1)
    correctness = (np.argmax(y_prob, axis=1) == labels).astype(float)
    return [(confidence, correctness)]


def _calibration_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    strategy: str,
    task: str | None,
) -> dict[str, Any]:
    """Expected and maximum calibration error, with per-class detail.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: BINNING_UNIFORM or BINNING_QUANTILE
        task: Explicit task shape override

    Returns:
        Mapping with "ece", "mce", "per_class_ece" and "per_class_mce". The
        per-class entries are None outside multi-label problems.
    """
    shape = resolve_task_shape(y_true, y_prob, task)
    matrix = _as_probability_matrix(y_prob)
    streams = _decision_streams(y_true, matrix, shape)

    errors = [_calibration_error(conf, correct, n_bins, strategy) for conf, correct in streams]
    per_class_ece = np.asarray([e for e, _ in errors], dtype=float)
    per_class_mce = np.asarray([m for _, m in errors], dtype=float)

    if shape != TASK_MULTILABEL:
        return {
            "ece": float(per_class_ece[0]),
            "mce": float(per_class_mce[0]),
            "per_class_ece": None,
            "per_class_mce": None,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        macro_ece = float(np.nanmean(per_class_ece)) if per_class_ece.size else np.nan
        macro_mce = float(np.nanmax(per_class_mce)) if per_class_mce.size else np.nan
    return {
        "ece": macro_ece,
        "mce": macro_mce,
        "per_class_ece": per_class_ece,
        "per_class_mce": per_class_mce,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    task: str | None = None,
) -> float:
    """Expected calibration error over equal-width probability bins.

    The gap between confidence and accuracy is averaged over bins weighted by
    how many samples fall in each. ECE is sensitive to n_bins and the two
    numbers are not comparable across different bin counts, so n_bins must
    travel with any reported value. The default of 15 follows Guo et al.
    (2017), which keeps values comparable with the temperature-scaling
    literature; use adaptive_calibration_error for the companion number that
    does not depend on where the equal-width edges happen to fall.

    Multi-label inputs are scored per class and averaged, never flattened.

    Args:
        y_true: True labels, (n_samples,) or multi-hot (n_samples, n_classes)
        y_prob: Predicted probabilities, (n_samples,) or (n_samples, n_classes)
        n_bins: Number of equal-width bins spanning [0, 1]
        task: Explicit task shape, otherwise inferred from the array shapes

    Returns:
        The error in [0, 1], or NaN when no bin is populated.
    """
    return _calibration_errors(y_true, y_prob, n_bins, BINNING_UNIFORM, task)["ece"]


def adaptive_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    task: str | None = None,
) -> float:
    """Expected calibration error over equal-mass probability bins.

    Equal-width bins are usually near-empty at the low-confidence end and
    crowded at the top, so an equal-width bin holding a handful of samples
    estimates its gap from almost nothing. Equal-mass bins put the same number
    of samples in every bin, so no bin's gap rests on a tiny sample.

    Report this beside the equal-width number rather than instead of it. The
    two are not reliably ordered and neither dominates: where they disagree,
    the disagreement is itself evidence that the binning is doing the work
    rather than the model.

    Args:
        y_true: True labels, (n_samples,) or multi-hot (n_samples, n_classes)
        y_prob: Predicted probabilities, (n_samples,) or (n_samples, n_classes)
        n_bins: Number of equal-mass bins
        task: Explicit task shape, otherwise inferred from the array shapes

    Returns:
        The error in [0, 1], or NaN when there are no samples.
    """
    return _calibration_errors(y_true, y_prob, n_bins, BINNING_QUANTILE, task)["ece"]


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    task: str | None = None,
) -> float:
    """Largest confidence-accuracy gap across equal-width bins.

    MCE reports the worst bin rather than the average one, so it answers how
    far the model can be trusted at its least reliable confidence level. It is
    noisier than ECE because a sparsely populated bin can set it, and it is
    never smaller than the ECE computed over the same bins. Multi-label inputs
    take the worst class, since averaging worst cases would understate them.

    Args:
        y_true: True labels, (n_samples,) or multi-hot (n_samples, n_classes)
        y_prob: Predicted probabilities, (n_samples,) or (n_samples, n_classes)
        n_bins: Number of equal-width bins spanning [0, 1]
        task: Explicit task shape, otherwise inferred from the array shapes

    Returns:
        The error in [0, 1], or NaN when no bin is populated.
    """
    return _calibration_errors(y_true, y_prob, n_bins, BINNING_UNIFORM, task)["mce"]


def brier_score(y_true: np.ndarray, y_prob: np.ndarray, task: str | None = None) -> float:
    """Mean squared error of the predicted probability vector.

    Two conventions are in use and this function applies the one that is
    standard for each shape, so the values are not comparable between shapes.
    Multi-class and binary problems use the original sum-over-classes form,
    which ranges over [0, 2]. Multi-label problems average the per-class binary
    Brier score, which ranges over [0, 1]. Tables are emitted per dataset, so a
    single table never mixes the two, but a figure spanning datasets would.

    Unlike ECE this is a proper scoring rule: it rewards sharpness as well as
    calibration, so a model can improve its ECE while its Brier score worsens.

    Args:
        y_true: True labels, (n_samples,) or multi-hot (n_samples, n_classes)
        y_prob: Predicted probabilities, (n_samples,) or (n_samples, n_classes)
        task: Explicit task shape, otherwise inferred from the array shapes

    Returns:
        The score, lower being better, or NaN when there are no samples.
    """
    shape = resolve_task_shape(y_true, y_prob, task)
    matrix = _as_probability_matrix(y_prob)
    if matrix.shape[0] == 0:
        return np.nan

    if shape == TASK_MULTILABEL:
        targets = np.asarray(y_true, dtype=float)
        return float(((matrix - targets) ** 2).mean(axis=0).mean())

    labels = np.asarray(y_true).astype(int).ravel()
    onehot = np.zeros_like(matrix)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return float(((matrix - onehot) ** 2).sum(axis=1).mean())


def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    task: str | None = None,
) -> dict[str, Any]:
    """Calculate every calibration metric in one pass.

    The bin count and resolved task shape are returned alongside the values
    because neither ECE nor MCE means anything without them, and a table that
    records only the number invites comparison between runs binned differently.

    Args:
        y_true: True labels, (n_samples,) or multi-hot (n_samples, n_classes)
        y_prob: Predicted probabilities, (n_samples,) or (n_samples, n_classes)
        n_bins: Number of bins for the binned metrics
        task: Explicit task shape, otherwise inferred from the array shapes

    Returns:
        Dictionary with "ece", "adaptive_ece", "mce" and "brier", the context
        keys "n_bins" and "task", and for multi-label inputs the per-class
        arrays "per_class_ece", "per_class_mce" and "per_class_brier".
    """
    results: dict[str, Any] = {"n_bins": int(n_bins), "task": None}
    try:
        results["task"] = resolve_task_shape(y_true, y_prob, task)
    except ValueError as error:
        logger.warning(f"Could not resolve task shape: {error}")
        for name in CALIBRATION_METRIC_NAMES:
            results[name] = np.nan
        return results

    try:
        uniform = _calibration_errors(y_true, y_prob, n_bins, BINNING_UNIFORM, results["task"])
        results["ece"] = uniform["ece"]
        results["mce"] = uniform["mce"]
        if uniform["per_class_ece"] is not None:
            results["per_class_ece"] = uniform["per_class_ece"]
            results["per_class_mce"] = uniform["per_class_mce"]
    except Exception as error:
        logger.warning(f"Could not compute ece: {error}")
        results["ece"] = np.nan
        results["mce"] = np.nan

    try:
        results["adaptive_ece"] = adaptive_calibration_error(
            y_true, y_prob, n_bins=n_bins, task=results["task"]
        )
    except Exception as error:
        logger.warning(f"Could not compute adaptive_ece: {error}")
        results["adaptive_ece"] = np.nan

    try:
        results["brier"] = brier_score(y_true, y_prob, task=results["task"])
        if results["task"] == TASK_MULTILABEL:
            matrix = _as_probability_matrix(y_prob)
            targets = np.asarray(y_true, dtype=float)
            results["per_class_brier"] = ((matrix - targets) ** 2).mean(axis=0)
    except Exception as error:
        logger.warning(f"Could not compute brier: {error}")
        results["brier"] = np.nan

    return results


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str] | None = None
) -> dict[str, float]:
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
            logger.warning(f"Could not compute {metric}: {e}")
            results[metric] = np.nan

    return results


def confusion_matrix_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
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
