"""Post-hoc recalibration of trained ECG classifiers.

Temperature scaling divides the logits by a single positive scalar fitted to
minimise negative log-likelihood on a held-out split. It cannot change which
class is predicted, so accuracy, precision, recall, F1 and AUROC are all
invariant under it; only the probabilities move. That is the point: it buys
calibration without costing discrimination.

The temperature must be fitted on validation data and reported on test data.
Fitting on test leaks the evaluation set into the model and reports an
optimistic calibration error.
"""

import logging
import warnings
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from deepecgkit.evaluation.metrics import (
    DEFAULT_CALIBRATION_BINS,
    TASK_MULTILABEL,
    calculate_calibration_metrics,
    resolve_task_shape,
)

logger = logging.getLogger(__name__)

LOGIT_CLIP_EPS = 1e-7
DEFAULT_TEMPERATURE_BOUNDS = (0.01, 100.0)
NEUTRAL_TEMPERATURE = 1.0
BOUND_TOLERANCE = 1e-3


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax computed without exponentiating a large positive number.

    Args:
        logits: Array of shape (n_samples, n_classes)

    Returns:
        Probabilities of the same shape, each row summing to one.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Elementwise logistic function that does not overflow on large negatives.

    Args:
        logits: Array of any shape

    Returns:
        Probabilities of the same shape.
    """
    return np.exp(-np.logaddexp(0.0, -logits))


def _is_single_logit(logits: np.ndarray) -> bool:
    """Whether an array holds one logit per sample rather than one per class.

    Args:
        logits: Array of shape (n_samples,), (n_samples, 1) or (n_samples, n)

    Returns:
        True when the array carries a single logit per sample.
    """
    return logits.ndim == 1 or logits.shape[-1] == 1


def probabilities_to_logits(y_prob: np.ndarray, task: str | None = None) -> np.ndarray:
    """Recover logits from stored probabilities, for callers who kept only those.

    Prefer real logits where they are available. This reconstruction is exact
    in principle, because softmax is shift-invariant and so log(p) is a valid
    representative of the logit vector, and because log(p / (1 - p)) inverts the
    sigmoid outright. It is the float32 storage that costs accuracy: a
    probability rounds to exactly 1.0 once the logit reaches about 16.6, and
    clipping at 1e-7 caps recoverable magnitudes at about 16.1. Measured over
    4000 samples, temperature-scaled probabilities rebuilt this way match those
    from true logits to about 4e-8 for softmax and 5e-5 for sigmoid at ordinary
    confidence, but drift to roughly 5e-2 once a fifth of the rows saturate.
    Confidently wrong models are exactly the ones a calibration study cares
    about, so reconstruction is a fallback and not the default path.

    The probabilities written to predictions.txt carry six decimals, which caps
    recoverable magnitudes at about 13.8.

    Args:
        y_prob: Probabilities of shape (n_samples,) or (n_samples, n_classes)
        task: Pass TASK_MULTILABEL for per-class sigmoid outputs. Otherwise a
            one- or two-column input is read as a single logistic decision and
            anything wider as a softmax over classes.

    Returns:
        Logits, shaped (n_samples,) for logistic inputs and
        (n_samples, n_classes) for softmax and multi-label inputs.
    """
    probabilities = np.clip(np.asarray(y_prob, dtype=float), LOGIT_CLIP_EPS, 1.0 - LOGIT_CLIP_EPS)

    if task == TASK_MULTILABEL:
        return np.log(probabilities) - np.log1p(-probabilities)

    if probabilities.ndim == 1:
        return np.log(probabilities) - np.log1p(-probabilities)

    if probabilities.shape[-1] <= 2:
        positive = probabilities[:, -1]
        return np.log(positive) - np.log1p(-positive)

    return np.log(probabilities)


def apply_temperature(
    logits: np.ndarray, temperature: float, task: str | None = None
) -> np.ndarray:
    """Divide logits by a temperature and map them back to probabilities.

    The returned layout matches what ECGTrainer stores, so scaled probabilities
    are a drop-in replacement for unscaled ones: a single logit per sample comes
    back as the two columns [1 - p, p].

    Args:
        logits: Raw model outputs, (n_samples,), (n_samples, 1) or
            (n_samples, n_classes)
        temperature: Positive scalar; values above one soften predictions
        task: Pass TASK_MULTILABEL for per-class sigmoid outputs, which cannot
            be told apart from multi-class logits by shape alone

    Returns:
        Probabilities of shape (n_samples, n_classes).

    Raises:
        ValueError: If temperature is not strictly positive.
    """
    if not temperature > 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    scaled = np.asarray(logits, dtype=float) / float(temperature)

    if task == TASK_MULTILABEL:
        return _sigmoid(scaled)

    if _is_single_logit(scaled):
        positive = _sigmoid(scaled.ravel())
        return np.column_stack([1.0 - positive, positive])

    return _softmax(scaled)


def temperature_nll(
    logits: np.ndarray,
    y_true: np.ndarray,
    temperature: float = NEUTRAL_TEMPERATURE,
    task: str | None = None,
) -> float:
    """Mean negative log-likelihood of the labels under a scaled model.

    This is the quantity temperature scaling minimises. It is computed through
    logsumexp and logaddexp rather than by taking the log of a probability, so
    a saturated logit costs precision instead of producing an infinity.

    Args:
        logits: Raw model outputs
        y_true: True labels, (n_samples,) or multi-hot (n_samples, n_classes)
        temperature: Positive scalar to divide the logits by
        task: Pass TASK_MULTILABEL for per-class sigmoid outputs

    Returns:
        The mean negative log-likelihood in nats, lower being better.

    Raises:
        ValueError: If temperature is not strictly positive.
    """
    if not temperature > 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    scaled = np.asarray(logits, dtype=float) / float(temperature)
    targets = np.asarray(y_true)

    if task == TASK_MULTILABEL or _is_single_logit(scaled):
        if task != TASK_MULTILABEL:
            scaled = scaled.ravel()
            targets = targets.ravel()
        signs = 2.0 * targets.astype(float) - 1.0
        return float(np.mean(np.logaddexp(0.0, -signs * scaled)))

    labels = targets.astype(int).ravel()
    rows = np.arange(labels.shape[0])
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[rows, labels]))


def fit_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    task: str | None = None,
    bounds: tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
) -> float:
    """Fit the single temperature that minimises NLL on the given split.

    Pass the validation split. The objective is one-dimensional and smooth, so
    a bounded scalar search is used rather than a gradient optimiser: it needs
    no autograd graph, no learning rate and no iteration count, and it returns
    the same answer every run.

    A fitted temperature above one means the model was overconfident and its
    probabilities have been softened; below one means it was underconfident.

    Args:
        logits: Raw model outputs on the fitting split
        y_true: True labels for that split
        task: Pass TASK_MULTILABEL for per-class sigmoid outputs
        bounds: Search interval for the temperature

    Returns:
        The fitted temperature, or 1.0 if the search failed, in which case a
        warning is issued and the probabilities are left untouched.
    """
    logits = np.asarray(logits, dtype=float)
    if logits.shape[0] == 0:
        warnings.warn(
            "Cannot fit a temperature on an empty split; leaving probabilities unscaled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return NEUTRAL_TEMPERATURE

    def objective(temperature: float) -> float:
        return temperature_nll(logits, y_true, temperature=temperature, task=task)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")

    if not result.success or not np.isfinite(result.x):
        warnings.warn(
            "Temperature search did not converge; leaving probabilities unscaled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return NEUTRAL_TEMPERATURE

    temperature = float(result.x)
    if min(abs(temperature - bounds[0]), abs(temperature - bounds[1])) < BOUND_TOLERANCE:
        warnings.warn(
            f"Fitted temperature {temperature:.4g} sits on the search bound {bounds}, so the "
            "optimum lies outside it. This usually means the fitting split is degenerate, for "
            "example holding a single class.",
            RuntimeWarning,
            stacklevel=2,
        )
    return temperature


class TemperatureScaler:
    """Single-parameter recalibrator fitted on one split and applied to another.

    The scaler holds one scalar. Fitting it on the validation split and calling
    transform on the test split is the supported use; fitting and reporting on
    the same split measures nothing.

    Args:
        task: Task shape, one of the TASK_ constants in
            deepecgkit.evaluation.metrics. Inferred from the labels at fit time
            when omitted, which is reliable because multi-label targets are the
            only two-dimensional ones.
        bounds: Search interval for the temperature

    Examples:
        >>> scaler = TemperatureScaler().fit(val_logits, val_targets)
        >>> test_probabilities = scaler.transform(test_logits)
        >>> round(scaler.temperature, 3)
        1.42
    """

    def __init__(
        self,
        task: str | None = None,
        bounds: tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
    ):
        self.task = task
        self.bounds = bounds
        self.temperature = NEUTRAL_TEMPERATURE
        self.fitted = False

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Fit the temperature on a held-out split.

        Args:
            logits: Raw model outputs on the fitting split, which should be
                validation data
            y_true: True labels for that split

        Returns:
            This scaler, so the call can be chained.
        """
        if self.task is None:
            self.task = TASK_MULTILABEL if np.asarray(y_true).ndim > 1 else None
        self.temperature = fit_temperature(logits, y_true, task=self.task, bounds=self.bounds)
        self.fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature to a different split.

        Args:
            logits: Raw model outputs to recalibrate

        Returns:
            Probabilities in the same layout ECGTrainer stores.

        Raises:
            RuntimeError: If called before fit.
        """
        if not self.fitted:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        return apply_temperature(logits, self.temperature, task=self.task)

    def nll(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Negative log-likelihood on a split under the fitted temperature.

        Args:
            logits: Raw model outputs
            y_true: True labels

        Returns:
            The mean negative log-likelihood in nats.
        """
        return temperature_nll(logits, y_true, temperature=self.temperature, task=self.task)


def resolve_calibration_task(y_true: np.ndarray, y_prob: np.ndarray) -> str | None:
    """Task shape in the form the scaling helpers expect.

    The scaling helpers only need to know whether the outputs are per-class
    sigmoids, because that is the one case shape inspection cannot settle on
    its own. Binary and multi-class both collapse to None.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        TASK_MULTILABEL for multi-label problems, otherwise None.
    """
    return TASK_MULTILABEL if resolve_task_shape(y_true, y_prob) == TASK_MULTILABEL else None


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    fit_targets: np.ndarray | None = None,
    fit_logits: np.ndarray | None = None,
    eval_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calibration before and after temperature scaling, as one flat mapping.

    This is the shape the paper table wants: one row per run holding both the
    uncalibrated numbers and what temperature scaling recovers. The scaled
    entries are NaN unless a fitting split is supplied, so a run that had no
    validation data still produces a well-formed row.

    Pass the test split as y_true and y_prob and the validation split as
    fit_targets and fit_logits. The temperature is never fitted on the split it
    is reported on.

    Args:
        y_true: True labels for the split being reported on
        y_prob: Predicted probabilities for that split
        n_bins: Number of bins for the binned metrics
        fit_targets: True labels for the validation split
        fit_logits: Raw model outputs for the validation split
        eval_logits: Raw model outputs for the reported split. Reconstructed
            from y_prob when omitted, which costs accuracy on saturated
            predictions; see probabilities_to_logits.

    Returns:
        Mapping with "ece", "adaptive_ece", "mce", "brier", their "_temp"
        counterparts measured after scaling, the fitted "temperature", and
        "calibration_bins" recording what produced the binned values.
    """
    task = resolve_calibration_task(y_true, y_prob)
    base = calculate_calibration_metrics(y_true, y_prob, n_bins=n_bins)

    report: dict[str, Any] = {
        "ece": base.get("ece", np.nan),
        "adaptive_ece": base.get("adaptive_ece", np.nan),
        "mce": base.get("mce", np.nan),
        "brier": base.get("brier", np.nan),
        "temperature": np.nan,
        "ece_temp": np.nan,
        "adaptive_ece_temp": np.nan,
        "mce_temp": np.nan,
        "brier_temp": np.nan,
        "calibration_bins": int(n_bins),
    }

    if fit_targets is None or fit_logits is None:
        return report

    try:
        scaler = TemperatureScaler(task=task).fit(fit_logits, fit_targets)
        logits = eval_logits
        if logits is None:
            logits = probabilities_to_logits(y_prob, task=task)
        scaled = scaler.transform(logits)
        after = calculate_calibration_metrics(y_true, scaled, n_bins=n_bins)
    except Exception as error:
        logger.warning(f"Could not apply temperature scaling: {error}")
        return report

    report["temperature"] = float(scaler.temperature)
    report["ece_temp"] = after.get("ece", np.nan)
    report["adaptive_ece_temp"] = after.get("adaptive_ece", np.nan)
    report["mce_temp"] = after.get("mce", np.nan)
    report["brier_temp"] = after.get("brier", np.nan)
    return report
