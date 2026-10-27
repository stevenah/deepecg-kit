"""Test-set sampling variation, estimated from the predictions a run stored.

A test score is one draw: another sample of recordings from the same
population would have given another score. Resampling the test recordings
with replacement and rescoring each resample estimates that variation without
retraining anything, which is what a single run per architecture lacks.

Recordings, not windows, are the resampling unit, because windows cut from
one recording are not independent. Two architectures evaluated on the same
test split are compared on identical resamples (a paired bootstrap), so the
interval of their difference excludes the variation they share.

Scores are recomputed with calculate_classification_metrics, the function
behind the stored test metrics, so a resample is scored by the same
definition as the run itself.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from deepecgkit.analysis.statistics import CONFIDENCE_LEVEL
from deepecgkit.evaluation.metrics import calculate_classification_metrics

PREDICTIONS_FILENAME = "predictions.txt"
DEFAULT_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0
BOOTSTRAP_METRICS = ("f1", "auroc", "precision", "recall", "mcc", "accuracy")

#: Stored metric names mapped to the names calculate_classification_metrics uses.
_METRIC_NAMES = {"auroc": "auc"}


class Predictions(NamedTuple):
    """The test predictions of one run.

    Attributes:
        y_true: Class indices (n_samples,), or multi-hot targets (n_samples, n_labels)
        y_prob: Predicted probabilities (n_samples, n_columns)
        groups: Recording of each sample, the unit that is resampled
    """

    y_true: np.ndarray
    y_prob: np.ndarray
    groups: np.ndarray


class Interval(NamedTuple):
    """A point estimate with its percentile bootstrap interval."""

    estimate: float
    lower: float
    upper: float


def read_predictions(run_dir: str | Path) -> Predictions:
    """Read predictions.txt from a run directory.

    Single-label files hold a true_label column and one prob_ column per
    class; multi-label files hold true_, pred_ and prob_ columns per label.
    Without a record column every sample is its own recording.

    Args:
        run_dir: A run directory, or the predictions file itself

    Returns:
        The targets, probabilities and recording of every test sample.

    Raises:
        FileNotFoundError: If the run stored no predictions.
        ValueError: If the file has neither layout.
    """
    path = Path(run_dir)
    if path.is_dir():
        path = path / PREDICTIONS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No {PREDICTIONS_FILENAME} at {path}")
    frame = pd.read_csv(path, sep="\t", dtype={"record": str})
    prob_columns = [column for column in frame.columns if column.startswith("prob_")]
    if "true_label" in frame.columns:
        y_true = frame["true_label"].to_numpy(dtype=int)
    else:
        names = [column[len("prob_") :] for column in prob_columns]
        true_columns = [f"true_{name}" for name in names]
        if not names or any(column not in frame.columns for column in true_columns):
            raise ValueError(f"{path} is neither a single-label nor a multi-label predictions file")
        y_true = frame[true_columns].to_numpy(dtype=int)
    groups = (
        frame["record"].to_numpy(dtype=str)
        if "record" in frame.columns
        else np.arange(len(frame)).astype(str)
    )
    return Predictions(y_true, frame[prob_columns].to_numpy(dtype=float), groups)


def _resamples(groups: np.ndarray, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Draw recordings with replacement and return the sample indices of each draw."""
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(names))]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([members[i] for i in rng.integers(0, len(names), len(names))])
        for _ in range(n_resamples)
    ]


def _scores(predictions: Predictions, indices: np.ndarray, metrics: Sequence[str]) -> np.ndarray:
    """Score one resample with the definitions of the stored test metrics."""
    names = [_METRIC_NAMES.get(metric, metric) for metric in metrics]
    values = calculate_classification_metrics(
        predictions.y_true[indices], predictions.y_prob[indices], metrics=names
    )
    return np.array([values.get(name, np.nan) for name in names], dtype=float)


def _interval(estimate: float, draws: np.ndarray, confidence_level: float) -> Interval:
    tail = 100 * (1 - confidence_level) / 2
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return Interval(float(estimate), np.nan, np.nan)
    lower, upper = np.percentile(finite, [tail, 100 - tail])
    return Interval(float(estimate), float(lower), float(upper))


def bootstrap_intervals(
    predictions: Predictions,
    metrics: Sequence[str] = ("f1", "auroc"),
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Interval]:
    """Percentile bootstrap intervals for one run's test scores.

    Args:
        predictions: The run's test predictions
        metrics: Stored metric names, from BOOTSTRAP_METRICS
        n_resamples: Number of resamples of the test recordings
        seed: Seed of the resampling
        confidence_level: Two-sided coverage

    Returns:
        Each metric mapped to its estimate on the full test set and interval.
    """
    everything = np.arange(len(predictions.groups))
    estimates = _scores(predictions, everything, metrics)
    draws = np.stack(
        [
            _scores(predictions, indices, metrics)
            for indices in _resamples(predictions.groups, n_resamples, seed)
        ]
    )
    return {
        metric: _interval(estimates[k], draws[:, k], confidence_level)
        for k, metric in enumerate(metrics)
    }


def paired_bootstrap_difference(
    reference: Predictions,
    other: Predictions,
    metrics: Sequence[str] = ("f1", "auroc"),
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Interval]:
    """Interval of the difference other minus reference, on identical resamples.

    Args:
        reference: Predictions of the reference run
        other: Predictions of the compared run, on the same test samples in
            the same order
        metrics: Stored metric names, from BOOTSTRAP_METRICS
        n_resamples: Number of resamples of the test recordings
        seed: Seed of the resampling
        confidence_level: Two-sided coverage

    Returns:
        Each metric mapped to the difference on the full test set and its interval.

    Raises:
        ValueError: If the two runs were not evaluated on the same samples.
    """
    if not (
        np.array_equal(reference.groups, other.groups)
        and np.array_equal(reference.y_true, other.y_true)
    ):
        raise ValueError(
            "The two runs were not evaluated on the same test samples in the same order, so "
            "they cannot be compared on paired resamples."
        )
    everything = np.arange(len(reference.groups))
    estimates = _scores(other, everything, metrics) - _scores(reference, everything, metrics)
    draws = np.stack(
        [
            _scores(other, indices, metrics) - _scores(reference, indices, metrics)
            for indices in _resamples(reference.groups, n_resamples, seed)
        ]
    )
    return {
        metric: _interval(estimates[k], draws[:, k], confidence_level)
        for k, metric in enumerate(metrics)
    }
