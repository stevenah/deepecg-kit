"""Statistics for summarising and comparing metrics across repeated training runs.

The functions here operate on plain sequences of per-seed scalars. They are
deliberately separate from :mod:`deepecgkit.evaluation.evaluator`, whose
``bootstrap_evaluate`` resamples individual test samples and recomputes metrics
from predictions; that estimator cannot consume a handful of finished run
scores. The output key names mirror it so the two read alike.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats

CONFIDENCE_LEVEL = 0.95
MIN_SEEDS_FOR_INTERVAL = 3
MIN_SEEDS_FOR_CLAIM = 5
MIN_COLLAPSE_GAP = 0.10
NO_CORRECTION = "none"
NADEAU_BENGIO = "nadeau-bengio"


def _finite(values: Sequence[float]) -> np.ndarray:
    """Drop non-finite entries and return the remainder as a float array.

    Args:
        values: Raw per-seed values, possibly containing NaN or infinities.

    Returns:
        A one-dimensional float array holding only the finite entries.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _check_confidence_level(confidence_level: float) -> None:
    """Reject confidence levels outside the open unit interval.

    Args:
        confidence_level: Two-sided coverage.

    Raises:
        ValueError: If the level is not strictly between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be strictly between 0 and 1, got {confidence_level}"
        )


def summarize_scores(
    values: Sequence[float], confidence_level: float = CONFIDENCE_LEVEL
) -> dict[str, float]:
    """Summarise one metric across seeds with a two-sided Student-t interval.

    Non-finite entries are dropped first, so "n" counts usable values rather
    than attempted runs. The t distribution is used because with three to ten
    seeds the variance is estimated from the same points; a percentile
    bootstrap over so few scalars cannot place an endpoint outside their
    observed range and therefore understates the interval.

    Args:
        values: Per-seed values of a single metric.
        confidence_level: Two-sided coverage.

    Returns:
        Mapping with "mean", "std" (sample standard deviation, ddof=1),
        "sem", "lower_ci", "upper_ci" and "n". With a single value the
        dispersion and interval keys are NaN rather than zero. With no values
        every key but "n" is NaN.

    Raises:
        ValueError: If confidence_level is not strictly between 0 and 1.
    """
    _check_confidence_level(confidence_level)
    finite = _finite(values)
    n = int(finite.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "n": 0,
        }

    mean = float(np.mean(finite))
    if n == 1:
        return {
            "mean": mean,
            "std": float("nan"),
            "sem": float("nan"),
            "lower_ci": float("nan"),
            "upper_ci": float("nan"),
            "n": 1,
        }

    std = float(np.std(finite, ddof=1))
    sem = std / math.sqrt(n)
    half_width = float(stats.t.ppf(0.5 + confidence_level / 2.0, n - 1)) * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "lower_ci": mean - half_width,
        "upper_ci": mean + half_width,
        "n": n,
    }


def nadeau_bengio_factor(n: int, n_train: int | None, n_test: int | None) -> float:
    """Variance inflation factor for repeated random subsampling.

    When every seed redraws the train/test partition of one fixed corpus the
    per-seed differences share training data and are positively correlated, so
    the ordinary standard error understates the spread. Nadeau and Bengio
    (2003), Machine Learning 52(3), replace ``1 / n`` with
    ``1 / n + n_test / n_train``.

    Args:
        n: Number of usable seeds.
        n_train: Training set size, or None when it was not recorded.
        n_test: Test set size, or None when it was not recorded.

    Returns:
        The multiplier applied to the sample standard deviation to obtain the
        corrected standard error. Falls back to the uncorrected
        ``1 / sqrt(n)`` when the split sizes are unavailable.
    """
    if n <= 0:
        return float("nan")
    base = 1.0 / n
    if not n_train or not n_test or n_train <= 0 or n_test <= 0:
        return math.sqrt(base)
    return math.sqrt(base + float(n_test) / float(n_train))


def paired_summary(
    differences: Sequence[float],
    confidence_level: float = CONFIDENCE_LEVEL,
    correction: str = NADEAU_BENGIO,
    n_train: int | None = None,
    n_test: int | None = None,
) -> dict[str, float]:
    """Summarise paired per-seed differences and test them against zero.

    Args:
        differences: One difference per shared seed.
        confidence_level: Two-sided coverage.
        correction: Either "nadeau-bengio" or "none".
        n_train: Mean training set size across the contributing runs.
        n_test: Mean test set size across the contributing runs.

    Returns:
        Mapping with "mean", "std", "sem", "lower_ci", "upper_ci", "t_stat",
        "p_value", "n", "n_wins" and "excludes_zero". A degenerate sample with
        zero variance yields NaN for the interval and the p-value rather than a
        zero-width interval, because zero observed spread is not infinite
        precision.

    Raises:
        ValueError: If correction is not a recognised name.
    """
    if correction not in (NO_CORRECTION, NADEAU_BENGIO):
        raise ValueError(f"Unknown correction: {correction}")
    _check_confidence_level(confidence_level)

    finite = _finite(differences)
    n = int(finite.size)
    result: dict[str, float] = {
        "mean": float("nan"),
        "std": float("nan"),
        "sem": float("nan"),
        "lower_ci": float("nan"),
        "upper_ci": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
        "n": n,
        "n_wins": 0,
        "excludes_zero": False,
    }
    if n == 0:
        return result

    result["mean"] = float(np.mean(finite))
    result["n_wins"] = int(np.sum(finite > 0))
    if n < 2:
        return result

    std = float(np.std(finite, ddof=1))
    result["std"] = std
    if std == 0.0:
        return result

    factor = (
        nadeau_bengio_factor(n, n_train, n_test)
        if correction == NADEAU_BENGIO
        else (1.0 / math.sqrt(n))
    )
    sem = std * factor
    result["sem"] = sem
    half_width = float(stats.t.ppf(0.5 + confidence_level / 2.0, n - 1)) * sem
    lower = result["mean"] - half_width
    upper = result["mean"] + half_width
    t_stat = result["mean"] / sem
    result["lower_ci"] = lower
    result["upper_ci"] = upper
    result["t_stat"] = t_stat
    result["p_value"] = float(2.0 * stats.t.sf(abs(t_stat), n - 1))
    result["excludes_zero"] = bool(lower > 0.0 or upper < 0.0)
    return result


def kendall_tau(ranks_a: Mapping[str, float], ranks_b: Mapping[str, float]) -> float:
    """Kendall's tau-b between two rankings over the same labels.

    Args:
        ranks_a: Label to rank for the first ranking.
        ranks_b: Label to rank for the second ranking.

    Returns:
        The tau-b coefficient. No p-value is returned: with the four or five
        architectures compared here the smallest attainable two-sided p-value
        is ``2 / len(labels)!``, so no single coefficient can reach
        conventional significance and reporting one would invite over-reading.

    Raises:
        ValueError: If the label sets differ or fewer than three labels are given.
    """
    labels = sorted(ranks_a)
    if labels != sorted(ranks_b):
        raise ValueError("Both rankings must cover the same labels")
    if len(labels) < 3:
        raise ValueError(f"Kendall's tau needs at least 3 labels, got {len(labels)}")

    first = [float(ranks_a[label]) for label in labels]
    second = [float(ranks_b[label]) for label in labels]
    tau, _ = stats.kendalltau(first, second, variant="b")
    return float(tau)


def min_attainable_kendall_p(n_labels: int) -> float:
    """Smallest two-sided p-value a tie-free Kendall's tau can attain.

    Args:
        n_labels: Number of ranked labels.

    Returns:
        ``2 / n_labels!`` under the exact permutation null, capped at 1.0. At
        four labels this is 0.083, so significance is unattainable however
        perfectly the rankings agree.

    Raises:
        ValueError: If fewer than three labels are given.
    """
    if n_labels < 3:
        raise ValueError(f"Kendall's tau needs at least 3 labels, got {n_labels}")
    return min(1.0, 2.0 / float(math.factorial(n_labels)))


def _adjust(pvalues: Sequence[float], scale: float, step_up: bool) -> list[float]:
    """Shared machinery for the step-down and step-up multiplicity adjusters.

    Args:
        pvalues: Raw p-values, possibly containing NaN.
        scale: Constant multiplier applied to the per-rank factor.
        step_up: True for a Benjamini style step-up, False for Holm step-down.

    Returns:
        Adjusted p-values aligned with the input order; NaN entries stay NaN.
    """
    raw = np.asarray(list(pvalues), dtype=float)
    usable = np.flatnonzero(np.isfinite(raw))
    adjusted = np.full(raw.shape, np.nan, dtype=float)
    m = usable.size
    if m == 0:
        return [float(value) for value in adjusted]

    order = usable[np.argsort(raw[usable], kind="mergesort")]
    if step_up:
        order = order[::-1]
        running = 1.0
        for position, index in enumerate(order):
            rank = m - position
            candidate = min(1.0, raw[index] * scale * m / rank)
            running = min(running, candidate)
            adjusted[index] = running
    else:
        running = 0.0
        for position, index in enumerate(order):
            candidate = min(1.0, raw[index] * scale * (m - position))
            running = max(running, candidate)
            adjusted[index] = running
    return [float(value) for value in adjusted]


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down familywise adjustment.

    Args:
        pvalues: Raw p-values, possibly containing NaN.

    Returns:
        Adjusted p-values in the input order.
    """
    return _adjust(pvalues, scale=1.0, step_up=False)


def benjamini_yekutieli_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Yekutieli false discovery rate adjustment.

    Yekutieli's correction is used rather than plain Benjamini-Hochberg
    because the comparisons here share training runs, models and datasets, so
    the positive dependence that Benjamini-Hochberg assumes is not
    established. The price is a factor of the harmonic number of the family
    size.

    Args:
        pvalues: Raw p-values, possibly containing NaN.

    Returns:
        Adjusted p-values in the input order.
    """
    raw = [value for value in pvalues if np.isfinite(value)]
    m = len(raw)
    if m == 0:
        return [float("nan") for _ in pvalues]
    harmonic = float(sum(1.0 / (i + 1) for i in range(m)))
    return _adjust(pvalues, scale=harmonic, step_up=True)


def is_bimodal(
    values: Sequence[float],
    baseline: float | None = None,
    min_gap: float = MIN_COLLAPSE_GAP,
) -> tuple[bool, str]:
    """Detect a converge-or-collapse pattern across seeds.

    A mean and interval are not a summary of a bimodal sample. A model that
    fails to train on a subset of seeds and is competitive on the rest is a
    more informative finding than a wide interval, so such cells are flagged
    rather than averaged.

    Detection keys on the largest gap between consecutive sorted values rather
    than on a median absolute deviation. With five seeds the deviation is far
    too noisy a scale estimate, and a robust-outlier rule fires on ordinary
    scatter; a collapse separates the sample by a wide absolute margin.

    Args:
        values: Per-seed values of a single metric.
        baseline: Optional score below which a run counts as collapsed.
        min_gap: Smallest separation, in metric units, that counts as a split.

    Returns:
        A pair of the flag and a short reason, empty when not flagged.
    """
    finite = _finite(values)
    if finite.size < 3:
        return False, ""

    if baseline is not None and np.any(finite < baseline):
        collapsed = int(np.sum(finite < baseline))
        return True, f"{collapsed}/{finite.size} seeds scored below the {baseline:.3f} baseline"

    ordered = np.sort(finite)
    gaps = np.diff(ordered)
    if gaps.size == 0:
        return False, ""
    index = int(np.argmax(gaps))
    largest = float(gaps[index])
    if largest >= min_gap:
        low = index + 1
        high = int(finite.size) - low
        return True, (f"{low} of {finite.size} seeds sit {largest:.3f} below the other {high}")
    return False, ""
