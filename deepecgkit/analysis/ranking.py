"""Ranking stability and cross-dataset flip tests across random seeds.

Ranks are formed inside a single dataset and seed, so the averaging
differences between datasets never contaminate them. Comparisons between two
models on one dataset are paired: within a seed both models see the same
split, so the split effect cancels in the difference.
"""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from deepecgkit.analysis.statistics import (
    CONFIDENCE_LEVEL,
    MIN_SEEDS_FOR_CLAIM,
    MIN_SEEDS_FOR_INTERVAL,
    NADEAU_BENGIO,
    benjamini_yekutieli_adjust,
    is_bimodal,
    kendall_tau,
    min_attainable_kendall_p,
    paired_summary,
)

DEFAULT_RANK_METRIC = "f1"
#: How the false-discovery-rate family of the flip-test grid is formed.
FDR_FAMILY_TESTABLE = "testable"
FDR_FAMILY_SIGNIFICANT = "significant"

#: Metrics for which a smaller value is the better one. They are negated
#: before ranking so that every statistic reads "higher is better".
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "ece",
        "adaptive_ece",
        "mce",
        "brier",
        "ece_temp",
        "adaptive_ece_temp",
        "mce_temp",
        "brier_temp",
    }
)

MIN_TEST_GROUPS = 10
MIN_MODELS_FOR_TAU = 3

VERDICT_SUPPORTED = "supported"
VERDICT_SUGGESTIVE = "suggestive"
VERDICT_UNRESOLVED = "unresolved"
VERDICT_NOT_REPRODUCED = "not_reproduced"
VERDICT_CONTRADICTED = "contradicted"
VERDICT_BIMODAL = "bimodal"
VERDICT_UNTESTABLE = "untestable"


def _metric_slice(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Select the rows for one metric.

    Args:
        runs: Long frame of per-run metric values.
        metric: Metric name.

    Returns:
        The matching rows.

    Raises:
        ValueError: If the metric is absent from the frame.
    """
    if "metric" not in runs.columns:
        raise ValueError("runs frame must carry a 'metric' column")
    selected = runs[runs["metric"] == metric]
    if selected.empty:
        raise ValueError(f"No rows for metric '{metric}'")
    if metric in LOWER_IS_BETTER:
        # Ranks, paired differences and flip tests all read "higher is
        # better"; a calibration error is negated so that the orientation
        # holds and a positive paired difference still means "A is better".
        selected = selected.assign(value=-selected["value"].to_numpy(dtype=float))
    return selected


def rank_within_seed(runs: pd.DataFrame, metric: str = DEFAULT_RANK_METRIC) -> pd.DataFrame:
    """Rank models inside each dataset and seed, best first.

    Args:
        runs: Long frame of per-run metric values.
        metric: Metric to rank on.

    Returns:
        A frame with dataset, seed, model, value, rank and rank_average. Ties
        share the better rank under "min" for display, while rank_average is
        used for Kendall's tau.
    """
    selected = _metric_slice(runs, metric)
    frames: list[pd.DataFrame] = []
    for (dataset, seed), group in selected.groupby(["dataset", "seed"], sort=True):
        usable = group[np.isfinite(group["value"])]
        if usable.empty:
            continue
        values = usable["value"].to_numpy(dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": usable["model"].tolist(),
                    "value": values,
                    "rank": stats.rankdata(-values, method="min"),
                    "rank_average": stats.rankdata(-values, method="average"),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["dataset", "seed", "model", "value", "rank", "rank_average"])
    return pd.concat(frames, ignore_index=True)


def rank_distribution(runs: pd.DataFrame, metric: str = DEFAULT_RANK_METRIC) -> pd.DataFrame:
    """Count how often each model attained each rank.

    Args:
        runs: Long frame of per-run metric values.
        metric: Metric to rank on.

    Returns:
        One row per dataset and model with a rank_<k> count column per
        attainable rank, the modal rank, the fraction of seeds at that rank,
        and the fraction of seeds spent at rank one.
    """
    ranked = rank_within_seed(runs, metric)
    if ranked.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for dataset, group in ranked.groupby("dataset", sort=True):
        n_models = int(group["model"].nunique())
        for model, sub in group.groupby("model", sort=True):
            counts = sub["rank"].value_counts()
            n_seeds = len(sub)
            row: dict[str, Any] = {
                "dataset": dataset,
                "model": model,
                "n_seeds": n_seeds,
                "n_models": n_models,
            }
            for position in range(1, n_models + 1):
                row[f"rank_{position}"] = int(counts.get(position, 0))
            modal_rank = int(counts.idxmax())
            row["mode_rank"] = modal_rank
            row["mode_fraction"] = float(counts.max()) / n_seeds if n_seeds else float("nan")
            row["top1_fraction"] = float(counts.get(1, 0)) / n_seeds if n_seeds else float("nan")
            row["mean_value"] = float(np.mean(sub["value"]))
            row["std_value"] = float(np.std(sub["value"], ddof=1)) if n_seeds > 1 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def tau_summary(runs: pd.DataFrame, metric: str = DEFAULT_RANK_METRIC) -> pd.DataFrame:
    """Kendall's tau-b between every pair of per-seed rankings.

    Args:
        runs: Long frame of per-run metric values.
        metric: Metric to rank on.

    Returns:
        One row per dataset with the mean, minimum and maximum tau over all
        seed pairs, the number of pairs, and the smallest two-sided p-value
        the coefficient could attain given the number of models. No p-value is
        computed: at four models that floor is 0.083, so significance is
        unreachable and the statistic is descriptive only.
    """
    ranked = rank_within_seed(runs, metric)
    rows: list[dict[str, Any]] = []
    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "n_models",
                "n_seeds",
                "n_pairs",
                "tau_mean",
                "tau_min",
                "tau_max",
                "min_attainable_p",
            ]
        )

    for dataset, group in ranked.groupby("dataset", sort=True):
        by_seed: dict[int, dict[str, float]] = {}
        for seed, sub in group.groupby("seed", sort=True):
            by_seed[int(seed)] = dict(zip(sub["model"], sub["rank_average"], strict=False))

        common = set.intersection(*(set(v) for v in by_seed.values())) if by_seed else set()
        n_models = len(common)
        taus: list[float] = []
        if n_models >= MIN_MODELS_FOR_TAU:
            for first, second in itertools.combinations(sorted(by_seed), 2):
                left = {model: by_seed[first][model] for model in common}
                right = {model: by_seed[second][model] for model in common}
                taus.append(kendall_tau(left, right))
        rows.append(
            {
                "dataset": dataset,
                "n_models": n_models,
                "n_seeds": len(by_seed),
                "n_pairs": len(taus),
                "tau_mean": float(np.mean(taus)) if taus else float("nan"),
                "tau_min": float(np.min(taus)) if taus else float("nan"),
                "tau_max": float(np.max(taus)) if taus else float("nan"),
                "min_attainable_p": (
                    min_attainable_kendall_p(n_models)
                    if n_models >= MIN_MODELS_FOR_TAU
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_values(
    selected: pd.DataFrame, model_a: str, model_b: str, dataset: str
) -> tuple[list[int], np.ndarray, np.ndarray, pd.DataFrame]:
    """Line up two models' per-seed values on one dataset.

    Args:
        selected: Rows for a single metric.
        model_a: First model.
        model_b: Second model.
        dataset: Dataset name.

    Returns:
        The shared seeds, the two aligned value arrays, and the contributing rows.
    """
    scope = selected[selected["dataset"] == dataset]
    left = scope[scope["model"] == model_a].set_index("seed")
    right = scope[scope["model"] == model_b].set_index("seed")
    shared = sorted(set(left.index) & set(right.index))
    shared = [
        seed
        for seed in shared
        if np.isfinite(left.loc[seed, "value"]) and np.isfinite(right.loc[seed, "value"])
    ]
    a_values = np.array([float(left.loc[seed, "value"]) for seed in shared], dtype=float)
    b_values = np.array([float(right.loc[seed, "value"]) for seed in shared], dtype=float)
    contributing = pd.concat([left.loc[shared], right.loc[shared]]) if shared else scope.iloc[0:0]
    return [int(seed) for seed in shared], a_values, b_values, contributing


def _mean_optional(frame: pd.DataFrame, column: str) -> int | None:
    """Mean of an optional integer column, or None when it is unpopulated.

    Args:
        frame: Rows contributing to a comparison.
        column: Column name.

    Returns:
        The rounded mean, or None.
    """
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean())) if not values.empty else None


def paired_difference(
    runs: pd.DataFrame,
    model_a: str,
    model_b: str,
    dataset: str,
    metric: str = DEFAULT_RANK_METRIC,
    confidence_level: float = CONFIDENCE_LEVEL,
    correction: str = NADEAU_BENGIO,
    min_test_groups: int = MIN_TEST_GROUPS,
) -> dict[str, Any]:
    """Paired per-seed difference between two models on one dataset.

    Pairing is exact: within a seed both models train and evaluate on the same
    split, so the split effect cancels. Across seeds the splits overlap
    heavily, so the standard error carries the Nadeau-Bengio correction.

    Args:
        runs: Long frame of per-run metric values.
        model_a: Model whose score is the minuend.
        model_b: Model whose score is the subtrahend.
        dataset: Dataset to compare on.
        metric: Metric to compare.
        confidence_level: Two-sided coverage.
        correction: Standard error correction, "nadeau-bengio" or "none".
        min_test_groups: Fewest independent test units that can support an
            interval.

    Returns:
        The paired summary augmented with seeds, averaging, the recorded split
        sizes, a bimodality flag and a reason string.
    """
    selected = _metric_slice(runs, metric)
    seeds, a_values, b_values, contributing = _paired_values(selected, model_a, model_b, dataset)
    differences = a_values - b_values

    n_train = _mean_optional(contributing, "n_train")
    n_test = _mean_optional(contributing, "n_test")
    n_test_groups = _mean_optional(contributing, "n_test_groups")

    summary = paired_summary(
        differences,
        confidence_level=confidence_level,
        correction=correction,
        n_train=n_train,
        n_test=n_test,
    )
    bimodal_a, reason_a = is_bimodal(a_values)
    bimodal_b, reason_b = is_bimodal(b_values)

    reason = ""
    if n_test_groups is not None and n_test_groups < min_test_groups:
        reason = (
            f"test partition holds {n_test_groups} independent unit(s), which cannot "
            "support an interval"
        )
    elif summary["n"] and summary["std"] == 0.0:
        reason = "zero variance across seeds; the interval is undefined, not infinitely precise"
    elif bimodal_a:
        reason = f"{model_a}: {reason_a}"
    elif bimodal_b:
        reason = f"{model_b}: {reason_b}"

    result: dict[str, Any] = dict(summary)
    result.update(
        {
            "model_a": model_a,
            "model_b": model_b,
            "dataset": dataset,
            "metric": metric,
            "seeds": seeds,
            "averaging": (
                contributing["averaging"].iloc[0] if not contributing.empty else "unknown"
            ),
            "n_train": n_train,
            "n_test": n_test,
            "n_test_groups": n_test_groups,
            "bimodal": bool(bimodal_a or bimodal_b),
            "testable": not (n_test_groups is not None and n_test_groups < min_test_groups),
            "reason": reason,
        }
    )
    return result


def flip_test(
    runs: pd.DataFrame,
    model_a: str,
    model_b: str,
    dataset_1: str,
    dataset_2: str,
    metric: str = DEFAULT_RANK_METRIC,
    confidence_level: float = CONFIDENCE_LEVEL,
    correction: str = NADEAU_BENGIO,
    min_test_groups: int = MIN_TEST_GROUPS,
    min_seeds: int = MIN_SEEDS_FOR_INTERVAL,
    min_seeds_for_claim: int = MIN_SEEDS_FOR_CLAIM,
) -> dict[str, Any]:
    """Intersection-union test for a cross-dataset ranking reversal.

    The reversal is supported only when both paired intervals exclude zero
    with opposite signs. Because the decision requires both components, no
    multiplicity correction is applied between them, and the reported p is the
    larger of the two. That p is an upper bound: pairing two-sided intervals
    with a sign requirement is a one-sided test at half the nominal level.

    Args:
        runs: Long frame of per-run metric values.
        model_a: First model.
        model_b: Second model.
        dataset_1: Dataset on which A is expected to lead.
        dataset_2: Dataset on which B is expected to lead.
        metric: Metric to compare.
        confidence_level: Two-sided coverage.
        correction: Standard error correction.
        min_test_groups: Fewest independent test units that can support a test.
        min_seeds: Fewest seeds for which an interval is computed at all.
        min_seeds_for_claim: Fewest seeds that may yield a supported verdict.

    Returns:
        Both paired summaries, the verdict, p_flip, and the averaging modes
        with a flag for whether the two magnitudes are comparable.
    """
    first = paired_difference(
        runs, model_a, model_b, dataset_1, metric, confidence_level, correction, min_test_groups
    )
    second = paired_difference(
        runs, model_a, model_b, dataset_2, metric, confidence_level, correction, min_test_groups
    )

    n_seeds = min(int(first["n"]), int(second["n"]))
    verdict, reason = _classify_flip(first, second, n_seeds, min_seeds, min_seeds_for_claim)

    # The intersection-union p-value exists whenever both components could
    # be tested; it is the larger component p. Which rows enter the
    # false-discovery-rate family is decided by all_flip_tests.
    if verdict in (VERDICT_UNTESTABLE, VERDICT_BIMODAL):
        p_flip = float("nan")
    else:
        p_flip = max(float(first["p_value"]), float(second["p_value"]))

    return {
        "model_a": model_a,
        "model_b": model_b,
        "dataset_1": dataset_1,
        "dataset_2": dataset_2,
        "metric": metric,
        "difference_1": first,
        "difference_2": second,
        "mean_1": first["mean"],
        "mean_2": second["mean"],
        "lower_1": first["lower_ci"],
        "upper_1": first["upper_ci"],
        "lower_2": second["lower_ci"],
        "upper_2": second["upper_ci"],
        "wins_1": first["n_wins"],
        "wins_2": second["n_wins"],
        "n_1": first["n"],
        "n_2": second["n"],
        "n_seeds": n_seeds,
        "averaging_1": first["averaging"],
        "averaging_2": second["averaging"],
        "comparable_magnitudes": first["averaging"] == second["averaging"],
        "verdict": verdict,
        "p_flip": p_flip,
        "reason": reason,
    }


def _classify_flip(
    first: dict[str, Any],
    second: dict[str, Any],
    n_seeds: int,
    min_seeds: int,
    min_seeds_for_claim: int,
) -> tuple[str, str]:
    """Assign a verdict to a pair of paired differences.

    Args:
        first: Paired difference on the first dataset.
        second: Paired difference on the second dataset.
        n_seeds: Seeds shared by both comparisons.
        min_seeds: Fewest seeds for which an interval is computed.
        min_seeds_for_claim: Fewest seeds that may yield a supported verdict.

    Returns:
        The verdict and a short reason.
    """
    if not first["testable"] or not second["testable"]:
        return VERDICT_UNTESTABLE, first["reason"] or second["reason"]
    if first["bimodal"] or second["bimodal"]:
        return VERDICT_BIMODAL, first["reason"] or second["reason"]
    if n_seeds < min_seeds:
        return VERDICT_UNTESTABLE, f"only {n_seeds} shared seed(s)"
    if not np.isfinite(first["mean"]) or not np.isfinite(second["mean"]):
        return VERDICT_UNTESTABLE, "no usable paired values"
    if first["std"] == 0.0 or second["std"] == 0.0:
        return VERDICT_UNTESTABLE, first["reason"] or second["reason"]

    opposite = (first["mean"] > 0) != (second["mean"] > 0)
    both_exclude = bool(first["excludes_zero"] and second["excludes_zero"])

    if not opposite:
        if both_exclude:
            return VERDICT_CONTRADICTED, "both differences share a sign and exclude zero"
        return VERDICT_NOT_REPRODUCED, "the mean ordering does not reverse"
    if not both_exclude:
        return VERDICT_UNRESOLVED, "the ordering reverses but an interval covers zero"
    if n_seeds < min_seeds_for_claim:
        return VERDICT_SUGGESTIVE, f"only {n_seeds} seeds; below the {min_seeds_for_claim} required"
    return VERDICT_SUPPORTED, ""


def all_flip_tests(
    runs: pd.DataFrame,
    metric: str = DEFAULT_RANK_METRIC,
    confidence_level: float = CONFIDENCE_LEVEL,
    correction: str = NADEAU_BENGIO,
    models: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    min_test_groups: int = MIN_TEST_GROUPS,
    fdr_family: str = FDR_FAMILY_TESTABLE,
) -> pd.DataFrame:
    """Run a flip test for every model pair shared by every dataset pair.

    Only models present on both datasets of a pair are compared, because
    models that failed to converge were omitted from some datasets. The
    exploratory grid is adjusted with Benjamini-Yekutieli over the family
    chosen by ``fdr_family``: every testable comparison (the default, so that
    the family does not depend on the outcomes), or only the comparisons
    whose verdict is supported, suggestive or contradicted (the historical
    behaviour, whose family size depends on the verdicts).

    Args:
        runs: Long frame of per-run metric values.
        metric: Metric to compare.
        confidence_level: Two-sided coverage.
        correction: Standard error correction.
        models: Restrict to these models.
        datasets: Restrict to these datasets.
        min_test_groups: Fewest independent test units that can support a test.
        fdr_family: FDR_FAMILY_TESTABLE or FDR_FAMILY_SIGNIFICANT.

    Returns:
        One row per tested combination, with raw and adjusted p-values and
        the family definition in ``fdr_family``.

    Raises:
        ValueError: If fdr_family is unknown.
    """
    if fdr_family not in (FDR_FAMILY_TESTABLE, FDR_FAMILY_SIGNIFICANT):
        raise ValueError(f"Unknown fdr_family: {fdr_family}")
    selected = _metric_slice(runs, metric)
    if models is not None:
        selected = selected[selected["model"].isin(list(models))]
    if datasets is not None:
        selected = selected[selected["dataset"].isin(list(datasets))]

    available = sorted(selected["dataset"].unique())
    rows: list[dict[str, Any]] = []
    for dataset_1, dataset_2 in itertools.combinations(available, 2):
        models_1 = set(selected[selected["dataset"] == dataset_1]["model"])
        models_2 = set(selected[selected["dataset"] == dataset_2]["model"])
        shared = sorted(models_1 & models_2)
        for model_a, model_b in itertools.combinations(shared, 2):
            result = flip_test(
                selected,
                model_a,
                model_b,
                dataset_1,
                dataset_2,
                metric=metric,
                confidence_level=confidence_level,
                correction=correction,
                min_test_groups=min_test_groups,
            )
            result.pop("difference_1", None)
            result.pop("difference_2", None)
            rows.append(result)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    in_family = np.isfinite(frame["p_flip"].to_numpy(dtype=float))
    if fdr_family == FDR_FAMILY_SIGNIFICANT:
        in_family &= (
            frame["verdict"]
            .isin([VERDICT_SUPPORTED, VERDICT_SUGGESTIVE, VERDICT_CONTRADICTED])
            .to_numpy()
        )
    family_p = [
        p if keep else float("nan") for p, keep in zip(frame["p_flip"], in_family, strict=False)
    ]
    frame["p_adjusted"] = benjamini_yekutieli_adjust(family_p)
    frame["fdr_family"] = fdr_family
    frame["fdr_family_size"] = int(in_family.sum())
    return frame.sort_values(["dataset_1", "dataset_2", "p_flip"], na_position="last").reset_index(
        drop=True
    )


def evaluate_rank_claim(
    runs: pd.DataFrame,
    model: str,
    dataset_best: str,
    dataset_worst: str,
    metric: str = DEFAULT_RANK_METRIC,
    opponents: Sequence[str] | None = None,
    confidence_level: float = CONFIDENCE_LEVEL,
    correction: str = NADEAU_BENGIO,
    min_test_groups: int = MIN_TEST_GROUPS,
) -> dict[str, Any]:
    """Decompose a "best here, worst there" claim into component flip tests.

    Every model present on both datasets is a component. Components that
    cannot be tested are reported as untestable rather than dropped, so the
    claim is never presented as supported on a silently reduced subset.

    Args:
        runs: Long frame of per-run metric values.
        model: The model the claim is about.
        dataset_best: Dataset on which the model is claimed strongest.
        dataset_worst: Dataset on which the model is claimed weakest.
        metric: Metric to compare.
        opponents: Restrict the components to these models.
        confidence_level: Two-sided coverage.
        correction: Standard error correction.
        min_test_groups: Fewest independent test units that can support a test.

    Returns:
        The component results, the overall verdict and the governing p-value.
    """
    selected = _metric_slice(runs, metric)
    models_best = set(selected[selected["dataset"] == dataset_best]["model"])
    models_worst = set(selected[selected["dataset"] == dataset_worst]["model"])
    candidates = sorted((models_best & models_worst) - {model})
    if opponents is not None:
        candidates = [name for name in candidates if name in set(opponents)]

    components: list[dict[str, Any]] = []
    for opponent in candidates:
        result = flip_test(
            selected,
            model,
            opponent,
            dataset_best,
            dataset_worst,
            metric=metric,
            confidence_level=confidence_level,
            correction=correction,
            min_test_groups=min_test_groups,
        )
        result.pop("difference_1", None)
        result.pop("difference_2", None)
        # The flip test is symmetric in the direction of the reversal. A claim
        # is not: the model must lead on dataset_best and trail on
        # dataset_worst. A supported reversal the other way round contradicts
        # the claim and is relabelled, with the reason recorded.
        leads_best = float(result["mean_1"]) > 0
        trails_worst = float(result["mean_2"]) < 0
        result["claimed_direction"] = bool(leads_best and trails_worst)
        if result["verdict"] in (VERDICT_SUPPORTED, VERDICT_SUGGESTIVE) and not (
            leads_best and trails_worst
        ):
            result["verdict"] = VERDICT_CONTRADICTED
            result["reason"] = "the ordering reverses, but opposite to the claimed direction"
        components.append(result)

    verdicts = [component["verdict"] for component in components]
    if not components:
        overall = VERDICT_UNTESTABLE
    elif all(verdict == VERDICT_SUPPORTED for verdict in verdicts):
        overall = VERDICT_SUPPORTED
    elif any(verdict == VERDICT_CONTRADICTED for verdict in verdicts):
        overall = VERDICT_CONTRADICTED
    elif any(verdict in (VERDICT_UNTESTABLE, VERDICT_BIMODAL) for verdict in verdicts):
        overall = VERDICT_UNTESTABLE
    elif any(verdict == VERDICT_NOT_REPRODUCED for verdict in verdicts):
        overall = VERDICT_NOT_REPRODUCED
    else:
        overall = VERDICT_UNRESOLVED

    finite = [c["p_flip"] for c in components if np.isfinite(c["p_flip"])]
    return {
        "model": model,
        "dataset_best": dataset_best,
        "dataset_worst": dataset_worst,
        "metric": metric,
        "components": components,
        "n_components": len(components),
        "n_supported": sum(1 for v in verdicts if v == VERDICT_SUPPORTED),
        "n_supported_in_direction": sum(
            1 for c in components if c["verdict"] == VERDICT_SUPPORTED and c["claimed_direction"]
        ),
        "n_untestable": sum(1 for v in verdicts if v in (VERDICT_UNTESTABLE, VERDICT_BIMODAL)),
        "verdict": overall,
        "p_claim": max(finite) if finite else float("nan"),
    }
