"""
Tests for deepecgkit.analysis.ranking module.
"""

import math

import pandas as pd
import pytest

from deepecgkit.analysis.ranking import (
    VERDICT_BIMODAL,
    VERDICT_CONTRADICTED,
    VERDICT_NOT_REPRODUCED,
    VERDICT_SUGGESTIVE,
    VERDICT_SUPPORTED,
    VERDICT_UNRESOLVED,
    VERDICT_UNTESTABLE,
    all_flip_tests,
    evaluate_rank_claim,
    flip_test,
    paired_difference,
    rank_distribution,
    rank_within_seed,
    tau_summary,
)

SEEDS = (42, 43, 44, 45, 46)


def make_runs(specification, metric="f1", n_test_groups=500, averaging="macro"):
    """Build a long run frame from nested per-seed values.

    Args:
        specification: Mapping of dataset to model to per-seed values.
        metric: Metric name to record.
        n_test_groups: Independent test units recorded for every run.
        averaging: Averaging mode recorded for every run.

    Returns:
        A frame shaped like the aggregation output.
    """
    rows = []
    for dataset, models in specification.items():
        for model, values in models.items():
            for seed, value in zip(SEEDS, values, strict=False):
                rows.append(
                    {
                        "run_dir": f"{model}-{dataset}-seed{seed}",
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "averaging": averaging,
                        "multi_label": False,
                        "n_train": 7000,
                        "n_val": 2000,
                        "n_test": 1000,
                        "n_test_groups": n_test_groups,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


SEPARATED = make_runs(
    {
        "d1": {
            "a": [0.700, 0.706, 0.698, 0.703, 0.699],
            "b": [0.650, 0.648, 0.653, 0.649, 0.652],
        },
        "d2": {
            "a": [0.600, 0.604, 0.597, 0.602, 0.598],
            "b": [0.700, 0.697, 0.703, 0.699, 0.702],
        },
    }
)


class TestRankWithinSeed:
    """Test per-seed ranking."""

    def test_best_model_takes_rank_one(self):
        """Ranks are assigned on descending metric value."""
        ranked = rank_within_seed(SEPARATED)
        first = ranked[(ranked["dataset"] == "d1") & (ranked["seed"] == 42)]
        assert first[first["model"] == "a"]["rank"].iloc[0] == 1
        assert first[first["model"] == "b"]["rank"].iloc[0] == 2

    def test_exact_ties_share_the_better_rank(self):
        """Tied models both take the competition rank."""
        tied = make_runs({"d1": {"a": [0.7] * 5, "b": [0.7] * 5, "c": [0.5] * 5}})
        ranked = rank_within_seed(tied)
        seed_rows = ranked[ranked["seed"] == 42]
        assert sorted(seed_rows["rank"].tolist()) == [1, 1, 3]

    def test_ties_get_an_averaged_rank_for_tau(self):
        """Kendall's tau needs midranks rather than competition ranks."""
        tied = make_runs({"d1": {"a": [0.7] * 5, "b": [0.7] * 5, "c": [0.5] * 5}})
        ranked = rank_within_seed(tied)
        seed_rows = ranked[ranked["seed"] == 42]
        assert sorted(seed_rows["rank_average"].tolist()) == [1.5, 1.5, 3.0]

    def test_unknown_metric_raises(self):
        """Ranking on a metric nobody recorded is an error, not an empty table."""
        with pytest.raises(ValueError, match="No rows for metric"):
            rank_within_seed(SEPARATED, metric="nonexistent")


class TestRankDistribution:
    """Test the rank frequency table."""

    def test_counts_sum_to_the_seed_count(self):
        """Every seed contributes exactly one rank per model."""
        distribution = rank_distribution(SEPARATED)
        row = distribution[(distribution["dataset"] == "d1") & (distribution["model"] == "a")].iloc[
            0
        ]
        assert row["rank_1"] + row["rank_2"] == row["n_seeds"]

    def test_top1_fraction_is_reported(self):
        """The share of seeds spent at rank one is the legible stability scalar."""
        distribution = rank_distribution(SEPARATED)
        row = distribution[(distribution["dataset"] == "d1") & (distribution["model"] == "a")].iloc[
            0
        ]
        assert row["top1_fraction"] == pytest.approx(1.0)


class TestTauSummary:
    """Test the ranking agreement summary."""

    def test_stable_rankings_give_tau_one(self):
        """When every seed agrees the coefficient is one throughout."""
        three = make_runs({"d1": {"a": [0.9] * 5, "b": [0.8] * 5, "c": [0.7] * 5}})
        summary = tau_summary(three)
        assert summary.iloc[0]["tau_mean"] == pytest.approx(1.0)
        assert summary.iloc[0]["n_pairs"] == 10

    def test_two_models_yields_no_coefficient(self):
        """Tau is degenerate below three models and is not invented."""
        summary = tau_summary(SEPARATED)
        assert math.isnan(summary.iloc[0]["tau_mean"])

    def test_permutation_floor_is_reported(self):
        """The table carries the smallest p the coefficient could reach."""
        three = make_runs({"d1": {"a": [0.9] * 5, "b": [0.8] * 5, "c": [0.7] * 5}})
        assert tau_summary(three).iloc[0]["min_attainable_p"] == pytest.approx(2.0 / 6)


class TestPairedDifference:
    """Test the within-dataset paired comparison."""

    def test_mean_difference_is_the_paired_mean(self):
        """The estimate is the mean of the per-seed differences."""
        result = paired_difference(SEPARATED, "a", "b", "d1")
        assert result["mean"] == pytest.approx(0.0508)
        assert result["n"] == 5
        assert result["n_wins"] == 5

    def test_correction_widens_the_interval(self):
        """Overlapping training sets make the plain interval too narrow."""
        plain = paired_difference(SEPARATED, "a", "b", "d1", correction="none")
        corrected = paired_difference(SEPARATED, "a", "b", "d1")
        assert corrected["sem"] > plain["sem"]

    def test_too_few_test_groups_is_not_testable(self):
        """Two held-out patients cannot support an interval."""
        scarce = make_runs(
            {
                "d1": {"a": [0.7] * 5, "b": [0.6] * 5},
                "d2": {"a": [0.6] * 5, "b": [0.7] * 5},
            },
            n_test_groups=2,
        )
        result = paired_difference(scarce, "a", "b", "d1")
        assert result["testable"] is False
        assert "independent unit" in result["reason"]

    def test_seeds_are_intersected(self):
        """Only seeds both models ran contribute to the comparison."""
        partial = SEPARATED[~((SEPARATED["model"] == "b") & (SEPARATED["seed"] == 46))]
        result = paired_difference(partial, "a", "b", "d1")
        assert result["n"] == 4
        assert 46 not in result["seeds"]


class TestFlipTest:
    """Test the intersection-union reversal test."""

    def test_large_opposite_effects_are_supported(self):
        """Two well-separated reversed differences resolve."""
        result = flip_test(SEPARATED, "a", "b", "d1", "d2")
        assert result["verdict"] == VERDICT_SUPPORTED
        assert result["mean_1"] > 0 > result["mean_2"]

    def test_noisy_reversal_stays_unresolved(self):
        """A reversal inside the noise is reported as unresolved."""
        noisy = make_runs(
            {
                "d1": {"a": [0.70, 0.66, 0.72, 0.68, 0.71], "b": [0.69, 0.71, 0.67, 0.70, 0.68]},
                "d2": {"a": [0.60, 0.62, 0.58, 0.61, 0.59], "b": [0.70, 0.72, 0.68, 0.71, 0.69]},
            }
        )
        assert flip_test(noisy, "a", "b", "d1", "d2")["verdict"] == VERDICT_UNRESOLVED

    def test_same_sign_differences_do_not_reproduce(self):
        """Without a reversal the claim simply fails."""
        same = make_runs(
            {
                "d1": {
                    "a": [0.700, 0.706, 0.698, 0.703, 0.699],
                    "b": [0.650, 0.648, 0.653, 0.649, 0.652],
                },
                "d2": {
                    "a": [0.700, 0.690, 0.710, 0.695, 0.705],
                    "b": [0.699, 0.706, 0.692, 0.703, 0.697],
                },
            }
        )
        assert flip_test(same, "a", "b", "d1", "d2")["verdict"] in (
            VERDICT_NOT_REPRODUCED,
            VERDICT_CONTRADICTED,
        )

    def test_same_sign_and_significant_is_contradicted(self):
        """A confidently non-reversed ordering contradicts the claim."""
        same = make_runs(
            {
                "d1": {
                    "a": [0.700, 0.706, 0.698, 0.703, 0.699],
                    "b": [0.650, 0.648, 0.653, 0.649, 0.652],
                },
                "d2": {
                    "a": [0.800, 0.806, 0.798, 0.803, 0.799],
                    "b": [0.700, 0.698, 0.703, 0.699, 0.702],
                },
            }
        )
        assert flip_test(same, "a", "b", "d1", "d2")["verdict"] == VERDICT_CONTRADICTED

    def test_scarce_test_groups_are_untestable(self):
        """The gate fires before any interval is reported."""
        scarce = make_runs(
            {
                "d1": {
                    "a": [0.700, 0.706, 0.698, 0.703, 0.699],
                    "b": [0.650, 0.648, 0.653, 0.649, 0.652],
                },
                "d2": {
                    "a": [0.600, 0.604, 0.597, 0.602, 0.598],
                    "b": [0.700, 0.697, 0.703, 0.699, 0.702],
                },
            },
            n_test_groups=2,
        )
        result = flip_test(scarce, "a", "b", "d1", "d2")
        assert result["verdict"] == VERDICT_UNTESTABLE
        assert math.isnan(result["p_flip"])

    def test_collapse_is_reported_as_bimodal(self):
        """A model that fails on some seeds is not summarised by a mean."""
        collapsing = make_runs(
            {
                "d1": {
                    "a": [0.70, 0.71, 0.19, 0.70, 0.18],
                    "b": [0.650, 0.648, 0.653, 0.649, 0.652],
                },
                "d2": {
                    "a": [0.600, 0.604, 0.597, 0.602, 0.598],
                    "b": [0.700, 0.697, 0.703, 0.699, 0.702],
                },
            }
        )
        assert flip_test(collapsing, "a", "b", "d1", "d2")["verdict"] == VERDICT_BIMODAL

    def test_fewer_than_five_seeds_cannot_be_supported(self):
        """Three seeds may only ever be suggestive."""
        few = SEPARATED[SEPARATED["seed"] < 45]
        result = flip_test(few, "a", "b", "d1", "d2")
        assert result["verdict"] == VERDICT_SUGGESTIVE
        assert "seeds" in result["reason"]

    def test_mixed_averaging_is_flagged_as_incomparable(self):
        """Magnitudes from different averaging modes must not be compared."""
        binary = make_runs(
            {
                "d2": {
                    "a": [0.600, 0.604, 0.597, 0.602, 0.598],
                    "b": [0.700, 0.697, 0.703, 0.699, 0.702],
                }
            },
            averaging="binary",
        )
        macro = make_runs(
            {
                "d1": {
                    "a": [0.700, 0.706, 0.698, 0.703, 0.699],
                    "b": [0.650, 0.648, 0.653, 0.649, 0.652],
                }
            }
        )
        result = flip_test(pd.concat([macro, binary]), "a", "b", "d1", "d2")
        assert result["comparable_magnitudes"] is False


class TestAllFlipTests:
    """Test the whole exploratory grid."""

    def test_only_shared_models_are_compared(self):
        """A model absent from one dataset yields no row for that pair."""
        lopsided = pd.concat([SEPARATED, make_runs({"d1": {"c": [0.5] * 5}})])
        grid = all_flip_tests(lopsided)
        assert "c" not in set(grid["model_a"]) | set(grid["model_b"])

    def test_adjusted_column_is_added(self):
        """The exploratory family carries a Benjamini-Yekutieli column."""
        grid = all_flip_tests(SEPARATED)
        assert "p_adjusted" in grid.columns
        assert grid.iloc[0]["p_adjusted"] >= grid.iloc[0]["p_flip"]


class TestEvaluateRankClaim:
    """Test decomposition of a best-here worst-there claim."""

    def test_all_supported_components_support_the_claim(self):
        """The claim holds only when every component holds."""
        result = evaluate_rank_claim(SEPARATED, "a", "d1", "d2")
        assert result["verdict"] == VERDICT_SUPPORTED
        assert result["n_components"] == 1

    def test_untestable_components_are_reported_not_dropped(self):
        """A claim is never presented as supported on a reduced subset."""
        scarce = make_runs(
            {
                "d1": {
                    "a": [0.700, 0.706, 0.698, 0.703, 0.699],
                    "b": [0.650, 0.648, 0.653, 0.649, 0.652],
                },
                "d2": {
                    "a": [0.600, 0.604, 0.597, 0.602, 0.598],
                    "b": [0.700, 0.697, 0.703, 0.699, 0.702],
                },
            },
            n_test_groups=2,
        )
        result = evaluate_rank_claim(scarce, "a", "d1", "d2")
        assert result["n_untestable"] == 1
        assert result["verdict"] == VERDICT_UNTESTABLE
