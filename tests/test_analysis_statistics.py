"""
Tests for deepecgkit.analysis.statistics module.
"""

import math

import numpy as np
import pytest

from deepecgkit.analysis.statistics import (
    benjamini_yekutieli_adjust,
    holm_adjust,
    is_bimodal,
    kendall_tau,
    min_attainable_kendall_p,
    nadeau_bengio_factor,
    paired_summary,
    summarize_scores,
)

SAMPLE = [0.70, 0.71, 0.69, 0.72, 0.68]


class TestSummarizeScores:
    """Test the across-seed summary."""

    def test_no_values_yields_nan_and_zero_count(self):
        """An empty sample reports nothing rather than raising."""
        result = summarize_scores([])
        assert result["n"] == 0
        assert math.isnan(result["mean"])
        assert math.isnan(result["lower_ci"])

    def test_single_value_has_no_interval(self):
        """One run carries no dispersion, so the interval is undefined."""
        result = summarize_scores([0.7])
        assert result["n"] == 1
        assert result["mean"] == pytest.approx(0.7)
        assert math.isnan(result["std"])
        assert math.isnan(result["lower_ci"])
        assert math.isnan(result["upper_ci"])

    def test_two_values_define_an_interval(self):
        """Two runs are enough for a t interval, however wide."""
        result = summarize_scores([0.6, 0.8])
        assert result["n"] == 2
        assert result["lower_ci"] < result["mean"] < result["upper_ci"]

    def test_std_uses_sample_denominator(self):
        """Dispersion across seeds is a sample standard deviation."""
        result = summarize_scores(SAMPLE)
        assert result["std"] == pytest.approx(float(np.std(SAMPLE, ddof=1)))

    def test_non_finite_values_are_dropped_from_the_count(self):
        """A failed metric reduces n rather than poisoning the mean."""
        result = summarize_scores([0.7, float("nan"), 0.8])
        assert result["n"] == 2
        assert result["mean"] == pytest.approx(0.75)

    def test_interval_is_wider_than_the_normal_approximation(self):
        """The t quantile exceeds 1.96 at five seeds."""
        result = summarize_scores(SAMPLE)
        half_width = result["upper_ci"] - result["mean"]
        assert half_width > 1.96 * result["sem"]

    def test_invalid_confidence_level_raises(self):
        """A coverage outside the unit interval is rejected."""
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            summarize_scores(SAMPLE, confidence_level=1.5)


class TestNadeauBengioFactor:
    """Test the correction for overlapping training sets."""

    def test_without_split_sizes_it_is_the_plain_factor(self):
        """Absent split sizes the ordinary standard error is used."""
        assert nadeau_bengio_factor(5, None, None) == pytest.approx(1.0 / math.sqrt(5))

    def test_widening_matches_the_published_formula(self):
        """The factor is sqrt(1/n + n_test/n_train)."""
        assert nadeau_bengio_factor(5, 7000, 1000) == pytest.approx(
            math.sqrt(1.0 / 5 + 1000.0 / 7000.0)
        )

    def test_correction_widens_rather_than_narrows(self):
        """Accounting for overlap can only increase the standard error."""
        assert nadeau_bengio_factor(5, 7000, 1000) > nadeau_bengio_factor(5, None, None)


class TestPairedSummary:
    """Test the paired difference summary."""

    def test_mean_and_wins_are_reported(self):
        """The point estimate and the sign count both describe the sample."""
        result = paired_summary([0.02, 0.03, 0.025, 0.028, 0.022], correction="none")
        assert result["n"] == 5
        assert result["n_wins"] == 5
        assert result["mean"] == pytest.approx(0.025)

    def test_zero_variance_is_undefined_not_certain(self):
        """Identical differences must not become an infinitely precise result."""
        result = paired_summary([0.1] * 5)
        assert result["std"] == 0.0
        assert math.isnan(result["p_value"])
        assert math.isnan(result["lower_ci"])
        assert result["excludes_zero"] is False

    def test_correction_raises_the_p_value(self):
        """Correcting for overlap makes the test more conservative."""
        differences = [0.02, 0.03, 0.025, 0.028, 0.022]
        plain = paired_summary(differences, correction="none")
        corrected = paired_summary(
            differences, correction="nadeau-bengio", n_train=7000, n_test=1000
        )
        assert corrected["sem"] > plain["sem"]
        assert corrected["p_value"] > plain["p_value"]

    def test_unknown_correction_raises(self):
        """An unrecognised correction name is rejected."""
        with pytest.raises(ValueError, match="Unknown correction"):
            paired_summary([0.1, 0.2], correction="bootstrap")


class TestKendallTau:
    """Test the ranking agreement coefficient."""

    def test_identical_rankings_agree_perfectly(self):
        """Two identical orderings give tau of one."""
        ranks = {"a": 1, "b": 2, "c": 3, "d": 4}
        assert kendall_tau(ranks, ranks) == pytest.approx(1.0)

    def test_reversed_rankings_disagree_perfectly(self):
        """A reversed ordering gives tau of minus one."""
        first = {"a": 1, "b": 2, "c": 3, "d": 4}
        second = {"a": 4, "b": 3, "c": 2, "d": 1}
        assert kendall_tau(first, second) == pytest.approx(-1.0)

    def test_mismatched_labels_raise(self):
        """Rankings over different models cannot be compared."""
        with pytest.raises(ValueError, match="same labels"):
            kendall_tau({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2, "d": 3})

    def test_two_labels_raise(self):
        """With two models tau is degenerate and is refused."""
        with pytest.raises(ValueError, match="at least 3 labels"):
            kendall_tau({"a": 1, "b": 2}, {"a": 2, "b": 1})


class TestMinAttainableKendallP:
    """Test the exact permutation floor."""

    @pytest.mark.parametrize("n_labels,expected", [(3, 2.0 / 6), (4, 2.0 / 24), (5, 2.0 / 120)])
    def test_floor_matches_two_over_factorial(self, n_labels, expected):
        """The floor is 2/n! under the tie-free permutation null."""
        assert min_attainable_kendall_p(n_labels) == pytest.approx(expected)

    def test_four_models_cannot_reach_significance(self):
        """At four models no coefficient can fall below 0.05."""
        assert min_attainable_kendall_p(4) > 0.05


class TestAdjusters:
    """Test the multiplicity adjustments."""

    def test_holm_matches_hand_computation(self):
        """Holm scales each ordered p by the remaining family size."""
        assert holm_adjust([0.001, 0.01, 0.03, 0.2, 0.5]) == pytest.approx(
            [0.005, 0.04, 0.09, 0.4, 0.5]
        )

    def test_benjamini_yekutieli_pays_the_harmonic_penalty(self):
        """The smallest adjusted value carries the harmonic factor."""
        pvalues = [0.001, 0.01, 0.03, 0.2, 0.5]
        harmonic = sum(1.0 / (i + 1) for i in range(5))
        adjusted = benjamini_yekutieli_adjust(pvalues)
        assert adjusted[0] == pytest.approx(0.001 * harmonic * 5 / 1)

    def test_adjusted_values_never_decrease_with_raw_value(self):
        """Both procedures preserve the ordering of the raw p-values."""
        pvalues = [0.001, 0.01, 0.03, 0.2, 0.5]
        for adjusted in (holm_adjust(pvalues), benjamini_yekutieli_adjust(pvalues)):
            assert adjusted == sorted(adjusted)

    def test_non_finite_entries_pass_through(self):
        """An untestable comparison stays untestable after adjustment."""
        adjusted = holm_adjust([0.01, float("nan"), 0.2])
        assert math.isnan(adjusted[1])
        assert not math.isnan(adjusted[0])

    def test_empty_family_is_all_nan(self):
        """With nothing testable there is nothing to adjust."""
        assert all(math.isnan(value) for value in benjamini_yekutieli_adjust([float("nan")]))


class TestIsBimodal:
    """Test the converge-or-collapse detector."""

    def test_partial_collapse_is_flagged(self):
        """A model that fails on some seeds is not summarised by a mean."""
        flagged, reason = is_bimodal([0.19, 0.19, 0.65, 0.68, 0.66])
        assert flagged is True
        assert "seeds" in reason

    def test_ordinary_scatter_is_not_flagged(self):
        """Normal seed-to-seed variation must not look like a collapse."""
        flagged, _ = is_bimodal([0.7048, 0.7012, 0.7003, 0.7091, 0.7066])
        assert flagged is False

    def test_baseline_catches_a_single_collapse(self):
        """A run below the majority-class baseline is a collapse."""
        flagged, reason = is_bimodal([0.19, 0.65, 0.66, 0.68, 0.67], baseline=0.30)
        assert flagged is True
        assert "baseline" in reason

    def test_too_few_values_is_not_flagged(self):
        """Two runs cannot establish a mode."""
        flagged, _ = is_bimodal([0.1, 0.9])
        assert flagged is False
