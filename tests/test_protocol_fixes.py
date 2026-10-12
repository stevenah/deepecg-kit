"""Regression tests for the analysis-protocol fixes made for the journal benchmark.

Each test pins one behaviour that the manuscript's Supplementary Note 3 used
to list as a known issue: failed runs entering aggregates, duplicates resolved
by modification time across different hyperparameters, flip-test verdicts
blind to the claimed direction, an outcome-dependent false-discovery-rate
family, calibration errors ranked as if higher were better, probe AUROC on a
two-class held-out split, probe columns misaligned with class indices, silent
stratification fallbacks, and run metadata without provenance.
"""

import json
import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from deepecgkit.analysis import aggregation, ranking
from deepecgkit.analysis.latex import flip_survival_table, results_table, tau_table
from deepecgkit.datasets.splitting import DataSplitter
from deepecgkit.evaluation.representation import linear_probe
from deepecgkit.utils.provenance import dataset_fingerprint, environment_summary, git_commit
from tests.test_analysis_aggregation import make_run
from tests.test_analysis_ranking import make_runs


class TestAggregatorExitStatus:
    def test_failed_runs_are_set_aside(self, tmp_path):
        ok = make_run(tmp_path, "resnet", "ptbxl", seed=42)
        bad = make_run(tmp_path, "resnet", "ptbxl", seed=43)
        (bad / "run_metadata.json").write_text(
            json.dumps({"seed": 43, "model": "resnet", "dataset": "ptbxl", "returncode": 1})
        )
        records, diagnostics = aggregation.discover_runs([tmp_path])
        assert [r.run_dir for r in records] == [ok]
        assert diagnostics["failed"] and str(bad) in diagnostics["failed"][0]

    def test_failed_runs_can_be_kept(self, tmp_path):
        bad = make_run(tmp_path, "resnet", "ptbxl", seed=43)
        (bad / "run_metadata.json").write_text(
            json.dumps({"seed": 43, "model": "resnet", "dataset": "ptbxl", "returncode": 2})
        )
        records, _ = aggregation.discover_runs([tmp_path], include_failed=True)
        assert len(records) == 1 and records[0].returncode == 2

    def test_runs_without_a_status_are_kept(self, tmp_path):
        make_run(tmp_path, "resnet", "ptbxl", seed=42)
        records, diagnostics = aggregation.discover_runs([tmp_path])
        assert len(records) == 1 and diagnostics["failed"] == []


class TestAggregatorDuplicateKey:
    def test_different_fingerprints_are_not_duplicates(self, tmp_path):
        for stem, fingerprint in (("a", "0123abcd"), ("b", "89efcdef")):
            run = make_run(tmp_path, "resnet", "ptbxl", seed=42, name=f"resnet-ptbxl-{stem}")
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "model": "resnet",
                        "dataset": "ptbxl",
                        "returncode": 0,
                        "params_fingerprint": fingerprint,
                    }
                )
            )
        records, diagnostics = aggregation.discover_runs([tmp_path])
        assert len(records) == 2
        assert diagnostics["duplicate"] == []

    def test_duplicate_diagnostic_names_dropped_and_kept(self, tmp_path):
        first = make_run(tmp_path, "resnet", "ptbxl", seed=42, name="first")
        second = make_run(tmp_path, "resnet", "ptbxl", seed=42, name="second")
        import os
        import time

        now = time.time()
        os.utime(first / "classification_metrics.csv", (now - 100, now - 100))
        os.utime(second / "classification_metrics.csv", (now, now))
        records, diagnostics = aggregation.discover_runs([tmp_path])
        assert [r.run_dir for r in records] == [second]
        assert "dropped" in diagnostics["duplicate"][0]
        assert str(first) in diagnostics["duplicate"][0]
        assert str(second) in diagnostics["duplicate"][0]


def _values(base: float, step: float = 0.002, n: int = 5) -> list[float]:
    return [base + step * i for i in range(n)]


def _reversal_runs(direction: int):
    """Runs where model A beats B on d1 and trails on d2 (direction 1), or the mirror."""
    return make_runs(
        {
            "d1": {
                "A": _values(0.80 + direction * 0.05, 0.002),
                "B": _values(0.80 - direction * 0.05, 0.004),
            },
            "d2": {
                "A": _values(0.70 - direction * 0.05, 0.002),
                "B": _values(0.70 + direction * 0.05, 0.004),
            },
        }
    )


class TestClaimDirection:
    def test_reversal_in_claimed_direction_is_supported(self):
        runs = _reversal_runs(+1)
        claim = ranking.evaluate_rank_claim(runs, "A", "d1", "d2")
        assert claim["verdict"] == ranking.VERDICT_SUPPORTED
        assert claim["n_supported_in_direction"] == 1
        assert claim["components"][0]["claimed_direction"] is True

    def test_reversal_against_the_claim_is_contradicted(self):
        runs = _reversal_runs(-1)
        claim = ranking.evaluate_rank_claim(runs, "A", "d1", "d2")
        assert claim["verdict"] == ranking.VERDICT_CONTRADICTED
        assert claim["n_supported_in_direction"] == 0
        assert "opposite" in claim["components"][0]["reason"]

    def test_symmetric_flip_test_is_unchanged(self):
        runs = _reversal_runs(-1)
        grid = ranking.all_flip_tests(runs)
        assert grid.iloc[0]["verdict"] == ranking.VERDICT_SUPPORTED


class TestFdrFamily:
    def test_testable_family_includes_unresolved_tests(self):
        runs = make_runs(
            {
                "d1": {"A": _values(0.80, 0.001), "B": _values(0.79, 0.004)},  # overlapping
                "d2": {"A": _values(0.60, 0.001), "B": _values(0.70, 0.003)},
            }
        )
        grid = ranking.all_flip_tests(runs)
        assert grid.iloc[0]["fdr_family"] == ranking.FDR_FAMILY_TESTABLE
        assert np.isfinite(grid.iloc[0]["p_flip"])
        assert grid.iloc[0]["fdr_family_size"] == 1
        significant = ranking.all_flip_tests(runs, fdr_family=ranking.FDR_FAMILY_SIGNIFICANT)
        if significant.iloc[0]["verdict"] not in (
            ranking.VERDICT_SUPPORTED,
            ranking.VERDICT_SUGGESTIVE,
            ranking.VERDICT_CONTRADICTED,
        ):
            assert significant.iloc[0]["fdr_family_size"] == 0
            assert np.isnan(significant.iloc[0]["p_adjusted"])

    def test_unknown_family_is_rejected(self):
        runs = _reversal_runs(+1)
        with pytest.raises(ValueError, match="fdr_family"):
            ranking.all_flip_tests(runs, fdr_family="all")


class TestLowerIsBetter:
    def test_calibration_error_ranks_lower_first(self):
        runs = make_runs({"d1": {"A": [0.10, 0.11, 0.12], "B": [0.02, 0.03, 0.04]}}, metric="ece")
        ranked = ranking.rank_within_seed(runs, metric="ece")
        best = ranked[ranked["rank"] == 1]["model"].unique().tolist()
        assert best == ["B"]

    def test_f1_still_ranks_higher_first(self):
        runs = make_runs({"d1": {"A": [0.10, 0.11, 0.12], "B": [0.02, 0.03, 0.04]}}, metric="f1")
        ranked = ranking.rank_within_seed(runs, metric="f1")
        assert ranked[ranked["rank"] == 1]["model"].unique().tolist() == ["A"]


class TestProbeScoring:
    def _features(self, labels, seed=0):
        rng = np.random.default_rng(seed)
        centres = np.eye(4) * 3.0
        return centres[labels] + rng.normal(scale=0.3, size=(len(labels), 4))

    def test_two_class_heldout_split_is_scored_one_vs_rest(self):
        rng = np.random.default_rng(1)
        train_labels = rng.integers(0, 4, 200)
        test_labels = np.array([2, 3] * 20)
        result = linear_probe(
            (self._features(train_labels), train_labels),
            (self._features(test_labels, seed=2), test_labels),
        )
        assert result.task == "multiclass"
        assert result.metrics["auroc_macro"] > 0.9
        assert "2 of 4 classes" in (result.note or "")

    def test_columns_follow_class_indices_when_training_lacks_a_class(self):
        rng = np.random.default_rng(3)
        train_labels = rng.choice([0, 1, 3], 200)  # class 2 never seen
        test_labels = np.array([0, 1, 3] * 20)
        result = linear_probe(
            (self._features(train_labels), train_labels),
            (self._features(test_labels, seed=4), test_labels),
        )
        assert result.metrics["accuracy"] > 0.9

    def test_non_convergence_is_noted(self):
        rng = np.random.default_rng(5)
        train_labels = rng.integers(0, 2, 60)
        test_labels = rng.integers(0, 2, 30)
        features = rng.normal(size=(60, 50))
        result = linear_probe(
            (features, train_labels),
            (rng.normal(size=(30, 50)), test_labels),
            max_iter=1,
        )
        assert "did not converge" in (result.note or "")


class TestSplitWarnings:
    def test_dropped_stratification_warns(self):
        groups = np.repeat(np.arange(6), 10)
        labels = np.zeros(60, dtype=int)
        labels[:10] = 1  # one class is the majority label of a single group
        dataset = TensorDataset(torch.zeros(60, 1, 8), torch.as_tensor(labels))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DataSplitter(
                dataset, val_split=0.2, test_split=0.2, seed=0, stratify=labels, groups=groups
            ).split()
        assert any("not stratified" in str(w.message) for w in caught)


class TestProvenance:
    def test_environment_summary_fields(self):
        summary = environment_summary("cpu")
        for key in ("host", "platform", "python_version", "torch_version", "device"):
            assert summary[key]
        assert summary["device"] == "cpu"

    def test_git_commit_is_a_hash_or_none(self):
        info = git_commit()
        assert info is None or (len(info["commit"]) == 40 and isinstance(info["dirty"], bool))

    def test_dataset_fingerprint_changes_with_the_files(self, tmp_path):
        (tmp_path / "a.dat").write_bytes(b"x" * 10)
        before = dataset_fingerprint(tmp_path)
        (tmp_path / "b.dat").write_bytes(b"y" * 5)
        after = dataset_fingerprint(tmp_path)
        assert before["n_files"] == 1 and after["n_files"] == 2
        assert before["sha256"] != after["sha256"]
        assert dataset_fingerprint(tmp_path / "missing") is None


class TestCaptionsFollowSettings:
    def _aggregated(self):
        runs = make_runs({"d1": {"A": _values(0.8), "B": _values(0.7)}})
        return aggregation.aggregate_runs(runs, confidence_level=0.9)

    def test_results_caption_states_level_and_split(self):
        table = results_table(self._aggregated(), "d1", metrics=("f1",), confidence_level=0.9)
        assert "90\\%" in table
        assert "patient-level" not in table
        assert "grouped split" in table

    def test_flip_caption_states_correction_and_family(self):
        grid = ranking.all_flip_tests(_reversal_runs(+1), correction="none")
        table = flip_survival_table(grid, confidence_level=0.99, correction="none")
        assert "99\\%" in table
        assert "Nadeau" not in table
        assert "every testable comparison" in table
        assert "selected by inspecting" not in table

    def test_tau_floor_follows_model_count(self):
        runs = make_runs({"d1": {m: _values(0.5 + 0.01 * i, n=3) for i, m in enumerate("ABCDE")}})
        taus = ranking.tau_summary(runs)
        table = tau_table(taus)
        assert "0.083" not in table
        assert "|M|=5" in table
