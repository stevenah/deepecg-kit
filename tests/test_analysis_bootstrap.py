"""Bootstrap of test recordings from stored predictions."""

import numpy as np
import pandas as pd
import pytest

from deepecgkit.analysis.bootstrap import (
    Predictions,
    bootstrap_intervals,
    paired_bootstrap_difference,
    read_predictions,
)
from deepecgkit.cli import main


def _single_label(path, seed, noise, n_records=60, windows=3):
    rng = np.random.default_rng(seed)
    labels = np.repeat(rng.integers(0, 3, n_records), windows)
    logits = rng.normal(0, noise, (len(labels), 3))
    logits[np.arange(len(labels)), labels] += 2.0
    prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "index": np.arange(len(labels)),
            "true_label": labels,
            "predicted_label": prob.argmax(axis=1),
            **{f"prob_{name}": prob[:, k] for k, name in enumerate(["A", "B", "C"])},
            "record": np.repeat([f"r{i:03d}" for i in range(n_records)], windows),
        }
    )
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / "predictions.txt", sep="\t", index=False, float_format="%.6f")
    return path


class TestReadPredictions:
    def test_single_label_file(self, tmp_path):
        stored = read_predictions(_single_label(tmp_path / "run", 0, 1.0))
        assert stored.y_true.shape == (180,)
        assert stored.y_prob.shape == (180, 3)
        assert len(np.unique(stored.groups)) == 60

    def test_multi_label_file(self, tmp_path):
        frame = pd.DataFrame(
            {
                "index": [0, 1, 2],
                "true_X": [1, 0, 1],
                "pred_X": [1, 0, 0],
                "prob_X": [0.9, 0.2, 0.4],
                "true_Y": [0, 0, 1],
                "pred_Y": [0, 0, 1],
                "prob_Y": [0.1, 0.3, 0.8],
                "record": ["7", "8", "9"],
            }
        )
        frame.to_csv(tmp_path / "predictions.txt", sep="\t", index=False)
        stored = read_predictions(tmp_path)
        assert stored.y_true.shape == (3, 2)
        assert stored.y_prob.tolist() == [[0.9, 0.1], [0.2, 0.3], [0.4, 0.8]]
        assert stored.groups.tolist() == ["7", "8", "9"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_predictions(tmp_path)


class TestIntervals:
    def test_interval_brackets_the_full_sample_score(self, tmp_path):
        stored = read_predictions(_single_label(tmp_path / "run", 0, 1.0))
        intervals = bootstrap_intervals(stored, ["f1", "auroc"], n_resamples=200)
        for interval in intervals.values():
            assert interval.lower <= interval.estimate <= interval.upper
            assert interval.upper - interval.lower > 0

    def test_resampling_is_reproducible(self, tmp_path):
        stored = read_predictions(_single_label(tmp_path / "run", 0, 1.0))
        first = bootstrap_intervals(stored, ["f1"], n_resamples=50, seed=3)
        second = bootstrap_intervals(stored, ["f1"], n_resamples=50, seed=3)
        assert first == second

    def test_recordings_are_resampled_whole(self):
        groups = np.repeat(["a", "b", "c", "d"], 5)
        stored = Predictions(np.repeat([0, 1, 0, 1], 5), np.tile([[0.6, 0.4]], (20, 1)), groups)
        intervals = bootstrap_intervals(stored, ["accuracy"], n_resamples=100)
        # Every draw takes whole recordings, so accuracy moves in steps of a recording.
        assert intervals["accuracy"].lower in (0.0, 0.25, 0.5, 0.75, 1.0)


class TestPairedDifference:
    def test_a_run_against_itself_differs_by_nothing(self, tmp_path):
        stored = read_predictions(_single_label(tmp_path / "run", 0, 1.0))
        difference = paired_bootstrap_difference(stored, stored, ["f1"], n_resamples=50)
        assert difference["f1"] == (0.0, 0.0, 0.0)

    def test_a_clearly_better_run_excludes_zero(self, tmp_path):
        weak = read_predictions(_single_label(tmp_path / "weak", 0, 3.0))
        strong = read_predictions(_single_label(tmp_path / "strong", 0, 0.3))
        difference = paired_bootstrap_difference(weak, strong, ["f1"], n_resamples=200)
        assert difference["f1"].lower > 0

    def test_runs_on_different_samples_are_refused(self, tmp_path):
        a = read_predictions(_single_label(tmp_path / "a", 0, 1.0))
        b = read_predictions(_single_label(tmp_path / "b", 1, 1.0))
        with pytest.raises(ValueError, match="same test samples"):
            paired_bootstrap_difference(a, b, ["f1"], n_resamples=10)


def test_bootstrap_command_writes_intervals_and_differences(tmp_path):
    runs = tmp_path / "runs"
    for model, noise in (("kanres", 0.3), ("simple-cnn", 1.0)):
        run = _single_label(runs / f"{model}-af-classification-seed42", 0, noise)
        pd.DataFrame(
            [{"model": model, "dataset": "af-classification", "f1": 0.5, "average": "macro"}]
        ).to_csv(run / "classification_metrics.csv", index=False)
    output = tmp_path / "out"
    result = main(
        ["bootstrap", "--runs-dir", str(runs), "--n-resamples", "50", "--output-dir", str(output)]
    )
    assert result == 0
    intervals = pd.read_csv(output / "bootstrap_intervals.csv")
    differences = pd.read_csv(output / "bootstrap_differences.csv")
    assert set(intervals["model"]) == {"kanres", "simple-cnn"}
    assert set(differences["reference"]) == {"kanres"}
    assert set(differences["model"]) == {"simple-cnn"}
