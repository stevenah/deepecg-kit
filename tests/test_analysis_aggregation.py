"""
Tests for deepecgkit.analysis.aggregation module.
"""

import json
import math
from pathlib import Path

import pytest

from deepecgkit.analysis.aggregation import (
    AVERAGING_MACRO_LABEL,
    AVERAGING_UNKNOWN,
    LEGACY_SEED,
    aggregate_runs,
    discover_runs,
    read_run_metrics,
    reconcile_averaging,
    runs_to_frame,
)

SINGLE_LABEL_HEADER = "model,dataset,accuracy,precision,recall,f1,auroc,mcc,average"
MULTI_LABEL_HEADER = "model,dataset,exact_match_accuracy,label_accuracy,precision,recall,f1,auroc"


def make_run(
    root: Path,
    model: str,
    dataset: str,
    seed=None,
    f1: float = 0.7,
    multi_label: bool = False,
    average: str = "macro",
    write_metadata: bool = True,
    n_test_groups: int = 500,
    name=None,
) -> Path:
    """Create a fake run directory holding one metrics row.

    Args:
        root: Parent directory.
        model: Model name written into the CSV.
        dataset: Dataset name written into the CSV.
        seed: Seed for the sidecar, or None to omit it.
        f1: F1 value to record.
        multi_label: Write the multi-label schema instead of the single-label one.
        average: Averaging mode for the single-label schema, or None to omit.
        write_metadata: Whether to write the run_metadata.json sidecar.
        n_test_groups: Independent test units to record in split_info.json.
        name: Directory name, defaulting to the seed-suffixed convention.

    Returns:
        The created run directory.
    """
    stem = name if name is not None else f"{model}-{dataset}-seed{seed}"
    run_dir = root / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    if multi_label:
        header = MULTI_LABEL_HEADER
        row = f"{model},{dataset},0.5,0.86,0.63,0.82,{f1},0.92"
    elif average is None:
        header = "model,dataset,accuracy,precision,recall,f1,auroc,mcc"
        row = f"{model},{dataset},0.77,0.69,0.64,{f1},0.91,0.57"
    else:
        header = SINGLE_LABEL_HEADER
        row = f"{model},{dataset},0.77,0.69,0.64,{f1},0.91,0.57,{average}"
    (run_dir / "classification_metrics.csv").write_text(f"{header}\n{row}\n")

    if write_metadata and seed is not None:
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"seed": seed, "model": model, "dataset": dataset, "returncode": 0})
        )
    (run_dir / "split_info.json").write_text(
        json.dumps({"n_train": 7000, "n_val": 2000, "n_test": 1000, "n_test_groups": n_test_groups})
    )
    return run_dir


class TestReadRunMetrics:
    """Test parsing a single run directory."""

    def test_single_label_schema_is_read(self, temp_dir):
        """The nine-column schema yields its six metrics."""
        record = read_run_metrics(make_run(temp_dir, "resnet", "af-classification", 42))
        assert record is not None
        assert record.model == "resnet"
        assert record.seed == 42
        assert record.averaging == "macro"
        assert set(record.metrics) == {"accuracy", "precision", "recall", "f1", "auroc", "mcc"}

    def test_multi_label_schema_is_macro_over_labels(self, temp_dir):
        """PTB-XL rows carry a different metric set and their own averaging."""
        record = read_run_metrics(make_run(temp_dir, "resnet", "ptbxl", 42, multi_label=True))
        assert record.averaging == AVERAGING_MACRO_LABEL
        assert "exact_match_accuracy" in record.metrics
        assert "mcc" not in record.metrics

    def test_missing_average_column_is_unknown(self, temp_dir):
        """A run predating the averaging column is not assumed macro."""
        record = read_run_metrics(
            make_run(temp_dir, "resnet", "af-classification", 42, average=None)
        )
        assert record.averaging == AVERAGING_UNKNOWN

    def test_binary_averaging_is_preserved(self, temp_dir):
        """Positive-class scores are labelled as such."""
        record = read_run_metrics(make_run(temp_dir, "crnn", "mitbih-afdb", 42, average="binary"))
        assert record.averaging == "binary"

    def test_seed_comes_from_the_sidecar(self, temp_dir):
        """The recorded seed wins over the directory name."""
        run_dir = make_run(temp_dir, "resnet", "ptbxl", 42, name="oddly-named-dir")
        assert read_run_metrics(run_dir).seed == 42

    def test_seed_falls_back_to_the_directory_suffix(self, temp_dir):
        """A run without a sidecar is still readable when named by convention."""
        run_dir = make_run(temp_dir, "resnet", "ptbxl", 43, write_metadata=False)
        assert read_run_metrics(run_dir).seed == 43

    def test_split_sizes_are_read(self, temp_dir):
        """The split diagnostic travels with the record."""
        record = read_run_metrics(make_run(temp_dir, "crnn", "mitbih-afdb", 42, n_test_groups=2))
        assert record.n_test_groups == 2
        assert record.n_train == 7000

    def test_directory_without_metrics_is_unusable(self, temp_dir):
        """A crashed run produces no metrics and is not a record."""
        empty = temp_dir / "crashed"
        empty.mkdir()
        assert read_run_metrics(empty) is None

    def test_empty_metrics_file_is_unusable(self, temp_dir):
        """A truncated metrics file is rejected rather than parsed."""
        run_dir = temp_dir / "truncated"
        run_dir.mkdir()
        (run_dir / "classification_metrics.csv").write_text("")
        assert read_run_metrics(run_dir) is None


class TestDiscoverRuns:
    """Test collecting runs from directory trees."""

    def test_seedless_runs_are_excluded_by_default(self, temp_dir):
        """A run with no recoverable seed cannot join a seeded comparison."""
        make_run(temp_dir, "resnet", "ptbxl", None, write_metadata=False, name="legacy-run")
        records, diagnostics = discover_runs([temp_dir])
        assert records == []
        assert len(diagnostics["no_seed"]) == 1

    def test_legacy_runs_are_tagged_when_included(self, temp_dir):
        """Opting in marks them with a sentinel rather than a real seed."""
        make_run(temp_dir, "resnet", "ptbxl", None, write_metadata=False, name="legacy-run")
        records, _ = discover_runs([temp_dir], include_legacy=True)
        assert [record.seed for record in records] == [LEGACY_SEED]

    def test_duplicates_collapse_to_the_newest(self, temp_dir):
        """A re-run supersedes the run it replaced."""
        make_run(temp_dir, "resnet", "ptbxl", 42, f1=0.60, name="old")
        make_run(temp_dir, "resnet", "ptbxl", 42, f1=0.80, name="new")
        records, diagnostics = discover_runs([temp_dir])
        assert len(records) == 1
        assert len(diagnostics["duplicate"]) == 1

    def test_duplicates_can_be_fatal(self, temp_dir):
        """Strict mode refuses to guess which run counts."""
        make_run(temp_dir, "resnet", "ptbxl", 42, name="one")
        make_run(temp_dir, "resnet", "ptbxl", 42, name="two")
        with pytest.raises(ValueError, match="Duplicate run"):
            discover_runs([temp_dir], on_duplicate="error")

    def test_recursive_search_finds_nested_runs(self, temp_dir):
        """Archived sweeps nested a level down are still discoverable."""
        make_run(temp_dir / "archive", "resnet", "ptbxl", 42)
        assert discover_runs([temp_dir])[0] == []
        assert len(discover_runs([temp_dir], recursive=True)[0]) == 1

    def test_unknown_duplicate_policy_raises(self, temp_dir):
        """An unrecognised policy is rejected before any work."""
        with pytest.raises(ValueError, match="Unknown on_duplicate"):
            discover_runs([temp_dir], on_duplicate="whatever")


class TestReconcileAveraging:
    """Test that incommensurable scores are never pooled."""

    def test_consistent_modes_are_untouched(self, temp_dir):
        """Agreeing runs pass through unchanged."""
        for seed in (42, 43):
            make_run(temp_dir, "resnet", "af-classification", seed)
        frame = runs_to_frame(discover_runs([temp_dir])[0])
        kept, conflicts = reconcile_averaging(frame)
        assert conflicts.empty
        assert len(kept) == len(frame)

    def test_minority_mode_is_dropped_with_a_warning(self, temp_dir):
        """One unlucky split must not blank four sound runs."""
        for seed in (42, 43, 44):
            make_run(temp_dir, "resnet", "unified-af", seed, average="macro")
        make_run(temp_dir, "resnet", "unified-af", 45, average="binary")
        frame = runs_to_frame(discover_runs([temp_dir])[0])
        with pytest.warns(RuntimeWarning, match="mixes averaging modes"):
            kept, conflicts = reconcile_averaging(frame)
        assert set(kept["averaging"]) == {"macro"}
        assert int(conflicts.iloc[0]["dropped_runs"]) == 1


class TestAggregateRuns:
    """Test the across-seed summary table."""

    def test_seeds_are_listed_and_counted(self, temp_dir):
        """Every contributing seed is recoverable from the output."""
        for seed, f1 in [(42, 0.70), (43, 0.72), (44, 0.68)]:
            make_run(temp_dir, "resnet", "ptbxl", seed, f1=f1, multi_label=True)
        aggregated = aggregate_runs(runs_to_frame(discover_runs([temp_dir])[0]))
        row = aggregated[aggregated["metric"] == "f1"].iloc[0]
        assert row["n"] == 3
        assert row["seeds"] == "42,43,44"
        assert row["mean"] == pytest.approx(0.70)

    def test_absent_metric_produces_no_row(self, temp_dir):
        """The multi-label schema has no mcc, so no mcc row appears."""
        make_run(temp_dir, "resnet", "ptbxl", 42, multi_label=True)
        aggregated = aggregate_runs(runs_to_frame(discover_runs([temp_dir])[0]))
        assert "mcc" not in set(aggregated["metric"])

    def test_single_run_has_no_interval(self, temp_dir):
        """One seed gives a mean but no dispersion."""
        make_run(temp_dir, "resnet", "ptbxl", 42, multi_label=True)
        aggregated = aggregate_runs(runs_to_frame(discover_runs([temp_dir])[0]))
        row = aggregated[aggregated["metric"] == "f1"].iloc[0]
        assert row["n"] == 1
        assert math.isnan(row["lower_ci"])

    def test_empty_input_yields_an_empty_table(self):
        """Nothing discovered is not an error."""
        assert aggregate_runs(runs_to_frame([])).empty
