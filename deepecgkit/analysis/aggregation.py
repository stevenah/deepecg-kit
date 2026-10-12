"""Discovery and aggregation of finished training runs.

A run directory is usable when it holds ``classification_metrics.csv``. Two
schemas are written by :mod:`deepecgkit.evaluation.report` depending on whether
the task is multi-label, and they share only precision, recall, f1, auroc and
the calibration columns, so the schema and the averaging mode travel with every
record.

Every metric name is looked up defensively, so runs produced before a metric
existed stay readable and simply contribute nothing to it. The
``calibration_bins`` column is deliberately not a metric: averaging a bin count
across seeds would be meaningless.
"""

import json
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from deepecgkit.analysis.statistics import CONFIDENCE_LEVEL, summarize_scores
from deepecgkit.evaluation.metrics import CALIBRATION_REPORT_KEYS

METRICS_FILENAME = "classification_metrics.csv"
METADATA_FILENAME = "run_metadata.json"
SPLIT_INFO_FILENAME = "split_info.json"

CALIBRATION_METRICS = CALIBRATION_REPORT_KEYS

SINGLE_LABEL_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "mcc",
    *CALIBRATION_METRICS,
)
MULTI_LABEL_METRICS = (
    "exact_match_accuracy",
    "label_accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    *CALIBRATION_METRICS,
)

AVERAGING_MACRO_LABEL = "macro-label"
AVERAGING_UNKNOWN = "unknown"
LEGACY_SEED = -1

SEED_DIR_PATTERN = re.compile(r"(?:^|[-_/])seed(\d+)$")

ON_DUPLICATE_NEWEST = "newest"
ON_DUPLICATE_ERROR = "error"
ON_DUPLICATE_ALL = "all"


class RunRecord(NamedTuple):
    """One finished training run and the metrics it produced."""

    run_dir: Path
    model: str
    dataset: str
    seed: int | None
    multi_label: bool
    averaging: str
    metrics: dict[str, float]
    n_train: int | None
    n_val: int | None
    n_test: int | None
    n_test_groups: int | None
    mtime: float
    returncode: int | None = None
    fingerprint: str | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None when it is absent or malformed.

    Args:
        path: File to read.

    Returns:
        The decoded mapping, or None.
    """
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _seed_from_directory(run_dir: Path) -> int | None:
    """Recover a seed from a directory name ending in a seed suffix.

    Args:
        run_dir: Run directory.

    Returns:
        The seed, or None when the name carries no suffix.
    """
    match = SEED_DIR_PATTERN.search(run_dir.name)
    return int(match.group(1)) if match else None


def read_run_metrics(run_dir: Path) -> RunRecord | None:
    """Read one run directory into a record.

    Model and dataset come from the metrics CSV, which is authoritative. The
    seed comes from ``run_metadata.json`` when present and falls back to a
    ``seed<N>`` suffix on the directory name, so runs produced before seeds
    were recorded remain readable.

    Args:
        run_dir: Directory holding classification_metrics.csv.

    Returns:
        A populated record, or None when the directory holds no usable metrics.
    """
    metrics_path = run_dir / METRICS_FILENAME
    if not metrics_path.is_file():
        return None
    try:
        frame = pd.read_csv(metrics_path)
    except (ValueError, OSError, pd.errors.ParserError):
        return None
    if frame.empty:
        return None

    row = frame.iloc[0]
    columns = set(frame.columns)
    if "model" not in columns or "dataset" not in columns:
        return None

    multi_label = "exact_match_accuracy" in columns
    metric_names = MULTI_LABEL_METRICS if multi_label else SINGLE_LABEL_METRICS
    metrics: dict[str, float] = {}
    for name in metric_names:
        if name in columns:
            metrics[name] = float(row[name])

    if multi_label:
        averaging = AVERAGING_MACRO_LABEL
    elif "average" in columns and isinstance(row["average"], str):
        averaging = str(row["average"])
    else:
        averaging = AVERAGING_UNKNOWN

    metadata = _read_json(run_dir / METADATA_FILENAME) or {}
    seed = metadata.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        seed = _seed_from_directory(run_dir)
    returncode = metadata.get("returncode")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        returncode = None
    fingerprint = metadata.get("params_fingerprint")
    if not isinstance(fingerprint, str):
        fingerprint = None

    split_info = _read_json(run_dir / SPLIT_INFO_FILENAME) or {}

    def _size(key: str) -> int | None:
        value = split_info.get(key)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

    return RunRecord(
        run_dir=run_dir,
        model=str(row["model"]),
        dataset=str(row["dataset"]),
        seed=seed,
        multi_label=multi_label,
        averaging=averaging,
        metrics=metrics,
        n_train=_size("n_train"),
        n_val=_size("n_val"),
        n_test=_size("n_test"),
        n_test_groups=_size("n_test_groups"),
        mtime=metrics_path.stat().st_mtime,
        returncode=returncode,
        fingerprint=fingerprint,
    )


def discover_runs(
    roots: Sequence[Path],
    recursive: bool = False,
    include_legacy: bool = False,
    on_duplicate: str = ON_DUPLICATE_NEWEST,
    include_failed: bool = False,
) -> tuple[list[RunRecord], dict[str, list[str]]]:
    """Collect usable runs beneath the given roots.

    A run whose ``run_metadata.json`` records a non-zero ``returncode`` (the
    batch runner writes the training subprocess's exit status there) failed
    after writing its metrics and is set aside under the "failed" diagnostic
    unless ``include_failed`` is given. Runs without a recorded status are
    kept.

    Two runs count as duplicates when they share dataset, model, seed and, when
    recorded, the batch runner's ``params_fingerprint``; runs of one pairing
    and seed with different hyperparameters are therefore kept apart rather
    than resolved by modification time.

    Args:
        roots: Directories to search.
        recursive: Search at any depth rather than only immediate children.
        include_legacy: Keep runs whose seed could not be determined, tagging
            them with a sentinel seed so they are visible but never silently
            pooled with seeded runs.
        on_duplicate: One of "newest", "error" or "all", controlling what
            happens when the same model, dataset and seed appear twice.
        include_failed: Keep runs whose recorded exit status is non-zero.

    Returns:
        The records and a diagnostics mapping from reason to directory names.

    Raises:
        ValueError: If on_duplicate is unrecognised, or is "error" and a
            duplicate is found.
    """
    if on_duplicate not in (ON_DUPLICATE_NEWEST, ON_DUPLICATE_ERROR, ON_DUPLICATE_ALL):
        raise ValueError(f"Unknown on_duplicate: {on_duplicate}")

    diagnostics: dict[str, list[str]] = {
        "unreadable": [],
        "no_seed": [],
        "duplicate": [],
        "failed": [],
    }
    candidates: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        pattern = f"**/{METRICS_FILENAME}" if recursive else f"*/{METRICS_FILENAME}"
        candidates.extend(sorted(path.parent for path in root_path.glob(pattern)))

    records: list[RunRecord] = []
    for run_dir in candidates:
        record = read_run_metrics(run_dir)
        if record is None:
            diagnostics["unreadable"].append(str(run_dir))
            continue
        if record.returncode not in (None, 0):
            diagnostics["failed"].append(f"{run_dir} (returncode {record.returncode})")
            if not include_failed:
                continue
        if record.seed is None:
            diagnostics["no_seed"].append(str(run_dir))
            if not include_legacy:
                continue
            record = record._replace(seed=LEGACY_SEED)
        records.append(record)

    if on_duplicate == ON_DUPLICATE_ALL:
        return records, diagnostics

    kept: dict[tuple[str, str, int, str | None], RunRecord] = {}
    for record in records:
        key = (record.dataset, record.model, int(record.seed), record.fingerprint)
        previous = kept.get(key)
        if previous is None:
            kept[key] = record
            continue
        if on_duplicate == ON_DUPLICATE_ERROR:
            raise ValueError(
                f"Duplicate run for {record.model} on {record.dataset} seed {record.seed}: "
                f"{previous.run_dir} and {record.run_dir}"
            )
        newer, older = (record, previous) if record.mtime > previous.mtime else (previous, record)
        diagnostics["duplicate"].append(f"{older.run_dir} (dropped; kept {newer.run_dir})")
        kept[key] = newer

    ordered = sorted(
        kept.values(), key=lambda r: (r.dataset, r.model, int(r.seed), r.fingerprint or "")
    )
    return ordered, diagnostics


def runs_to_frame(records: Sequence[RunRecord]) -> pd.DataFrame:
    """Flatten records into one row per run and metric.

    Args:
        records: Discovered runs.

    Returns:
        A long frame with columns run_dir, model, dataset, seed, averaging,
        multi_label, n_train, n_val, n_test, n_test_groups, metric and value.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        for metric, value in record.metrics.items():
            rows.append(
                {
                    "run_dir": str(record.run_dir),
                    "model": record.model,
                    "dataset": record.dataset,
                    "seed": record.seed,
                    "averaging": record.averaging,
                    "multi_label": record.multi_label,
                    "n_train": record.n_train,
                    "n_val": record.n_val,
                    "n_test": record.n_test,
                    "n_test_groups": record.n_test_groups,
                    "metric": metric,
                    "value": value,
                }
            )
    columns = [
        "run_dir",
        "model",
        "dataset",
        "seed",
        "averaging",
        "multi_label",
        "n_train",
        "n_val",
        "n_test",
        "n_test_groups",
        "metric",
        "value",
    ]
    return pd.DataFrame(rows, columns=columns)


def reconcile_averaging(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop runs whose averaging mode disagrees with their group's majority.

    ``resolve_average`` returns "binary" whenever the test split holds two or
    fewer classes, so precision, recall and F1 are positive-class values there
    and macro elsewhere. Pooling the two would average incommensurable
    quantities. The minority is dropped rather than blanking the whole cell,
    because one unlucky split should not discard four sound runs.

    Args:
        runs: Long frame from :func:`runs_to_frame`.

    Returns:
        The surviving rows and a frame describing what was dropped.
    """
    if runs.empty:
        return runs, pd.DataFrame(columns=["dataset", "model", "kept_averaging", "dropped_runs"])

    keep_mask = pd.Series(True, index=runs.index)
    conflicts: list[dict[str, Any]] = []
    for (dataset, model), group in runs.groupby(["dataset", "model"], sort=True):
        modes = group.drop_duplicates("run_dir")["averaging"]
        if modes.nunique() <= 1:
            continue
        majority = modes.value_counts().idxmax()
        offending = group["averaging"] != majority
        keep_mask.loc[group.index[offending]] = False
        dropped = sorted(set(group.loc[offending, "run_dir"]))
        conflicts.append(
            {
                "dataset": dataset,
                "model": model,
                "kept_averaging": majority,
                "dropped_runs": len(dropped),
            }
        )
        warnings.warn(
            f"{model} on {dataset} mixes averaging modes {sorted(set(modes))}; keeping "
            f"'{majority}' and dropping {len(dropped)} run(s)",
            RuntimeWarning,
            stacklevel=2,
        )

    conflict_frame = pd.DataFrame(
        conflicts, columns=["dataset", "model", "kept_averaging", "dropped_runs"]
    )
    return runs.loc[keep_mask].reset_index(drop=True), conflict_frame


def aggregate_runs(runs: pd.DataFrame, confidence_level: float = CONFIDENCE_LEVEL) -> pd.DataFrame:
    """Summarise every metric across seeds for each dataset and model.

    Args:
        runs: Long frame from :func:`runs_to_frame`, already reconciled.
        confidence_level: Two-sided coverage for the Student-t interval.

    Returns:
        One row per dataset, model and metric with mean, std, sem, interval
        bounds, the usable count and the contributing seeds.
    """
    columns = [
        "dataset",
        "model",
        "metric",
        "averaging",
        "n_runs",
        "n",
        "mean",
        "std",
        "sem",
        "lower_ci",
        "upper_ci",
        "seeds",
    ]
    if runs.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (dataset, model, metric), group in runs.groupby(["dataset", "model", "metric"], sort=True):
        summary = summarize_scores(group["value"].tolist(), confidence_level=confidence_level)
        finite = group.loc[np.isfinite(group["value"]), "seed"]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "averaging": group["averaging"].iloc[0],
                "n_runs": len(group),
                "n": summary["n"],
                "mean": summary["mean"],
                "std": summary["std"],
                "sem": summary["sem"],
                "lower_ci": summary["lower_ci"],
                "upper_ci": summary["upper_ci"],
                "seeds": ",".join(str(int(s)) for s in sorted(finite.tolist())),
            }
        )
    return pd.DataFrame(rows, columns=columns)
