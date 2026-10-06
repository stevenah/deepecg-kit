"""
Run multiple training experiments defined in a YAML config file.

Each experiment is executed as a separate `deepecg train` subprocess,
identical to running the CLI manually. A `seeds:` list expands every
experiment into one run per seed, so architecture rankings can be reported
with variance rather than from a single run. Stops on the first failure
unless --continue-on-error; use --resume to continue from where it left off.

Usage:
    python scripts/run_experiments.py                        # uses experiments.yaml
    python scripts/run_experiments.py --config my.yaml       # custom config
    python scripts/run_experiments.py --dry-run              # print commands without running
    python scripts/run_experiments.py --seeds 42 43 44       # override the configured seeds
    python scripts/run_experiments.py --only resnet:ptbxl    # restrict to one pairing
    python scripts/run_experiments.py --limit 2              # attempt only the first two runs
    python scripts/run_experiments.py --resume               # skip already-completed runs
    python scripts/run_experiments.py --continue-on-error    # keep going past a failure
    python scripts/run_experiments.py --reset                # clear saved progress
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from deepecgkit.analysis.aggregation import (
    METADATA_FILENAME,
    METRICS_FILENAME,
    SPLIT_INFO_FILENAME,
)
from deepecgkit.cli.defaults import TRAIN_DEFAULTS

DEFAULTS = dict(TRAIN_DEFAULTS)

PARAM_KEYS = set(DEFAULTS.keys()) | {
    "data_dir",
    "output_dir",
    "weights",
    "force_download",
    "multi_label",
    "sampling_rate",
    "segment_duration",
    "binary_classification",
    "normalization",
}

SEEDS_KEY = "seeds"
RUNS_DIR_DEFAULT = "runs"
STATE_VERSION = 2
MAX_SEED = 2**32 - 1
FINGERPRINT_LENGTH = 8


def validate_seeds(value: Any, source: str) -> list[int]:
    """Validate and normalise a configured seed list.

    Seeds are never coerced. A scalar is rejected rather than wrapped and
    duplicates are rejected rather than silently collapsed, because both
    usually mean the config says something other than its author intended.

    Args:
        value: The raw value parsed from YAML or the command line.
        source: Human-readable origin, used in error messages.

    Returns:
        The validated seeds in their configured order.

    Raises:
        ValueError: If the value is not a list of distinct in-range integers.
    """
    if not isinstance(value, list | tuple):
        raise ValueError(f"{source}: seeds must be a list, got {type(value).__name__}")
    seeds = list(value)
    if not seeds:
        raise ValueError(f"{source}: seeds must not be empty")

    validated: list[int] = []
    for entry in seeds:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError(
                f"{source}: seeds must be integers, got {entry!r} ({type(entry).__name__})"
            )
        if entry < 0 or entry > MAX_SEED:
            raise ValueError(f"{source}: seed {entry} is outside 0..{MAX_SEED}")
        validated.append(int(entry))

    duplicates = sorted({seed for seed in validated if validated.count(seed) > 1})
    if duplicates:
        raise ValueError(f"{source}: duplicate seeds {duplicates}")
    return validated


def resolve_seeds(
    experiment: dict[str, Any],
    file_defaults: dict[str, Any],
    resolved_seed: int,
    seeds_override: Sequence[int] | None = None,
) -> list[int]:
    """Choose the seed list for one experiment.

    Precedence runs from the command line, through a per-experiment key, to a
    defaults key, and finally to the single seed already in force. That last
    rung is derived rather than a literal 42, so a config pinning seed: 7
    still produces exactly one run at seed 7.

    Args:
        experiment: The raw experiment mapping from the config.
        file_defaults: The config's defaults mapping.
        resolved_seed: The scalar seed already merged for this experiment.
        seeds_override: Seeds supplied on the command line.

    Returns:
        The seeds this experiment expands over.

    Raises:
        ValueError: If a configured seed list is invalid.
    """
    if seeds_override is not None:
        return validate_seeds(list(seeds_override), "--seeds")
    if SEEDS_KEY in experiment:
        label = f"experiment {experiment.get('model')}/{experiment.get('dataset')}"
        return validate_seeds(experiment[SEEDS_KEY], label)
    if SEEDS_KEY in file_defaults:
        return validate_seeds(file_defaults[SEEDS_KEY], "defaults")
    return [int(resolved_seed)]


def load_experiments(
    config_path: str, seeds_override: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """Read the config into experiment mappings carrying a resolved seed list.

    Args:
        config_path: Path to the YAML config.
        seeds_override: Seeds supplied on the command line.

    Returns:
        One mapping per experiment, each with a "seeds" list.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    file_defaults = config.get("defaults", {}) or {}
    merged_defaults = {**DEFAULTS, **{k: v for k, v in file_defaults.items() if k != SEEDS_KEY}}

    experiments = []
    for exp in config.get("experiments", []) or []:
        model = exp.get("model")
        dataset = exp.get("dataset")
        if not model or not dataset:
            print(f"Skipping experiment missing model or dataset: {exp}", file=sys.stderr)
            continue

        params = dict(merged_defaults)
        for key in PARAM_KEYS:
            if key in exp:
                params[key] = exp[key]

        seeds = resolve_seeds(exp, file_defaults, params["seed"], seeds_override)
        experiments.append({"model": model, "dataset": dataset, SEEDS_KEY: seeds, **params})

    return experiments


def params_fingerprint(experiment: dict[str, Any]) -> str:
    """Stable short hash of everything that affects a run except its seed.

    Args:
        experiment: An experiment mapping.

    Returns:
        The leading hex characters of the digest.
    """
    ignored = {SEEDS_KEY, "seed", "output_dir"}
    material = {k: v for k, v in experiment.items() if k not in ignored}
    encoded = json.dumps(material, sort_keys=True, default=str).encode()
    return hashlib.md5(encoded).hexdigest()[:FINGERPRINT_LENGTH]


def run_key(model: str, dataset: str, seed: int) -> str:
    """Build the stable identifier for one expanded run.

    Args:
        model: Model name.
        dataset: Dataset name.
        seed: Seed for this run.

    Returns:
        The key used in the progress cache.
    """
    return f"{model}|{dataset}|{seed}"


def _output_dir_for(experiment: dict[str, Any], seed: int, n_seeds: int, runs_dir: str) -> str:
    """Choose the run directory for one expanded run.

    An explicit output_dir is left untouched when the experiment has a single
    seed, so existing configs keep their paths. With several seeds a sibling
    suffix is appended rather than a nested subdirectory, because nesting
    defeats recovery of the seed from the directory name.

    Args:
        experiment: The experiment mapping.
        seed: Seed for this run.
        n_seeds: How many seeds this experiment expands over.
        runs_dir: Root for generated run directories.

    Returns:
        The run directory as a string.
    """
    explicit = experiment.get("output_dir")
    if explicit:
        return str(explicit) if n_seeds == 1 else f"{explicit}-seed{seed}"
    stem = f"{experiment['model']}-{experiment['dataset']}-seed{seed}"
    return str(Path(runs_dir) / stem)


def expand_runs(
    experiments: Sequence[dict[str, Any]],
    runs_dir: str = RUNS_DIR_DEFAULT,
    only: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand each experiment into one run per seed, experiment-major.

    The "seeds" key is removed and a scalar "seed" set, so a run mapping is
    distinguishable from an experiment mapping by shape alone.

    Args:
        experiments: Experiment mappings from :func:`load_experiments`.
        runs_dir: Root for generated run directories.
        only: Restrict to "model:dataset" pairings.

    Returns:
        One mapping per run.

    Raises:
        ValueError: If two runs would still share an output directory.
    """
    wanted = set(only) if only else None
    runs: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        pairing = f"{experiment['model']}:{experiment['dataset']}"
        if wanted is not None and pairing not in wanted:
            continue
        seeds = experiment[SEEDS_KEY]
        fingerprint = params_fingerprint(experiment)
        for seed in seeds:
            run = {key: value for key, value in experiment.items() if key != SEEDS_KEY}
            run["seed"] = seed
            run["experiment_index"] = index
            run["params_fingerprint"] = fingerprint
            run["output_dir"] = _output_dir_for(experiment, seed, len(seeds), runs_dir)
            run["key"] = run_key(experiment["model"], experiment["dataset"], seed)
            runs.append(run)

    _disambiguate_output_dirs(runs)
    return runs


def _disambiguate_output_dirs(runs: list[dict[str, Any]]) -> None:
    """Append a fingerprint wherever two distinct runs would collide.

    Two experiments differing only in a hyperparameter share a model, dataset
    and seed, so their generated directories coincide. The fingerprint
    separates them instead of letting one overwrite the other.

    Args:
        runs: Expanded runs, modified in place.

    Raises:
        ValueError: If a collision survives disambiguation.
    """
    seen: dict[str, int] = {}
    for run in runs:
        seen[run["output_dir"]] = seen.get(run["output_dir"], 0) + 1

    for run in runs:
        if seen[run["output_dir"]] > 1:
            run["output_dir"] = f"{run['output_dir']}-{run['params_fingerprint']}"
            run["key"] = f"{run['key']}|{run['params_fingerprint']}"

    final: set[str] = set()
    for run in runs:
        if run["output_dir"] in final:
            raise ValueError(
                f"Two runs would share the output directory {run['output_dir']}; "
                "give one of them an explicit output_dir"
            )
        final.add(run["output_dir"])


def run_is_complete(output_dir: str) -> bool:
    """Whether a directory holds a successfully finished run.

    The run directory is the source of truth. Both the sidecar recording a
    zero exit status and the metrics file the aggregator reads must be
    present, which is the same contract the aggregator applies.

    Args:
        output_dir: Candidate run directory.

    Returns:
        True when the run completed and produced metrics.
    """
    directory = Path(output_dir)
    if not (directory / METRICS_FILENAME).is_file():
        return False
    metadata_path = directory / METADATA_FILENAME
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (ValueError, OSError):
        return False
    return isinstance(metadata, dict) and metadata.get("returncode") == 0


def write_run_metadata(run: dict[str, Any], returncode: int, duration: float) -> None:
    """Record what produced a run directory, next to its metrics.

    Args:
        run: The expanded run mapping.
        returncode: Exit status of the training subprocess.
        duration: Wall-clock seconds the run took.
    """
    directory = Path(run["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": run["key"],
        "model": run["model"],
        "dataset": run["dataset"],
        "seed": run["seed"],
        "experiment_index": run["experiment_index"],
        "params_fingerprint": run["params_fingerprint"],
        "output_dir": run["output_dir"],
        "returncode": returncode,
        "duration_seconds": round(duration, 3),
        "command": build_command(run),
    }
    # Merge into what `deepecg train` wrote for the same run rather than
    # clobbering it.
    metadata_path = directory / METADATA_FILENAME
    existing: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except (ValueError, OSError):
            pass
    existing.update(payload)
    metadata_path.write_text(json.dumps(existing, indent=2) + "\n")


def clear_stale_outputs(output_dir: str) -> None:
    """Remove the artefacts that would make a re-run look successful.

    Args:
        output_dir: Run directory about to be reused.
    """
    directory = Path(output_dir)
    for name in (METRICS_FILENAME, METADATA_FILENAME, SPLIT_INFO_FILENAME, "predictions.txt"):
        path = directory / name
        if path.is_file():
            path.unlink()


def _state_path(config_path: str) -> Path:
    """Deterministic state file path derived from the config file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        The progress cache path, alongside the config.
    """
    tag = hashlib.md5(str(Path(config_path).resolve()).encode()).hexdigest()[:8]
    return Path(config_path).resolve().parent / f".experiments_state_{tag}.json"


def _load_state(config_path: str, runs_dir: str = RUNS_DIR_DEFAULT) -> dict[str, dict[str, Any]]:
    """Load the progress cache keyed by model, dataset and seed.

    Args:
        config_path: Path to the YAML config.
        runs_dir: Root the caller intends to write into, checked for drift.

    Returns:
        Completed run records keyed by run key, empty when no state exists.

    Raises:
        ValueError: If the file uses the pre-seed format.
    """
    path = _state_path(config_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())

    completed = data.get("completed", [])
    is_v1 = data.get("version", 1) < STATE_VERSION or any(
        isinstance(entry, int) for entry in completed
    )
    if is_v1:
        raise ValueError(
            f"Progress state at {path} uses the old format (v1), which does not record "
            "seeds and points at timestamped run directories with no metrics index; "
            "re-run with --reset."
        )

    if data.get("runs_dir") not in (None, runs_dir):
        print(
            f"Note: progress state was written for runs_dir '{data.get('runs_dir')}' but "
            f"'{runs_dir}' was requested; completion is re-checked on disk.",
            file=sys.stderr,
        )
    return {entry["key"]: entry for entry in completed if isinstance(entry, dict)}


def _save_state(
    config_path: str, completed: dict[str, dict[str, Any]], runs_dir: str = RUNS_DIR_DEFAULT
) -> None:
    """Write the progress cache.

    Args:
        config_path: Path to the YAML config.
        completed: Completed run records keyed by run key.
        runs_dir: Root the runs were written into.
    """
    path = _state_path(config_path)
    payload = {
        "version": STATE_VERSION,
        "config_path": str(Path(config_path).resolve()),
        "runs_dir": runs_dir,
        "completed": [completed[key] for key in sorted(completed)],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _clear_state(config_path: str) -> None:
    """Delete the progress cache if it exists.

    Args:
        config_path: Path to the YAML config.
    """
    path = _state_path(config_path)
    if path.exists():
        path.unlink()
        print(f"Cleared progress state: {path}")
    else:
        print("No progress state to clear.")


def build_command(exp: dict[str, Any]) -> list[str]:
    """Build the `deepecg train` command for one run.

    Args:
        exp: An expanded run mapping.

    Returns:
        The command as an argument list.
    """
    cmd = [
        "deepecg",
        "train",
        "--model",
        exp["model"],
        "--dataset",
        exp["dataset"],
        "--epochs",
        str(exp["epochs"]),
        "--batch-size",
        str(exp["batch_size"]),
        "--learning-rate",
        str(exp["learning_rate"]),
        "--val-split",
        str(exp["val_split"]),
        "--test-split",
        str(exp["test_split"]),
        "--num-workers",
        str(exp["num_workers"]),
        "--accelerator",
        str(exp["accelerator"]),
        "--devices",
        str(exp["devices"]),
        "--early-stopping-patience",
        str(exp["early_stopping_patience"]),
        "--seed",
        str(exp["seed"]),
    ]

    if exp.get("data_dir"):
        cmd.extend(["--data-dir", exp["data_dir"]])
    if exp.get("output_dir"):
        cmd.extend(["--output-dir", str(exp["output_dir"])])
    if exp.get("weights"):
        cmd.extend(["--weights", exp["weights"]])
    if exp.get("force_download"):
        cmd.append("--force-download")
    if exp.get("multi_label"):
        cmd.append("--multi-label")
    if exp.get("sampling_rate"):
        cmd.extend(["--sampling-rate", str(exp["sampling_rate"])])
    if exp.get("segment_duration"):
        cmd.extend(["--segment-duration", str(exp["segment_duration"])])
    if exp.get("binary_classification"):
        cmd.append("--binary-classification")
    if exp.get("normalization"):
        cmd.extend(["--normalization", exp["normalization"]])

    return cmd


def format_duration(seconds: float) -> str:
    """Render a duration compactly.

    Args:
        seconds: Elapsed seconds.

    Returns:
        A short human-readable string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Run multiple training experiments")
    parser.add_argument(
        "--config",
        default="experiments.yaml",
        help="Path to experiments YAML config (default: experiments.yaml)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override every configured seed list with these seeds",
    )
    parser.add_argument(
        "--runs-dir",
        default=RUNS_DIR_DEFAULT,
        help=f"Root directory for generated run directories (default: {RUNS_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="MODEL:DATASET",
        help="Restrict to these model/dataset pairings",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Attempt at most this many runs, leaving saved progress intact",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already completed",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed run instead of stopping",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run into directories that already hold a completed run",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear saved progress and exit",
    )
    return parser


def _echo_plan(runs: Sequence[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
    """Print the resolved seeds and the commands that will run.

    The seed list is always echoed. YAML resolves a zero-padded literal such
    as 042 as octal, so seeds: [042, 043] silently becomes [34, 35]; printing
    the resolved list is the only cheap way to make that visible.

    Args:
        runs: Expanded runs.
        completed: Completed run records.
    """
    seeds = sorted({run["seed"] for run in runs})
    print(f"{'=' * 60}")
    print(f"  {len(runs)} runs loaded")
    print(f"  Resolved seeds: {seeds}")
    if completed:
        print(f"  {len(completed)} already done, {len(runs) - len(completed)} remaining")
    print(f"{'=' * 60}")
    for position, run in enumerate(runs, 1):
        skip = " [SKIP]" if run["key"] in completed else ""
        print(f"  [{position}] {' '.join(build_command(run))}{skip}")
    print()


def _reconcile_with_disk(
    runs: Sequence[dict[str, Any]], completed: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Trust the run directories over the cache.

    A cached entry whose directory no longer holds metrics is dropped, and a
    directory that finished without reaching the cache is adopted, so an
    interrupted sweep neither re-trains work present on disk nor skips work
    that is missing from it.

    Args:
        runs: Expanded runs.
        completed: Cached completion records.

    Returns:
        The reconciled records.
    """
    reconciled: dict[str, dict[str, Any]] = {}
    for run in runs:
        key = run["key"]
        if run_is_complete(run["output_dir"]):
            reconciled[key] = completed.get(
                key,
                {
                    "key": key,
                    "model": run["model"],
                    "dataset": run["dataset"],
                    "seed": run["seed"],
                    "experiment_index": run["experiment_index"],
                    "output_dir": run["output_dir"],
                    "params_fingerprint": run["params_fingerprint"],
                    "duration_seconds": None,
                },
            )
        elif key in completed:
            print(
                f"Note: {key} was recorded complete but {run['output_dir']} no longer holds "
                "metrics; it will be re-run.",
                file=sys.stderr,
            )
    return reconciled


def _print_summary(results: Sequence[dict[str, Any]], total_elapsed: float) -> None:
    """Print the per-run outcome table.

    Args:
        results: Recorded outcomes.
        total_elapsed: Wall-clock seconds for the whole batch.
    """
    print(f"\n{'=' * 60}")
    print(f"  Summary  ({format_duration(total_elapsed)} total)")
    print(f"{'=' * 60}")
    for entry in results:
        if entry["status"] == "OK":
            print(f"  [PASS] {entry['label']}  ({format_duration(entry['duration'])})")
        else:
            print(f"  [{entry['status']}] {entry['label']}")

    passed = sum(1 for entry in results if entry["status"] == "OK")
    skipped = sum(1 for entry in results if entry["status"] == "SKIP")
    pending = sum(1 for entry in results if entry["status"] == "PENDING")
    failures = sum(1 for entry in results if entry["status"] == "FAIL")
    print(f"\n  {passed} ran, {skipped} skipped, {pending} not attempted, {failures} failed.")


def _execute(
    runs: Sequence[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    """Run the expanded list, recording progress as it goes.

    Args:
        runs: Expanded runs.
        completed: Completion records, updated in place.
        args: Parsed command-line arguments.

    Returns:
        Process exit status.
    """
    results: list[dict[str, Any]] = []
    total_start = time.time()
    attempted = 0
    failed = 0

    for position, run in enumerate(runs, 1):
        label = f"[{position}/{len(runs)}] {run['model']} x {run['dataset']} seed {run['seed']}"

        if run["key"] in completed:
            results.append({"label": label, "status": "SKIP"})
            print(f"\n  {label} -> SKIP (already completed)")
            continue

        if args.limit is not None and attempted >= args.limit:
            results.append({"label": label, "status": "PENDING"})
            continue

        if run_is_complete(run["output_dir"]) and not args.overwrite:
            print(
                f"\n  {label} -> REFUSED: {run['output_dir']} already holds a completed run. "
                "Pass --overwrite to replace it, or --resume to skip it.",
                file=sys.stderr,
            )
            return 1

        if args.overwrite:
            clear_stale_outputs(run["output_dir"])

        cmd = build_command(run)
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        start = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start
        attempted += 1
        write_run_metadata(run, result.returncode, elapsed)

        if result.returncode == 0:
            completed[run["key"]] = {
                "key": run["key"],
                "model": run["model"],
                "dataset": run["dataset"],
                "seed": run["seed"],
                "experiment_index": run["experiment_index"],
                "output_dir": run["output_dir"],
                "params_fingerprint": run["params_fingerprint"],
                "duration_seconds": round(elapsed, 3),
            }
            _save_state(args.config, completed, runs_dir=args.runs_dir)
            results.append({"label": label, "status": "OK", "duration": elapsed})
            print(f"\n  {label} -> OK ({format_duration(elapsed)})")
            continue

        failed += 1
        results.append({"label": label, "status": "FAIL", "duration": elapsed})
        print(f"\n  {label} -> FAIL ({format_duration(elapsed)})")
        if not args.continue_on_error:
            print("\n  Stopping. Fix the issue and re-run with --resume to continue.")
            print(f"  Progress saved: {_state_path(args.config)}")
            return 1

    _print_summary(results, time.time() - total_start)

    fully_attempted = args.limit is None and all(entry["status"] != "PENDING" for entry in results)
    if fully_attempted and failed == 0:
        _clear_state(args.config)
    elif failed:
        print(f"\n  {failed} run(s) failed. Re-run with --resume to retry them.")

    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, defaulting to sys.argv.

    Returns:
        Process exit status.
    """
    args = _build_parser().parse_args(argv)

    if args.reset:
        _clear_state(args.config)
        return 0

    try:
        experiments = load_experiments(args.config, seeds_override=args.seeds)
    except ValueError as error:
        print(f"Invalid configuration: {error}", file=sys.stderr)
        return 1

    if not experiments:
        print("No experiments defined in config.")
        return 0

    try:
        runs = expand_runs(experiments, runs_dir=args.runs_dir, only=args.only)
    except ValueError as error:
        print(f"Cannot expand experiments: {error}", file=sys.stderr)
        return 1

    if not runs:
        print("No runs selected.")
        return 0

    completed: dict[str, dict[str, Any]] = {}
    if args.resume:
        try:
            completed = _load_state(args.config, runs_dir=args.runs_dir)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
        completed = _reconcile_with_disk(runs, completed)

    _echo_plan(runs, completed)

    if args.dry_run:
        print("Dry run complete. No training started.")
        return 0

    return _execute(runs, completed, args)


if __name__ == "__main__":
    sys.exit(main())
