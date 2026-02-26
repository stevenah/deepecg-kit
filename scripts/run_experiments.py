"""
Run multiple training experiments defined in a YAML config file.

Each experiment is executed as a separate `deepecg train` subprocess,
identical to running the CLI manually. Stops on first failure; use
--resume to continue from where it left off.

Usage:
    python scripts/run_experiments.py                       # uses experiments.yaml
    python scripts/run_experiments.py --config my.yaml      # custom config
    python scripts/run_experiments.py --dry-run              # print commands without running
    python scripts/run_experiments.py --resume               # resume after fixing a failure
    python scripts/run_experiments.py --reset                # clear saved progress
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml  # type: ignore[import-untyped]

DEFAULTS = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "val_split": 0.2,
    "test_split": 0.1,
    "num_workers": 4,
    "accelerator": "auto",
    "devices": 1,
    "early_stopping_patience": 10,
    "seed": 42,
}

PARAM_KEYS = set(DEFAULTS.keys()) | {
    "data_dir",
    "output_dir",
    "weights",
    "force_download",
    "multi_label",
    "sampling_rate",
    "binary_classification",
    "normalization",
}


def load_experiments(config_path: str) -> list:
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    file_defaults = config.get("defaults", {})
    merged_defaults = {**DEFAULTS, **file_defaults}

    experiments = []
    for exp in config.get("experiments", []):
        model = exp.get("model")
        dataset = exp.get("dataset")
        if not model or not dataset:
            print(f"Skipping experiment missing model or dataset: {exp}", file=sys.stderr)
            continue

        params = dict(merged_defaults)
        for key in PARAM_KEYS:
            if key in exp:
                params[key] = exp[key]

        experiments.append({"model": model, "dataset": dataset, **params})

    return experiments


def _state_path(config_path: str) -> Path:
    """Deterministic state file path derived from the config file."""
    tag = hashlib.md5(str(Path(config_path).resolve()).encode()).hexdigest()[:8]
    return Path(config_path).resolve().parent / f".experiments_state_{tag}.json"


def _load_state(config_path: str) -> set[int]:
    """Return set of 0-based indices that already completed successfully."""
    path = _state_path(config_path)
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return set(data.get("completed", []))


def _save_state(config_path: str, completed: set[int]) -> None:
    path = _state_path(config_path)
    path.write_text(json.dumps({"completed": sorted(completed)}, indent=2) + "\n")


def _clear_state(config_path: str) -> None:
    path = _state_path(config_path)
    if path.exists():
        path.unlink()
        print(f"Cleared progress state: {path}")
    else:
        print("No progress state to clear.")


def build_command(exp: dict) -> list[str]:
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
        cmd.extend(["--output-dir", exp["output_dir"]])
    if exp.get("weights"):
        cmd.extend(["--weights", exp["weights"]])
    if exp.get("force_download"):
        cmd.append("--force-download")
    if exp.get("multi_label"):
        cmd.append("--multi-label")
    if exp.get("sampling_rate"):
        cmd.extend(["--sampling-rate", str(exp["sampling_rate"])])
    if exp.get("binary_classification"):
        cmd.append("--binary-classification")
    if exp.get("normalization"):
        cmd.extend(["--normalization", exp["normalization"]])

    return cmd


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def main():
    parser = argparse.ArgumentParser(description="Run multiple training experiments")
    parser.add_argument(
        "--config",
        default="experiments.yaml",
        help="Path to experiments YAML config (default: experiments.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last failure, skipping completed experiments",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear saved progress and exit",
    )
    args = parser.parse_args()

    if args.reset:
        _clear_state(args.config)
        return

    experiments = load_experiments(args.config)

    if not experiments:
        print("No experiments defined in config.")
        return

    completed = _load_state(args.config) if args.resume else set()
    remaining = sum(1 for i in range(len(experiments)) if i not in completed)

    print(f"{'=' * 60}")
    print(f"  {len(experiments)} experiments loaded", end="")
    if completed:
        print(f" ({len(completed)} already done, {remaining} remaining)")
    else:
        print()
    print(f"{'=' * 60}")

    for i, exp in enumerate(experiments, 1):
        cmd = build_command(exp)
        skip = " [SKIP]" if (i - 1) in completed else ""
        print(f"  [{i}] {' '.join(cmd)}{skip}")

    print()

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    results = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        idx = i - 1
        label = f"[{i}/{len(experiments)}] {exp['model']} x {exp['dataset']}"

        if idx in completed:
            results.append({"label": label, "status": "OK", "duration": 0, "skipped": True})
            print(f"\n  {label} -> SKIP (already completed)")
            continue

        cmd = build_command(exp)

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        start = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start

        if result.returncode == 0:
            completed.add(idx)
            _save_state(args.config, completed)
            results.append({"label": label, "status": "OK", "duration": elapsed})
            print(f"\n  {label} -> OK ({format_duration(elapsed)})")
        else:
            results.append({"label": label, "status": "FAIL", "duration": elapsed})
            print(f"\n  {label} -> FAIL ({format_duration(elapsed)})")
            print("\n  Stopping. Fix the issue and re-run with --resume to continue.")
            print(f"  Progress saved: {_state_path(args.config)}")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  Summary  ({format_duration(total_elapsed)} total)")
    print(f"{'=' * 60}")

    for r in results:
        if r.get("skipped"):
            print(f"  [SKIP] {r['label']}")
        else:
            print(f"  [PASS] {r['label']}  ({format_duration(r['duration'])})")

    print(f"\n  All {len(experiments)} experiments passed.")
    _clear_state(args.config)


if __name__ == "__main__":
    main()
