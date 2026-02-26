"""
Run multiple training experiments defined in a YAML config file.

Each experiment is executed as a separate `deepecg train` subprocess,
identical to running the CLI manually.

Usage:
    python scripts/run_experiments.py                       # uses experiments.yaml
    python scripts/run_experiments.py --config my.yaml      # custom config
    python scripts/run_experiments.py --dry-run              # print commands without running
"""

import argparse
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
    args = parser.parse_args()

    experiments = load_experiments(args.config)

    if not experiments:
        print("No experiments defined in config.")
        return

    print(f"{'=' * 60}")
    print(f"  {len(experiments)} experiments loaded")
    print(f"{'=' * 60}")

    for i, exp in enumerate(experiments, 1):
        cmd = build_command(exp)
        print(f"  [{i}] {' '.join(cmd)}")

    print()

    if args.dry_run:
        print("Dry run complete. No training started.")
        return

    results = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        label = f"[{i}/{len(experiments)}] {exp['model']} x {exp['dataset']}"
        cmd = build_command(exp)

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        start = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start

        status = "OK" if result.returncode == 0 else "FAIL"
        results.append({"label": label, "status": status, "duration": elapsed})

        print(f"\n  {label} -> {status} ({format_duration(elapsed)})")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"  Summary  ({format_duration(total_elapsed)} total)")
    print(f"{'=' * 60}")

    for r in results:
        marker = "PASS" if r["status"] == "OK" else "FAIL"
        print(f"  [{marker}] {r['label']}  ({format_duration(r['duration'])})")

    failed = sum(1 for r in results if r["status"] != "OK")
    if failed:
        print(f"\n  {failed}/{len(results)} experiments failed.")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} experiments passed.")


if __name__ == "__main__":
    main()
