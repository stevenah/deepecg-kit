"""
Aggregate per-run test metrics across random seeds.

Reads every run directory holding a classification_metrics.csv and reports,
per model and dataset, the mean, standard deviation and 95% confidence
interval of each metric. Intervals are Student-t over seeds; with a handful of
runs a percentile bootstrap cannot place an endpoint outside the observed
values and would understate the spread.

Usage:
    python scripts/aggregate_results.py --runs-dir runs
    python scripts/aggregate_results.py --runs-dir runs --output-dir analysis --emit-latex
    python scripts/aggregate_results.py --runs-dir a b --recursive --include-legacy
    python scripts/aggregate_results.py --runs-dir runs --models resnet crnn
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from deepecgkit.analysis.aggregation import (
    ON_DUPLICATE_ALL,
    ON_DUPLICATE_ERROR,
    ON_DUPLICATE_NEWEST,
    aggregate_runs,
    discover_runs,
    reconcile_averaging,
    runs_to_frame,
)
from deepecgkit.analysis.latex import results_table, write_tables
from deepecgkit.analysis.statistics import CONFIDENCE_LEVEL


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Aggregate run metrics across seeds")
    parser.add_argument(
        "--runs-dir",
        nargs="+",
        default=["runs"],
        help="One or more directories holding run directories (default: runs)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write CSV and LaTeX output here instead of printing only",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for run directories at any depth",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include runs whose seed cannot be determined, under a sentinel seed",
    )
    parser.add_argument(
        "--on-duplicate",
        choices=[ON_DUPLICATE_NEWEST, ON_DUPLICATE_ERROR, ON_DUPLICATE_ALL],
        default=ON_DUPLICATE_NEWEST,
        help="What to do when one model, dataset and seed appears twice",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict to these models",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to these datasets",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1", "precision", "recall"],
        help="Metrics for the LaTeX tables, in column order",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=CONFIDENCE_LEVEL,
        help=f"Two-sided coverage (default: {CONFIDENCE_LEVEL})",
    )
    parser.add_argument(
        "--emit-latex",
        action="store_true",
        help="Write a booktabs table per dataset",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, defaulting to sys.argv.

    Returns:
        Process exit status.
    """
    args = _build_parser().parse_args(argv)

    try:
        records, diagnostics = discover_runs(
            [Path(root) for root in args.runs_dir],
            recursive=args.recursive,
            include_legacy=args.include_legacy,
            on_duplicate=args.on_duplicate,
        )
    except ValueError as error:
        print(f"Discovery failed: {error}", file=sys.stderr)
        return 1

    if not records:
        print(f"No usable runs found under {', '.join(args.runs_dir)}", file=sys.stderr)
        return 1

    frame = runs_to_frame(records)
    if args.models:
        frame = frame[frame["model"].isin(args.models)]
    if args.datasets:
        frame = frame[frame["dataset"].isin(args.datasets)]
    if frame.empty:
        print("No runs left after filtering.", file=sys.stderr)
        return 1

    frame, conflicts = reconcile_averaging(frame)
    aggregated = aggregate_runs(frame, confidence_level=args.confidence_level)

    for reason, entries in sorted(diagnostics.items()):
        if entries:
            print(f"  {len(entries)} run(s) {reason}")
    if not conflicts.empty:
        print("\nAveraging conflicts resolved by majority:")
        print(conflicts.to_string(index=False))

    print(f"\n{len(records)} runs across {frame['dataset'].nunique()} dataset(s)\n")
    shown = aggregated[aggregated["metric"].isin(args.metrics)]
    print(
        shown[
            ["dataset", "model", "metric", "n", "mean", "std", "lower_ci", "upper_ci", "averaging"]
        ].to_string(index=False)
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "seed_runs_long.csv", index=False)
        aggregated.to_csv(output_dir / "seed_metrics_long.csv", index=False)
        written = ["seed_runs_long.csv", "seed_metrics_long.csv"]

        if args.emit_latex:
            tables = {
                f"results_{dataset.replace('-', '_')}": results_table(
                    aggregated, dataset, metrics=args.metrics
                )
                for dataset in sorted(aggregated["dataset"].unique())
            }
            written.extend(Path(path).name for path in write_tables(tables, output_dir))

        print(f"\nWrote {len(written)} file(s) to {output_dir}:")
        for name in written:
            print(f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
