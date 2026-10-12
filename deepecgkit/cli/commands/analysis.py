"""Aggregate and rank commands: multi-seed analysis over run directories.

These wrap deepecgkit.analysis for the run directories that `deepecg train`
and scripts/run_experiments.py produce. The tabular results are printed to
stdout as the product output, matching the standalone scripts these commands
replaced (which remain as thin shims).
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from deepecgkit.analysis.aggregation import (
    ON_DUPLICATE_ALL,
    ON_DUPLICATE_ERROR,
    ON_DUPLICATE_NEWEST,
    aggregate_runs,
    discover_runs,
    reconcile_averaging,
    runs_to_frame,
)
from deepecgkit.analysis.latex import (
    flip_survival_table,
    rank_distribution_table,
    results_table,
    tau_table,
    write_tables,
)
from deepecgkit.analysis.ranking import (
    DEFAULT_RANK_METRIC,
    MIN_TEST_GROUPS,
    all_flip_tests,
    evaluate_rank_claim,
    rank_distribution,
    rank_within_seed,
    tau_summary,
)
from deepecgkit.analysis.statistics import CONFIDENCE_LEVEL, NADEAU_BENGIO, NO_CORRECTION


def add_discovery_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the run-discovery arguments both analysis commands share.

    Args:
        parser: The subcommand's parser
    """
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
    parser.add_argument("--models", nargs="+", default=None, help="Restrict to these models")
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict to these datasets")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=CONFIDENCE_LEVEL,
        help=f"Two-sided coverage (default: {CONFIDENCE_LEVEL})",
    )
    parser.add_argument(
        "--emit-latex",
        action="store_true",
        help="Write booktabs tables",
    )


def _discover_frame(
    runs_dir: list[str],
    recursive: bool,
    include_legacy: bool,
    on_duplicate: str,
    models: list[str] | None,
    datasets: list[str] | None,
) -> tuple[list, pd.DataFrame, dict] | None:
    """Discover run records and turn them into the filtered long frame.

    Returns:
        Tuple of (records, frame, diagnostics), or None with the failure
        already printed to stderr.
    """
    try:
        records, diagnostics = discover_runs(
            [Path(root) for root in runs_dir],
            recursive=recursive,
            include_legacy=include_legacy,
            on_duplicate=on_duplicate,
        )
    except ValueError as error:
        print(f"Discovery failed: {error}", file=sys.stderr)
        return None

    if not records:
        print(f"No usable runs found under {', '.join(runs_dir)}", file=sys.stderr)
        return None

    frame = runs_to_frame(records)
    if models:
        frame = frame[frame["model"].isin(models)]
    if datasets:
        frame = frame[frame["dataset"].isin(datasets)]
    if frame.empty:
        print("No runs left after filtering.", file=sys.stderr)
        return None
    return records, frame, diagnostics


def aggregate(
    runs_dir: list[str],
    output_dir: str | None = None,
    recursive: bool = False,
    include_legacy: bool = False,
    on_duplicate: str = ON_DUPLICATE_NEWEST,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    metrics: list[str] | None = None,
    confidence_level: float = CONFIDENCE_LEVEL,
    emit_latex: bool = False,
) -> int:
    """Aggregate per-run test metrics across random seeds.

    Reports, per model and dataset, the mean, standard deviation and
    confidence interval of each metric. Intervals are Student-t over seeds.

    Returns:
        Zero on success, one on failure.
    """
    metrics = list(metrics) if metrics else ["f1", "precision", "recall"]
    discovered = _discover_frame(
        runs_dir, recursive, include_legacy, on_duplicate, models, datasets
    )
    if discovered is None:
        return 1
    records, frame, diagnostics = discovered

    frame, conflicts = reconcile_averaging(frame)
    aggregated = aggregate_runs(frame, confidence_level=confidence_level)

    for reason, entries in sorted(diagnostics.items()):
        if entries:
            print(f"  {len(entries)} run(s) {reason}")
    if not conflicts.empty:
        print("\nAveraging conflicts resolved by majority:")
        print(conflicts.to_string(index=False))

    print(f"\n{len(records)} runs across {frame['dataset'].nunique()} dataset(s)\n")
    shown = aggregated[aggregated["metric"].isin(metrics)]
    print(
        shown[
            ["dataset", "model", "metric", "n", "mean", "std", "lower_ci", "upper_ci", "averaging"]
        ].to_string(index=False)
    )

    if output_dir:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target / "seed_runs_long.csv", index=False)
        aggregated.to_csv(target / "seed_metrics_long.csv", index=False)
        written = ["seed_runs_long.csv", "seed_metrics_long.csv"]

        if emit_latex:
            tables = {
                f"results_{dataset.replace('-', '_')}": results_table(
                    aggregated, dataset, metrics=metrics, confidence_level=confidence_level
                )
                for dataset in sorted(aggregated["dataset"].unique())
            }
            written.extend(Path(path).name for path in write_tables(tables, target))

        print(f"\nWrote {len(written)} file(s) to {target}:")
        for name in written:
            print(f"  {name}")

    return 0


def _print_claim(result: dict) -> None:
    """Print one decomposed rank claim."""
    print(
        f"\nClaim: {result['model']} strongest on {result['dataset_best']}, "
        f"weakest on {result['dataset_worst']}"
    )
    print(f"  Overall: {result['verdict'].upper()}")
    print(
        f"  {result['n_supported']}/{result['n_components']} components supported, "
        f"{result['n_untestable']} untestable"
    )
    for component in result["components"]:
        detail = f" ({component['reason']})" if component["reason"] else ""
        print(f"    vs {component['model_b']:16s} {component['verdict']}{detail}")


def rank(
    runs_dir: list[str],
    output_dir: str | None = None,
    recursive: bool = False,
    include_legacy: bool = False,
    on_duplicate: str = ON_DUPLICATE_NEWEST,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    metric: str = DEFAULT_RANK_METRIC,
    correction: str = NADEAU_BENGIO,
    min_test_groups: int = MIN_TEST_GROUPS,
    confidence_level: float = CONFIDENCE_LEVEL,
    claim: list[str] | None = None,
    emit_latex: bool = False,
) -> int:
    """Report how stable architecture rankings are across random seeds.

    Counts how often every model took each rank, gives Kendall's tau between
    per-seed rankings, and tests which cross-task ranking reversals survive
    seed variance.

    Returns:
        Zero on success, one on failure.
    """
    discovered = _discover_frame(
        runs_dir, recursive, include_legacy, on_duplicate, models, datasets
    )
    if discovered is None:
        return 1
    records, frame, _ = discovered
    frame, _ = reconcile_averaging(frame)

    if frame[frame["metric"] == metric].empty:
        print(f"No runs carry the metric '{metric}'", file=sys.stderr)
        return 1

    ranked = rank_within_seed(frame, metric)
    distribution = rank_distribution(frame, metric)
    taus = tau_summary(frame, metric)
    flips = all_flip_tests(
        frame,
        metric=metric,
        confidence_level=confidence_level,
        correction=correction,
        min_test_groups=min_test_groups,
    )

    seeds = sorted(int(seed) for seed in frame["seed"].unique())
    print(f"{len(records)} runs over seeds {seeds}, ranking on {metric}\n")
    print("Rank distribution")
    rank_columns = [c for c in distribution.columns if c.startswith("rank_")]
    print(
        distribution[
            ["dataset", "model", *rank_columns, "mode_rank", "top1_fraction", "mean_value"]
        ].to_string(index=False)
    )
    print("\nKendall tau between per-seed rankings")
    print(taus.to_string(index=False))

    if not flips.empty:
        print("\nCross-task ranking reversals")
        print(
            flips[
                [
                    "model_a",
                    "model_b",
                    "dataset_1",
                    "dataset_2",
                    "mean_1",
                    "mean_2",
                    "n_seeds",
                    "verdict",
                    "p_flip",
                    "p_adjusted",
                ]
            ].to_string(index=False)
        )
        counts = flips["verdict"].value_counts()
        print("\nVerdicts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    claims = []
    for specification in claim or []:
        parts = specification.split(":")
        if len(parts) != 3:
            print(f"Malformed --claim '{specification}'", file=sys.stderr)
            return 1
        result = evaluate_rank_claim(
            frame,
            parts[0],
            parts[1],
            parts[2],
            metric=metric,
            confidence_level=confidence_level,
            correction=correction,
            min_test_groups=min_test_groups,
        )
        claims.append(result)
        _print_claim(result)

    if output_dir:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(target / "seed_ranks.csv", index=False)
        distribution.to_csv(target / "rank_distribution.csv", index=False)
        taus.to_csv(target / "kendall_tau.csv", index=False)
        written = ["seed_ranks.csv", "rank_distribution.csv", "kendall_tau.csv"]
        if not flips.empty:
            flips.to_csv(target / "flip_tests.csv", index=False)
            written.append("flip_tests.csv")
        if claims:
            pd.DataFrame(
                [{k: v for k, v in c.items() if k != "components"} for c in claims]
            ).to_csv(target / "rank_claims.csv", index=False)
            written.append("rank_claims.csv")

        if emit_latex:
            tables = {
                "rank_distribution": rank_distribution_table(distribution, taus, metric=metric),
                "kendall_tau": tau_table(taus),
            }
            if not flips.empty:
                tables["flip_survival"] = flip_survival_table(
                    flips,
                    metric=metric,
                    confidence_level=confidence_level,
                    correction=correction,
                )
            written.extend(Path(path).name for path in write_tables(tables, target))

        print(f"\nWrote {len(written)} file(s) to {target}:")
        for name in written:
            print(f"  {name}")

    return 0


def add_parser(subparsers: Any) -> Any:
    """Register the aggregate and rank subcommands."""
    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate per-run test metrics across random seeds"
    )
    add_discovery_arguments(aggregate_parser)
    aggregate_parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1", "precision", "recall"],
        help="Metrics for the report and LaTeX tables, in column order",
    )
    aggregate_parser.set_defaults(
        handler="aggregate", to_kwargs=_aggregate_kwargs, pass_logger=False
    )

    rank_parser = subparsers.add_parser(
        "rank", help="Report how stable architecture rankings are across seeds"
    )
    add_discovery_arguments(rank_parser)
    rank_parser.add_argument(
        "--metric",
        default=DEFAULT_RANK_METRIC,
        help=f"Metric to rank on (default: {DEFAULT_RANK_METRIC})",
    )
    rank_parser.add_argument(
        "--correction",
        choices=[NADEAU_BENGIO, NO_CORRECTION],
        default=NADEAU_BENGIO,
        help="Standard error correction for the paired differences",
    )
    rank_parser.add_argument(
        "--min-test-groups",
        type=int,
        default=MIN_TEST_GROUPS,
        help=f"Fewest independent test units that can support a test (default: {MIN_TEST_GROUPS})",
    )
    rank_parser.add_argument(
        "--claim",
        nargs="+",
        default=None,
        metavar="MODEL:BEST_DATASET:WORST_DATASET",
        help="Evaluate a 'best here, worst there' claim",
    )
    rank_parser.set_defaults(handler="rank", to_kwargs=_rank_kwargs, pass_logger=False)
    return aggregate_parser


def _discovery_kwargs(args: Any) -> dict[str, Any]:
    return {
        "runs_dir": args.runs_dir,
        "output_dir": args.output_dir,
        "recursive": args.recursive,
        "include_legacy": args.include_legacy,
        "on_duplicate": args.on_duplicate,
        "models": args.models,
        "datasets": args.datasets,
        "confidence_level": args.confidence_level,
        "emit_latex": args.emit_latex,
    }


def _aggregate_kwargs(args: Any) -> dict[str, Any]:
    return {**_discovery_kwargs(args), "metrics": args.metrics}


def _rank_kwargs(args: Any) -> dict[str, Any]:
    return {
        **_discovery_kwargs(args),
        "metric": args.metric,
        "correction": args.correction,
        "min_test_groups": args.min_test_groups,
        "claim": args.claim,
    }
