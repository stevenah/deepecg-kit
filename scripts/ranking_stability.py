"""
Report how stable architecture rankings are across random seeds.

For each dataset this counts how often every model took each rank, gives
Kendall's tau between the per-seed rankings, and tests which cross-task
ranking reversals survive seed variance. A reversal is supported only when
both paired differences exclude zero with opposite signs.

Usage:
    python scripts/ranking_stability.py --runs-dir runs
    python scripts/ranking_stability.py --runs-dir runs --output-dir analysis --emit-latex
    python scripts/ranking_stability.py --runs-dir runs --metric auroc
    python scripts/ranking_stability.py --runs-dir runs --claim inception-time:ptbxl:af-classification
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from deepecgkit.analysis.aggregation import (
    ON_DUPLICATE_ALL,
    ON_DUPLICATE_ERROR,
    ON_DUPLICATE_NEWEST,
    discover_runs,
    reconcile_averaging,
    runs_to_frame,
)
from deepecgkit.analysis.latex import (
    flip_survival_table,
    rank_distribution_table,
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


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Ranking stability across random seeds")
    parser.add_argument("--runs-dir", nargs="+", default=["runs"], help="Run directories")
    parser.add_argument("--output-dir", default=None, help="Write CSV and LaTeX output here")
    parser.add_argument("--recursive", action="store_true", help="Search at any depth")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include runs whose seed cannot be determined",
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
        "--metric",
        default=DEFAULT_RANK_METRIC,
        help=f"Metric to rank on (default: {DEFAULT_RANK_METRIC})",
    )
    parser.add_argument(
        "--correction",
        choices=[NADEAU_BENGIO, NO_CORRECTION],
        default=NADEAU_BENGIO,
        help="Standard error correction for the paired differences",
    )
    parser.add_argument(
        "--min-test-groups",
        type=int,
        default=MIN_TEST_GROUPS,
        help=f"Fewest independent test units that can support a test (default: {MIN_TEST_GROUPS})",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=CONFIDENCE_LEVEL,
        help=f"Two-sided coverage (default: {CONFIDENCE_LEVEL})",
    )
    parser.add_argument(
        "--claim",
        nargs="+",
        default=None,
        metavar="MODEL:BEST_DATASET:WORST_DATASET",
        help="Evaluate a 'best here, worst there' claim",
    )
    parser.add_argument("--emit-latex", action="store_true", help="Write booktabs tables")
    return parser


def _print_claim(result: dict) -> None:
    """Print one decomposed rank claim.

    Args:
        result: Output of :func:`evaluate_rank_claim`.
    """
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


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, defaulting to sys.argv.

    Returns:
        Process exit status.
    """
    args = _build_parser().parse_args(argv)

    try:
        records, _ = discover_runs(
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
    frame, _ = reconcile_averaging(frame)

    if frame[frame["metric"] == args.metric].empty:
        print(f"No runs carry the metric '{args.metric}'", file=sys.stderr)
        return 1

    ranked = rank_within_seed(frame, args.metric)
    distribution = rank_distribution(frame, args.metric)
    taus = tau_summary(frame, args.metric)
    flips = all_flip_tests(
        frame,
        metric=args.metric,
        confidence_level=args.confidence_level,
        correction=args.correction,
        min_test_groups=args.min_test_groups,
    )

    seeds = sorted(int(seed) for seed in frame["seed"].unique())
    print(f"{len(records)} runs over seeds {seeds}, ranking on {args.metric}\n")
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
    for specification in args.claim or []:
        parts = specification.split(":")
        if len(parts) != 3:
            print(f"Malformed --claim '{specification}'", file=sys.stderr)
            return 1
        result = evaluate_rank_claim(
            frame,
            parts[0],
            parts[1],
            parts[2],
            metric=args.metric,
            confidence_level=args.confidence_level,
            correction=args.correction,
            min_test_groups=args.min_test_groups,
        )
        claims.append(result)
        _print_claim(result)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(output_dir / "seed_ranks.csv", index=False)
        distribution.to_csv(output_dir / "rank_distribution.csv", index=False)
        taus.to_csv(output_dir / "kendall_tau.csv", index=False)
        written = ["seed_ranks.csv", "rank_distribution.csv", "kendall_tau.csv"]
        if not flips.empty:
            flips.to_csv(output_dir / "flip_tests.csv", index=False)
            written.append("flip_tests.csv")
        if claims:
            pd.DataFrame(
                [{k: v for k, v in claim.items() if k != "components"} for claim in claims]
            ).to_csv(output_dir / "rank_claims.csv", index=False)
            written.append("rank_claims.csv")

        if args.emit_latex:
            tables = {
                "rank_distribution": rank_distribution_table(
                    distribution, taus, metric=args.metric
                ),
                "kendall_tau": tau_table(taus),
            }
            if not flips.empty:
                tables["flip_survival"] = flip_survival_table(flips, metric=args.metric)
            written.extend(Path(path).name for path in write_tables(tables, output_dir))

        print(f"\nWrote {len(written)} file(s) to {output_dir}:")
        for name in written:
            print(f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
