"""Post-hoc analysis of finished training runs.

This subpackage aggregates the metrics written by completed runs, summarises
them across random seeds, and tests whether cross-dataset ranking reversals
survive that variance. It is distinct from
:mod:`deepecgkit.evaluation`, which evaluates a live model.
"""

from deepecgkit.analysis.aggregation import (
    RunRecord,
    aggregate_runs,
    discover_runs,
    read_run_metrics,
    reconcile_averaging,
    runs_to_frame,
)
from deepecgkit.analysis.latex import (
    escape_latex,
    flip_survival_table,
    rank_distribution_table,
    results_table,
    tau_table,
    transfer_matrix_table,
    write_tables,
)
from deepecgkit.analysis.ranking import (
    all_flip_tests,
    evaluate_rank_claim,
    rank_distribution,
    rank_within_seed,
    tau_summary,
)
from deepecgkit.analysis.statistics import (
    benjamini_yekutieli_adjust,
    holm_adjust,
    is_bimodal,
    kendall_tau,
    min_attainable_kendall_p,
    nadeau_bengio_factor,
    paired_summary,
    summarize_scores,
)

__all__ = [
    "RunRecord",
    "aggregate_runs",
    "all_flip_tests",
    "benjamini_yekutieli_adjust",
    "discover_runs",
    "escape_latex",
    "evaluate_rank_claim",
    "flip_survival_table",
    "holm_adjust",
    "is_bimodal",
    "kendall_tau",
    "min_attainable_kendall_p",
    "nadeau_bengio_factor",
    "paired_summary",
    "rank_distribution",
    "rank_distribution_table",
    "rank_within_seed",
    "read_run_metrics",
    "reconcile_averaging",
    "results_table",
    "runs_to_frame",
    "summarize_scores",
    "tau_summary",
    "tau_table",
    "transfer_matrix_table",
    "write_tables",
]
