"""booktabs table emitters for the seed-variance analysis.

Every literal containing a backslash is a raw string. Written as an ordinary
string, "\\toprule" is a tab followed by "oprule" and "\\bottomrule" is a
backspace followed by "ottomrule"; both are valid Python escapes, so neither
the configured ruff rules nor W605 reports them and the emitted table silently
fails to compile.

The tables need only the booktabs package, which the manuscript already loads.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepecgkit.analysis.ranking import (
    VERDICT_BIMODAL,
    VERDICT_CONTRADICTED,
    VERDICT_NOT_REPRODUCED,
    VERDICT_SUGGESTIVE,
    VERDICT_SUPPORTED,
    VERDICT_UNRESOLVED,
    VERDICT_UNTESTABLE,
)

MISSING_CELL = "--"
DEFAULT_PRECISION = 3

MODEL_DISPLAY_NAMES = {
    "resnet": "ResNet1D",
    "se-resnet": "SE-ResNet1D",
    "inception-time": "InceptionTime",
    "convnext-v2": "ConvNeXt-V2",
    "crnn": "CRNN",
    "medformer": "Medformer",
    "mamba": "Mamba",
}

DATASET_DISPLAY_NAMES = {
    "ptbxl": "PTB-XL",
    "af-classification": "PhysioNet 2017",
    "mitbih-afdb": "MIT-BIH AFDB",
    "unified-af": "Unified AF",
    "ltafdb": "LTAFDB",
}

LOWER_IS_BETTER = frozenset(
    {
        "ece",
        "adaptive_ece",
        "mce",
        "brier",
        "ece_temp",
        "adaptive_ece_temp",
        "mce_temp",
        "brier_temp",
    }
)

METRIC_DISPLAY_NAMES = {
    "f1": "F1",
    "precision": "Prec.",
    "recall": "Recall",
    "auroc": "AUROC",
    "accuracy": "Acc.",
    "mcc": "MCC",
    "exact_match_accuracy": "Exact match",
    "label_accuracy": "Label acc.",
    "ece": "ECE",
    "adaptive_ece": "AECE",
    "mce": "MCE",
    "brier": "Brier",
    "temperature": r"$T$",
    "ece_temp": r"ECE$_T$",
    "adaptive_ece_temp": r"AECE$_T$",
    "mce_temp": r"MCE$_T$",
    "brier_temp": r"Brier$_T$",
    "auroc_macro": "AUROC",
    "balanced_accuracy": "Bal. acc.",
    "f1_macro": "F1",
}

BASELINE_DISPLAY_NAMES = {
    "random-init": "Random init",
}

VERDICT_DISPLAY_NAMES = {
    VERDICT_SUPPORTED: r"\textbf{Supported}",
    VERDICT_SUGGESTIVE: "Suggestive",
    VERDICT_UNRESOLVED: "Unresolved",
    VERDICT_NOT_REPRODUCED: "Not reproduced",
    VERDICT_CONTRADICTED: "Contradicted",
    VERDICT_BIMODAL: "Bimodal",
    VERDICT_UNTESTABLE: "Untestable",
}

_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def escape_latex(text: str) -> str:
    """Escape the characters LaTeX treats specially.

    Args:
        text: Raw text.

    Returns:
        Text safe to place in a table cell.
    """
    escaped = str(text)
    for character, replacement in _ESCAPES:
        escaped = escaped.replace(character, replacement)
    return escaped


def display_model(name: str) -> str:
    """Render a registry model name for the manuscript.

    Args:
        name: Registry name such as "se-resnet".

    Returns:
        The display name, escaped when unrecognised.
    """
    return MODEL_DISPLAY_NAMES.get(name, escape_latex(name))


def display_dataset(name: str) -> str:
    """Render a registry dataset name for the manuscript.

    Args:
        name: Registry name such as "af-classification".

    Returns:
        The display name, escaped when unrecognised.
    """
    return DATASET_DISPLAY_NAMES.get(name, escape_latex(name))


def display_metric(name: str) -> str:
    """Render a metric column name for the manuscript.

    Args:
        name: Metric key such as "f1".

    Returns:
        The display name, escaped when unrecognised.
    """
    return METRIC_DISPLAY_NAMES.get(name, escape_latex(name))


def _number(value: float, precision: int = DEFAULT_PRECISION) -> str:
    """Format one number, or a dash when it is not finite.

    Args:
        value: Value to render.
        precision: Decimal places.

    Returns:
        The formatted number or the missing-cell marker.
    """
    if value is None or not np.isfinite(value):
        return MISSING_CELL
    return f"{value:.{precision}f}"


def _signed(value: float, precision: int = DEFAULT_PRECISION) -> str:
    """Format one number with an explicit sign in maths mode.

    Args:
        value: Value to render.
        precision: Decimal places.

    Returns:
        The formatted number or the missing-cell marker.
    """
    if value is None or not np.isfinite(value):
        return MISSING_CELL
    return f"${value:+.{precision}f}$"


def format_cell(
    mean: float,
    lower: float,
    upper: float,
    n: int = 0,
    precision: int = DEFAULT_PRECISION,
) -> str:
    """Render a mean with its interval beneath, in the paper's style.

    Args:
        mean: Point estimate.
        lower: Interval lower bound.
        upper: Interval upper bound.
        n: Number of contributing seeds.
        precision: Decimal places.

    Returns:
        A table cell. A single seed renders bare with no interval, because one
        run carries no dispersion; a missing value renders as a dash.
    """
    if mean is None or not np.isfinite(mean):
        return MISSING_CELL
    point = f"${mean:.{precision}f}$"
    if n == 1 or not np.isfinite(lower) or not np.isfinite(upper):
        return point
    return f"{point} {{\\scriptsize [{lower:.{precision}f}, {upper:.{precision}f}]}}"


def _note_rows(notes: Sequence[str], n_columns: int) -> list[str]:
    """Wrap table notes as full-width footnote rows.

    Args:
        notes: Note lines, already escaped.
        n_columns: Table column count.

    Returns:
        Rendered rows.
    """
    return [rf"\multicolumn{{{n_columns}}}{{l}}{{\footnotesize {note}}} \\" for note in notes]


def _wrap_table(
    body: Sequence[str],
    column_spec: str,
    caption: str,
    label: str,
    size: str = r"\small",
) -> str:
    """Assemble a complete single-column table float.

    Args:
        body: Rows between the rules, including the header row.
        column_spec: The tabular column specification.
        caption: Caption text, already escaped where needed.
        label: Cross-reference label.
        size: Font size macro.

    Returns:
        The table as a string.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        size,
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
    ]
    lines.extend(body)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def results_table(
    aggregated: pd.DataFrame,
    dataset: str,
    metrics: Sequence[str] = ("f1", "precision", "recall"),
    label: str | None = None,
    precision: int = DEFAULT_PRECISION,
) -> str:
    """Per-dataset mean and interval table, replacing a single-run table.

    Args:
        aggregated: Frame from :func:`deepecgkit.analysis.aggregation.aggregate_runs`.
        dataset: Dataset to render.
        metrics: Metric columns in display order.
        label: Cross-reference label.
        precision: Decimal places.

    Returns:
        The table as a string. Nothing is bolded: marking the best mean and
        everything whose interval overlaps it is a pairwise comparison the
        per-model intervals cannot support. Rows are ordered by the first
        metric, ascending when that metric is one where lower is better, so a
        calibration table leads with the best-calibrated model rather than the
        worst.
    """
    scope = aggregated[aggregated["dataset"] == dataset]
    label = label or f"tab:seed_{dataset.replace('-', '_')}"
    ascending = metrics[0] in LOWER_IS_BETTER
    ranking = (
        scope[scope["metric"] == metrics[0]]
        .sort_values("mean", ascending=ascending)["model"]
        .tolist()
    )

    header = " & ".join(["Model"] + [display_metric(m) for m in metrics] + ["$n$"])
    body = [header + r" \\", r"\midrule"]
    averagings = set()
    for model in ranking:
        cells = [display_model(model)]
        seeds = 0
        for metric in metrics:
            row = scope[(scope["model"] == model) & (scope["metric"] == metric)]
            if row.empty:
                cells.append(MISSING_CELL)
                continue
            entry = row.iloc[0]
            seeds = int(entry["n"])
            averagings.add(str(entry["averaging"]))
            cells.append(
                format_cell(entry["mean"], entry["lower_ci"], entry["upper_ci"], seeds, precision)
            )
        cells.append(str(seeds))
        body.append(" & ".join(cells) + r" \\")

    notes = [
        "Intervals describe how much each model moves when the split and the initialisation",
        r"are redrawn; they are not a pairwise test. See Table~\ref{tab:flip_survival}.",
    ]
    if averagings:
        notes.append(f"Averaging: {', '.join(sorted(escape_latex(a) for a in averagings))}.")
    body.append(r"\midrule")
    body.extend(_note_rows(notes, len(metrics) + 2))

    caption = (
        f"{display_dataset(dataset)} results, mean over $n$ random seeds. Each seed redraws the "
        "patient-level split and reinitialises the weights, so brackets give two-sided 95\\% "
        "Student-$t$ intervals over that combined variation. The interval is a within-corpus "
        "statement and is not a generalisation interval."
    )
    return _wrap_table(body, "l" + "c" * (len(metrics) + 1), caption, label)


def rank_distribution_table(
    distribution: pd.DataFrame,
    taus: pd.DataFrame | None = None,
    label: str = "tab:rank_distribution",
    metric: str = "f1",
) -> str:
    """How often each model attained each rank, per dataset.

    Args:
        distribution: Frame from :func:`deepecgkit.analysis.ranking.rank_distribution`.
        taus: Optional frame from :func:`deepecgkit.analysis.ranking.tau_summary`.
        label: Cross-reference label.
        metric: Metric that was ranked, for the caption.

    Returns:
        The table as a string.
    """
    max_models = int(distribution["n_models"].max()) if not distribution.empty else 0
    n_columns = max_models + 4
    header = " & ".join(
        ["Model"]
        + [f"R{position}" for position in range(1, max_models + 1)]
        + ["Mode", "Top-1", f"{display_metric(metric)} (mean $\\pm$ sd)"]
    )
    body = [header + r" \\"]

    for dataset in sorted(distribution["dataset"].unique()):
        scope = distribution[distribution["dataset"] == dataset].sort_values(
            "mean_value", ascending=False
        )
        n_seeds = int(scope["n_seeds"].max())
        n_models = int(scope["n_models"].max())
        body.append(r"\midrule")
        body.append(
            rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{display_dataset(dataset)}}} "
            rf"({n_models} models, {n_seeds} seeds)}} \\"
        )
        for _, row in scope.iterrows():
            cells = [display_model(str(row["model"]))]
            for position in range(1, max_models + 1):
                column = f"rank_{position}"
                if column in scope.columns and np.isfinite(row.get(column, np.nan)):
                    cells.append(str(int(row[column])))
                else:
                    cells.append(MISSING_CELL)
            mode_count = round(float(row["mode_fraction"]) * row["n_seeds"])
            cells.append(f"{int(row['mode_rank'])} ({mode_count}/{int(row['n_seeds'])})")
            cells.append(f"{float(row['top1_fraction']):.2f}")
            std = row["std_value"]
            spread = f" $\\pm$ {float(std):.3f}" if np.isfinite(std) else ""
            cells.append(f"${float(row['mean_value']):.3f}$" + spread)
            body.append(" & ".join(cells) + r" \\")

    notes = ["Ties share the better rank, so a rank column may over-fill."]
    if taus is not None and not taus.empty:
        rendered = []
        for _, row in taus.iterrows():
            if not np.isfinite(row["tau_mean"]):
                continue
            rendered.append(
                f"{display_dataset(str(row['dataset']))} ${float(row['tau_mean']):.2f}$ "
                f"$[{float(row['tau_min']):.2f}, {float(row['tau_max']):.2f}]$"
            )
        if rendered:
            joined = "; ".join(rendered)
            notes.append(r"Kendall $\bar\tau_b$ over all seed-pair rankings: " + joined + ".")
            notes.append(
                r"With $|M|\le 4$ models the smallest attainable two-sided $p$ is $2/|M|!=0.083$,"
            )
            notes.append("so no coefficient can be significant; these are descriptive only.")
    body.append(r"\midrule")
    body.extend(_note_rows(notes, n_columns))

    caption = (
        f"Distribution of {display_metric(metric)} ranks over random seeds. Each cell counts the "
        "seeds in which the model attained that rank within its dataset (rank 1 = best); rows sum "
        "to the seed count. Dashes mark ranks that cannot occur because fewer models were trained "
        "on that dataset."
    )
    return _wrap_table(body, "l" + "c" * (n_columns - 1), caption, label, size=r"\scriptsize")


def flip_survival_table(
    flips: pd.DataFrame,
    label: str = "tab:flip_survival",
    metric: str = "f1",
    exclusions: Sequence[str] | None = None,
) -> str:
    """The headline table: which cross-task reversals survive seed variance.

    Args:
        flips: Frame from :func:`deepecgkit.analysis.ranking.all_flip_tests`.
        label: Cross-reference label.
        metric: Metric that was compared, for the caption.
        exclusions: Model or dataset restrictions to echo into the notes.

    Returns:
        The table as a string.
    """
    n_columns = 6
    header = " & ".join(
        [
            "$A$ vs.\\ $B$",
            r"$\Delta_1$ [95\% CI]",
            "$k/n$",
            r"$\Delta_2$ [95\% CI]",
            r"$p_{\mathrm{BY}}$",
            "Verdict",
        ]
    )
    body = [header + r" \\"]
    mixed_averaging = False

    for (dataset_1, dataset_2), scope in flips.groupby(["dataset_1", "dataset_2"], sort=True):
        body.append(r"\midrule")
        body.append(
            rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{display_dataset(str(dataset_1))} "
            rf"$\rightarrow$ {display_dataset(str(dataset_2))}}}}} \\"
        )
        for _, row in scope.iterrows():
            comparable = bool(row.get("comparable_magnitudes", True))
            marker = "" if comparable else r"^{\dagger}"
            if not comparable:
                mixed_averaging = True
            first = _interval_cell(row["mean_1"], row["lower_1"], row["upper_1"], marker)
            second = _interval_cell(row["mean_2"], row["lower_2"], row["upper_2"], marker)
            wins = (
                f"{int(row['wins_1'])}/{int(row['n_1'])}"
                if np.isfinite(row["n_1"]) and int(row["n_1"]) > 0
                else MISSING_CELL
            )
            adjusted = row.get("p_adjusted", float("nan"))
            pair = f"{display_model(str(row['model_a']))} / {display_model(str(row['model_b']))}"
            verdict = str(row["verdict"])
            body.append(
                " & ".join(
                    [
                        pair,
                        first,
                        wins,
                        second,
                        _number(adjusted, 3) if np.isfinite(adjusted) else MISSING_CELL,
                        VERDICT_DISPLAY_NAMES.get(verdict, escape_latex(verdict)),
                    ]
                )
                + r" \\"
            )

    notes = [
        "Supported: both intervals exclude zero with opposite signs. Unresolved: the mean",
        "ordering reverses but an interval covers zero. Not reproduced: the ordering does not",
        "reverse. Bimodal: a model collapses on a subset of seeds, so a mean is not a summary.",
        "Untestable: too few independent test units or seeds to support an interval.",
    ]
    if mixed_averaging:
        notes.append(
            r"$^{\dagger}$ the two $\Delta$ are differences of differently averaged scores and are"
        )
        notes.append("not comparable in magnitude; only their signs enter the test.")
    if exclusions:
        notes.append("Restricted to: " + escape_latex(", ".join(exclusions)) + ".")
    body.append(r"\midrule")
    body.extend(_note_rows(notes, n_columns))

    caption = (
        "Cross-task ranking reversals under seed variance. Each row compares two architectures "
        "on two datasets over $n$ shared random seeds; every seed redraws the patient-level split "
        "and reinitialises the weights. $\\Delta_i$ is the mean paired per-seed difference in "
        f"{display_metric(metric)} on dataset $D_i$, with a two-sided 95\\% Student-$t$ interval "
        "whose standard error carries the Nadeau--Bengio correction for the training-set overlap "
        "between seeds. $k/n$ counts the seeds in which $A$ led on $D_1$. A reversal is supported "
        "only when both intervals exclude zero with opposite signs (an intersection--union test), "
        "so $p_{\\mathrm{BY}}$ is the larger component $p$ and is an upper bound. All comparisons "
        "are exploratory: they were selected by inspecting the single-seed tables."
    )
    return _wrap_table(body, "lccccl", caption, label)


def _interval_cell(mean: float, lower: float, upper: float, marker: str = "") -> str:
    """Render a signed mean with its interval.

    Args:
        mean: Point estimate.
        lower: Interval lower bound.
        upper: Interval upper bound.
        marker: Optional superscript appended to the point estimate.

    Returns:
        A table cell, or a dash when no interval exists.
    """
    if mean is None or not np.isfinite(mean):
        return MISSING_CELL
    inner = f"{mean:+.3f}{marker}"
    if not np.isfinite(lower) or not np.isfinite(upper):
        return f"${inner}$"
    return f"${inner}$ $[{lower:+.3f}, {upper:+.3f}]$"


def tau_table(taus: pd.DataFrame, label: str = "tab:kendall_tau") -> str:
    """Supplementary Kendall's tau table.

    Args:
        taus: Frame from :func:`deepecgkit.analysis.ranking.tau_summary`.
        label: Cross-reference label.

    Returns:
        The table as a string.
    """
    n_columns = 6
    header = " & ".join(
        ["Dataset", "$|M|$", "Seeds", "Pairs", r"$\bar\tau_b$ [min, max]", "$p_{\\min}$"]
    )
    body = [header + r" \\", r"\midrule"]
    for _, row in taus.iterrows():
        if np.isfinite(row["tau_mean"]):
            summary = (
                f"${float(row['tau_mean']):.2f}$ "
                f"$[{float(row['tau_min']):.2f}, {float(row['tau_max']):.2f}]$"
            )
        else:
            summary = MISSING_CELL
        body.append(
            " & ".join(
                [
                    display_dataset(str(row["dataset"])),
                    str(int(row["n_models"])),
                    str(int(row["n_seeds"])),
                    str(int(row["n_pairs"])),
                    summary,
                    _number(row["min_attainable_p"], 3),
                ]
            )
            + r" \\"
        )
    body.append(r"\midrule")
    body.extend(
        _note_rows(
            [
                r"$p_{\min}=2/|M|!$ is the smallest two-sided $p$ the exact permutation null can",
                "attain. No p-values are reported: at four models the floor is 0.083, so no",
                "coefficient here can reach significance and these are descriptive only.",
            ],
            n_columns,
        )
    )
    caption = (
        r"Kendall $\tau_b$ between per-seed rankings, over every pair of seeds within each "
        "dataset. The coefficients are mutually dependent, so no interval or pooled test is given."
    )
    return _wrap_table(body, "lccccc", caption, label)


def transfer_matrix_table(
    matrix: pd.DataFrame,
    metric: str = "auroc_macro",
    baseline_row: str | None = None,
    label: str = "tab:transfer_matrix",
    precision: int = DEFAULT_PRECISION,
) -> str:
    """Cross-dataset linear-probe transfer matrix.

    Args:
        matrix: Frame from :meth:`deepecgkit.evaluation.TransferMatrix.to_frame`,
            indexed by encoder source with one column per probe target.
        metric: Metric filling the cells, named in the caption.
        baseline_row: Index entry holding the untrained-encoder floor, moved to
            the top and ruled off from the trained encoders.
        label: Cross-reference label.
        precision: Decimal places.

    Returns:
        The table as a string. Nothing is bolded: a single probe score carries
        no interval, so marking a best cell would assert a comparison the
        numbers cannot support.
    """
    targets = list(matrix.columns)
    header = " & ".join(["Encoder source"] + [display_dataset(t) for t in targets])
    body = [header + r" \\", r"\midrule"]

    ordered = list(matrix.index)
    if baseline_row is not None and baseline_row in ordered:
        ordered = [baseline_row] + [row for row in ordered if row != baseline_row]

    for source in ordered:
        name = BASELINE_DISPLAY_NAMES.get(source) or display_dataset(source)
        cells = [name] + [_number(matrix.loc[source, t], precision) for t in targets]
        body.append(" & ".join(cells) + r" \\")
        if baseline_row is not None and source == baseline_row:
            body.append(r"\midrule")

    notes = [
        "Each cell freezes the encoder trained on its row's corpus, embeds the column corpus,",
        "and fits a standardised logistic probe on that corpus's training split alone.",
    ]
    if baseline_row is not None and baseline_row in matrix.index:
        notes.append("The first row probes an untrained encoder and is the floor for every column.")
    body.append(r"\midrule")
    body.extend(_note_rows(notes, len(targets) + 1))

    caption = (
        f"Cross-dataset transfer, {display_metric(metric)} of a linear probe on a held-out "
        "split. Rows are the corpus each frozen encoder was trained on, columns the corpus it "
        "was probed on; the diagonal is the in-domain probe. Every corpus is run single-lead at "
        "a common sampling rate so that each encoder sees an identical input, and the probe head "
        "is refitted per column because the label spaces differ."
    )
    return _wrap_table(body, "l" + "c" * len(targets), caption, label)


def write_tables(tables: dict[str, str], output_dir: Any) -> list[str]:
    """Write rendered tables to .tex files.

    Args:
        tables: Mapping from stem to rendered table.
        output_dir: Destination directory, created if absent.

    Returns:
        The written paths as strings.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for stem, content in sorted(tables.items()):
        path = directory / f"{stem}.tex"
        path.write_text(content)
        written.append(str(path))
    return written
