"""
Tests for deepecgkit.analysis.latex module.
"""

import pandas as pd
import pytest

from deepecgkit.analysis.aggregation import aggregate_runs
from deepecgkit.analysis.latex import (
    LOWER_IS_BETTER,
    MISSING_CELL,
    display_dataset,
    display_model,
    escape_latex,
    flip_survival_table,
    format_cell,
    rank_distribution_table,
    results_table,
    tau_table,
    transfer_matrix_table,
    write_tables,
)
from deepecgkit.analysis.ranking import all_flip_tests, rank_distribution, tau_summary

SEEDS = (42, 43, 44, 45, 46)

VALUES = {
    "ptbxl": {
        "inception-time": [0.700, 0.706, 0.698, 0.703, 0.699],
        "crnn": [0.650, 0.648, 0.653, 0.649, 0.652],
    },
    "af-classification": {
        "inception-time": [0.600, 0.604, 0.597, 0.602, 0.598],
        "crnn": [0.700, 0.697, 0.703, 0.699, 0.702],
    },
}


def make_runs():
    """Build a long run frame spanning two datasets and two models.

    Returns:
        A frame shaped like the aggregation output.
    """
    rows = []
    for dataset, models in VALUES.items():
        for model, values in models.items():
            for seed, value in zip(SEEDS, values):
                for metric, offset in (("f1", 0.0), ("precision", 0.01), ("recall", -0.01)):
                    rows.append(
                        {
                            "run_dir": f"{model}-{dataset}-seed{seed}",
                            "model": model,
                            "dataset": dataset,
                            "seed": seed,
                            "averaging": "macro",
                            "multi_label": False,
                            "n_train": 7000,
                            "n_val": 2000,
                            "n_test": 1000,
                            "n_test_groups": 500,
                            "metric": metric,
                            "value": value + offset,
                        }
                    )
    return pd.DataFrame(rows)


RUNS = make_runs()
BOOKTABS_RULES = ("\\toprule", "\\midrule", "\\bottomrule")


class TestEscapeLatex:
    """Test escaping of characters LaTeX treats specially."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("se_resnet", "se\\_resnet"),
            ("a & b", "a \\& b"),
            ("50%", "50\\%"),
            ("x#y", "x\\#y"),
        ],
    )
    def test_special_characters_are_escaped(self, raw, expected):
        """Each special character gains its backslash."""
        assert escape_latex(raw) == expected

    def test_plain_text_is_untouched(self):
        """Ordinary names pass through unchanged."""
        assert escape_latex("InceptionTime") == "InceptionTime"


class TestDisplayNames:
    """Test the registry-to-manuscript name mapping."""

    def test_known_model_is_renamed(self):
        """Registry names are replaced by the manuscript spelling."""
        assert display_model("se-resnet") == "SE-ResNet1D"

    def test_known_dataset_is_renamed(self):
        """Dataset keys map to the names used in the paper."""
        assert display_dataset("af-classification") == "PhysioNet 2017"

    def test_unknown_name_is_escaped_not_dropped(self):
        """An unregistered name still reaches the table, safely."""
        assert display_model("my_model") == "my\\_model"


class TestFormatCell:
    """Test rendering of a mean with its interval."""

    def test_interval_is_rendered_beneath_the_mean(self):
        """A multi-seed cell carries its bounds."""
        cell = format_cell(0.704, 0.697, 0.711, n=5)
        assert "0.704" in cell
        assert "[0.697, 0.711]" in cell

    def test_single_seed_has_no_interval(self):
        """One run carries no dispersion, so none is printed."""
        cell = format_cell(0.704, float("nan"), float("nan"), n=1)
        assert cell == "$0.704$"

    def test_missing_value_renders_a_dash(self):
        """An absent metric is visibly absent."""
        assert format_cell(float("nan"), 0.1, 0.2, n=5) == MISSING_CELL


class TestTables:
    """Test the rendered booktabs tables."""

    @pytest.fixture
    def tables(self):
        """Every table rendered from the shared synthetic runs."""
        flips = all_flip_tests(RUNS)
        taus = tau_summary(RUNS)
        return {
            "flip_survival": flip_survival_table(flips),
            "rank_distribution": rank_distribution_table(rank_distribution(RUNS), taus),
            "results": results_table(aggregate_runs(RUNS), "ptbxl"),
            "kendall_tau": tau_table(taus),
        }

    def test_every_table_carries_genuine_booktabs_rules(self, tables):
        """The rules must be real control sequences, not escape characters.

        Written as an ordinary string "\\toprule" is a tab followed by
        "oprule", which no configured lint rule reports. Asserting on the
        escaped form is what makes this test meaningful.
        """
        for name, table in tables.items():
            for rule in BOOKTABS_RULES:
                assert rule in table, f"{name} is missing {rule}"

    def test_no_control_characters_leaked_into_output(self, tables):
        """A mis-escaped literal would leave a tab, backspace or form feed."""
        for name, table in tables.items():
            for character in ("\t", "\x08", "\x0c"):
                assert character not in table, f"{name} contains a control character"

    def test_tables_are_balanced_floats(self, tables):
        """Each table opens and closes its float and its tabular."""
        for table in tables.values():
            assert table.count("\\begin{table}") == table.count("\\end{table}") == 1
            assert table.count("\\begin{tabular}") == table.count("\\end{tabular}") == 1

    def test_every_row_matches_the_column_count(self, tables):
        """A row with the wrong number of ampersands will not compile."""
        for name, table in tables.items():
            spec = table.split("\\begin{tabular}{")[1].split("}")[0]
            expected = len(spec.replace("|", ""))
            for line in table.splitlines():
                if not line.endswith("\\\\") or "multicolumn" in line:
                    continue
                assert line.count("&") == expected - 1, f"{name}: {line}"

    def test_model_names_use_manuscript_spelling(self, tables):
        """Registry names never reach the page."""
        assert "InceptionTime" in tables["results"]
        assert "inception-time" not in tables["results"]

    def test_flip_table_labels_its_verdicts(self, tables):
        """The headline column states the outcome in words."""
        assert "Supported" in tables["flip_survival"]

    def test_results_caption_warns_against_pairwise_reading(self, tables):
        """The interval must not be mistaken for a comparison."""
        assert "not a pairwise test" in tables["results"]

    def test_tau_table_states_the_significance_floor(self, tables):
        """The reader is told no coefficient here can be significant."""
        assert "0.083" in tables["kendall_tau"]

    def test_write_tables_creates_files(self, tables, temp_dir):
        """Rendered tables land as .tex files named by their stem."""
        written = write_tables(tables, temp_dir / "out")
        assert len(written) == len(tables)
        for path in written:
            assert path.endswith(".tex")
            assert "\\toprule" in (temp_dir / "out" / path.split("/")[-1]).read_text()


class TestCalibrationTableOrdering:
    """Test that lower-is-better metrics are not ranked upside down."""

    def test_calibration_metrics_are_marked_lower_is_better(self):
        """Ranking by ECE ascending is the difference between best and worst first."""
        assert "ece" in LOWER_IS_BETTER
        assert "brier" in LOWER_IS_BETTER
        assert "f1" not in LOWER_IS_BETTER

    def test_calibration_table_leads_with_the_best_calibrated_model(self):
        """The first row of an ECE table must be the model with the lowest ECE."""
        runs = pd.DataFrame(
            [
                {
                    "run_dir": f"r{index}",
                    "model": model,
                    "dataset": "ptbxl",
                    "seed": seed,
                    "averaging": "macro",
                    "multi_label": False,
                    "n_train": 100,
                    "n_val": 20,
                    "n_test": 20,
                    "n_test_groups": 20,
                    "metric": "ece",
                    "value": value,
                }
                for index, (model, seed, value) in enumerate(
                    [("resnet", 1, 0.30), ("resnet", 2, 0.32), ("crnn", 1, 0.05), ("crnn", 2, 0.07)]
                )
            ]
        )

        table = results_table(aggregate_runs(runs), "ptbxl", metrics=("ece",))
        body = table.split(r"\midrule")[1]

        assert body.index("CRNN") < body.index("ResNet1D")

    def test_discrimination_table_still_leads_with_the_highest_score(self):
        """Flipping the sort for calibration must not flip it for F1."""
        runs = pd.DataFrame(
            [
                {
                    "run_dir": f"r{index}",
                    "model": model,
                    "dataset": "ptbxl",
                    "seed": seed,
                    "averaging": "macro",
                    "multi_label": False,
                    "n_train": 100,
                    "n_val": 20,
                    "n_test": 20,
                    "n_test_groups": 20,
                    "metric": "f1",
                    "value": value,
                }
                for index, (model, seed, value) in enumerate(
                    [("resnet", 1, 0.30), ("resnet", 2, 0.32), ("crnn", 1, 0.85), ("crnn", 2, 0.87)]
                )
            ]
        )

        table = results_table(aggregate_runs(runs), "ptbxl", metrics=("f1",))
        body = table.split(r"\midrule")[1]

        assert body.index("CRNN") < body.index("ResNet1D")


TRANSFER_SOURCES = ["random-init", "af-classification", "ptbxl"]
TRANSFER_TARGETS = ["af-classification", "ptbxl", "mitbih-afdb"]


def make_transfer_matrix():
    """Build a transfer matrix frame with one unscored cell.

    Returns:
        A frame indexed by encoder source with one column per probe target.
    """
    return pd.DataFrame(
        [
            [0.512, 0.601, 0.498],
            [0.884, 0.702, 0.771],
            [0.690, float("nan"), 0.664],
        ],
        index=TRANSFER_SOURCES,
        columns=TRANSFER_TARGETS,
    )


class TestTransferMatrixTable:
    """The transfer matrix renders as a booktabs table with the floor on top."""

    def test_wraps_a_booktabs_table(self):
        """The output is a complete table float."""
        table = transfer_matrix_table(make_transfer_matrix())
        assert r"\toprule" in table
        assert r"\bottomrule" in table
        assert r"\begin{tabular}{lccc}" in table

    def test_uses_dataset_display_names(self):
        """Registry names are rendered as the manuscript's names."""
        table = transfer_matrix_table(make_transfer_matrix())
        assert "PhysioNet 2017" in table
        assert "PTB-XL" in table
        assert "MIT-BIH AFDB" in table

    def test_unscored_cells_render_as_a_dash(self):
        """A cell the probe could not score is a dash, not a NaN literal."""
        table = transfer_matrix_table(make_transfer_matrix())
        assert MISSING_CELL in table
        assert "nan" not in table.lower()

    def test_baseline_row_leads_and_is_ruled_off(self):
        """The floor is the first body row and is separated from the trained encoders."""
        table = transfer_matrix_table(make_transfer_matrix(), baseline_row="random-init")
        body = table.split(r"\midrule")
        assert "Random init" in body[1]
        assert "PhysioNet 2017" in body[2]

    def test_baseline_row_is_optional(self):
        """Without a baseline the rows keep the frame's own order."""
        table = transfer_matrix_table(make_transfer_matrix())
        assert "Random init" in table
        assert table.count(r"\midrule") == 2

    def test_metric_names_the_caption(self):
        """The caption says which metric fills the cells."""
        assert "AUROC" in transfer_matrix_table(make_transfer_matrix())
        assert "Acc." in transfer_matrix_table(make_transfer_matrix(), metric="accuracy")

    def test_label_is_settable(self):
        """The cross-reference label can be chosen by the caller."""
        table = transfer_matrix_table(make_transfer_matrix(), label="tab:probe")
        assert r"\label{tab:probe}" in table
