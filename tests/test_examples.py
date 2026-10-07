"""
The scripts in examples/, run for real on a small synthetic dataset.

They are what a newcomer runs first and nothing else exercises them: linting
them, or checking that the names they import exist, does not notice a script
that hangs, stops half-way, or prints a wrong answer. Each one is run here in
this process with its own command line.
"""

import json
import runpy
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from deepecgkit.models import SimpleCNN
from tests.conftest import write_af_dataset

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Small batches on the CPU keep the runs to a second or two each.
COMMON = ["--batch-size", "8", "--device", "cpu"]


def run_example(monkeypatch, script: str, *args: str) -> None:
    """Run one example as `python examples/<script> <args>` would."""
    monkeypatch.setattr(sys, "argv", [script, *map(str, args)])
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")


@pytest.fixture(scope="module")
def data_home(tmp_path_factory) -> Path:
    """A data home holding a synthetic af-classification dataset, as a download would leave it."""
    home = tmp_path_factory.mktemp("home")
    write_af_dataset(home / "datasets" / "afclassificationdataset")
    return home


@pytest.fixture(autouse=True)
def example_environment(data_home, monkeypatch):
    """Point the examples at the synthetic data and keep their plots off the screen."""
    monkeypatch.setenv("DEEPECGKIT_HOME", str(data_home))
    matplotlib.use("Agg", force=True)


@pytest.fixture(scope="module")
def trained_checkpoint(data_home, tmp_path_factory) -> Path:
    """The checkpoint train_basic.py leaves behind, for the examples that start from one."""
    output = tmp_path_factory.mktemp("basic")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("DEEPECGKIT_HOME", str(data_home))
        run_example(
            patcher,
            "train_basic.py",
            "--model",
            "simple-cnn",
            "--epochs",
            "1",
            "--output-dir",
            output,
            *COMMON,
        )
    (checkpoint,) = (output / "checkpoints").glob("*.pt")
    return checkpoint


@pytest.fixture
def recording(tmp_path) -> Path:
    """Thirty seconds of a single lead at 300 Hz, as a raw .npy recording."""
    path = tmp_path / "ecg.npy"
    np.save(path, 300.0 * np.sin(np.linspace(0.0, 90.0, 9000)))
    return path


def test_train_basic_reports_test_results(trained_checkpoint, capsys, monkeypatch, tmp_path):
    run_example(
        monkeypatch,
        "train_basic.py",
        "--model",
        "simple-cnn",
        "--epochs",
        "1",
        "--output-dir",
        tmp_path,
        *COMMON,
    )

    output = capsys.readouterr().out
    assert "Test accuracy:" in output
    assert "Best checkpoint:" in output


def test_train_advanced_writes_its_plots_and_metrics(monkeypatch, tmp_path):
    run_example(
        monkeypatch,
        "train_advanced.py",
        "--model",
        "simple-cnn",
        "--epochs",
        "1",
        "--output-dir",
        tmp_path,
        *COMMON,
    )

    assert (tmp_path / "classification_metrics.csv").exists()
    assert {path.name for path in (tmp_path / "plots").iterdir()} >= {
        "confusion_matrix.png",
        "calibration_curve.png",
        "loss.png",
    }


def test_evaluate_model_reports_intervals(trained_checkpoint, monkeypatch, tmp_path):
    """It used to stop at its bootstrap step."""
    run_example(
        monkeypatch,
        "evaluate_model.py",
        "--checkpoint",
        trained_checkpoint,
        "--output-dir",
        tmp_path,
        "--n-bootstrap",
        "20",
        *COMMON,
    )

    report = (tmp_path / "evaluation_report.csv").read_text()
    assert "95% CI Lower" in report.splitlines()[0]
    assert (tmp_path / "confusion_matrix.png").exists()


def test_inference_names_a_class(trained_checkpoint, recording, monkeypatch, tmp_path):
    output = tmp_path / "predictions.json"

    run_example(
        monkeypatch,
        "inference.py",
        "--checkpoint",
        trained_checkpoint,
        "--input",
        recording,
        "--sampling-rate",
        "300",
        "--device",
        "cpu",
        "--output",
        output,
    )

    (result,) = json.loads(output.read_text())
    assert result["segments"] == 3
    assert result["predicted_class"] in {"Normal", "AF", "Other", "Noisy"}
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


def test_inference_reads_a_single_logit_as_a_probability(recording, monkeypatch, tmp_path):
    """A softmax over one output is always 1.0: every recording came out "class 0, certain"."""
    checkpoint = tmp_path / "binary.pt"
    torch.save(
        {
            "model_state_dict": SimpleCNN(input_channels=1, output_size=1).state_dict(),
            "model_config": {"name": "simple-cnn", "input_channels": 1, "output_size": 1},
            "train_config": {"binary_classification": True},
            "data_config": {
                "class_names": ["Non-AF", "AF"],
                "preprocessing": {
                    "sampling_rate": 300,
                    "segment_duration_seconds": 10.0,
                    "normalization": "zscore",
                },
            },
        },
        checkpoint,
    )
    output = tmp_path / "predictions.json"

    run_example(
        monkeypatch,
        "inference.py",
        "--checkpoint",
        checkpoint,
        "--input",
        recording,
        "--sampling-rate",
        "300",
        "--device",
        "cpu",
        "--output",
        output,
    )

    (result,) = json.loads(output.read_text())
    probabilities = result["probabilities"]
    assert set(probabilities) == {"Non-AF", "AF"}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert 0.0 < probabilities["AF"] < 1.0


def test_explore_dataset_plots_a_sample(monkeypatch, tmp_path):
    """It used to hang on its first plot."""
    run_example(
        monkeypatch,
        "explore_dataset.py",
        "--dataset",
        "af-classification",
        "--num-samples",
        "1",
        "--save-dir",
        tmp_path,
    )

    assert len(list(tmp_path.glob("sample_*.png"))) == 1


def test_feature_extraction_on_synthetic_signals(monkeypatch, capsys):
    run_example(monkeypatch, "feature_extraction.py", "--num-samples", "8", *COMMON)

    assert "Feature shape: (8, 64)" in capsys.readouterr().out


def test_feature_extraction_probes_a_checkpoint(trained_checkpoint, monkeypatch, capsys, tmp_path):
    """The model comes from the checkpoint; --model used to have to repeat it."""
    output = tmp_path / "features.npz"

    run_example(
        monkeypatch,
        "feature_extraction.py",
        "--dataset",
        "af-classification",
        "--checkpoint",
        trained_checkpoint,
        "--output",
        output,
        *COMMON,
    )

    assert "Linear probe accuracy:" in capsys.readouterr().out
    assert output.exists()


def test_model_comparison_runs_every_model(monkeypatch, capsys):
    run_example(monkeypatch, "model_comparison.py", "--num-runs", "1", "--device", "cpu")

    output = capsys.readouterr().out
    assert "kanres: OK" in output
    assert output.count(": OK") >= 20
