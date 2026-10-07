"""
The Python snippets in the documentation, executed.

A page's blocks run in order in one namespace, the way a reader would paste
them, on a small synthetic dataset and with training cut to one epoch. A page
written as fragments of a larger script is given the names it takes for
granted (a fitted trainer, arrays of predictions, a recording on disk); a
block that is a placeholder, or needs the network or several real datasets,
is left out by a phrase it contains, with the reason beside it.

Checking that the names a snippet uses exist does not catch what this does: a
plot that needs a folder nobody created, a function handed the wrong layout,
an example that asks for a result that was never kept.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wfdb

from deepecgkit import registry
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGTrainer
from deepecgkit.utils import weights as weights_module
from tests.conftest import write_af_dataset

DOCS = Path(__file__).resolve().parent.parent / "docs"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.S)

PAGES = [
    "index.md",
    "getting-started/quickstart.md",
    "examples/train-basic.md",
    "examples/train-advanced.md",
    "examples/model-comparison.md",
    "user-guide/datasets.md",
    "user-guide/training.md",
    "user-guide/evaluation.md",
    "user-guide/calibration.md",
    "user-guide/feature-extraction.md",
    "user-guide/explainability.md",
    "user-guide/model-zoo.md",
]

#: Blocks that are not run, identified by a phrase they contain.
LEFT_OUT = {
    "PTBXLDataset.download_to()": "would download PTB-XL",
    "ModelClass(": "a placeholder for any model class",
    "def forward(self, x):\n    x = self.extract_features(x)": "a method shown out of its class",
    "plot_roc_curve(y_true, y_prob,": "a two-class plot; the running example has four classes",
}

#: Lines that stand for something the reader supplies, removed before running.
PLACEHOLDER_LINES = ("model = ...  # any nn.Module\n",)

#: Checkpoint paths the pages use as examples, each given a real checkpoint.
CHECKPOINT_PATHS = (
    "runs/my-run/checkpoints/best-epoch.pt",
    "runs/my-run/checkpoints/epoch=07-val_loss=0.4123.pt",
    "runs/my-run/checkpoints/my-checkpoint.pt",
)


@pytest.fixture(scope="module")
def data_home(tmp_path_factory) -> Path:
    """A data home holding a synthetic af-classification dataset."""
    home = tmp_path_factory.mktemp("docs-home")
    write_af_dataset(home / "datasets" / "afclassificationdataset")
    return home


@pytest.fixture(autouse=True)
def documentation_environment(data_home, monkeypatch, tmp_path):
    """Run on the CPU, for one epoch, in an empty folder, leaving the registries as found."""
    monkeypatch.setenv("DEEPECGKIT_HOME", str(data_home))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    matplotlib.use("Agg", force=True)

    fit = ECGTrainer.fit
    monkeypatch.setattr(
        ECGTrainer, "fit", lambda self, data, epochs=1, **options: fit(self, data, 1, **options)
    )

    # The built-ins register on first lookup; they have to be in the tables
    # before a copy is taken, or putting the copy back would empty them.
    registry.get_model_names()
    registry.get_dataset_names()
    models = dict(registry._MODEL_REGISTRY)
    datasets = dict(registry._DATASET_REGISTRY)
    weights = dict(weights_module.WEIGHTS_REGISTRY)
    yield
    for table, saved in (
        (registry._MODEL_REGISTRY, models),
        (registry._DATASET_REGISTRY, datasets),
        (weights_module.WEIGHTS_REGISTRY, weights),
    ):
        table.clear()
        table.update(saved)
    plt.close("all")


def _data_module() -> ECGDataModule:
    module = ECGDataModule(
        dataset_class=get_dataset("af-classification"), batch_size=16, download=True
    )
    module.setup(stage="fit")
    return module


@pytest.fixture(scope="module")
def fitted(data_home, tmp_path_factory) -> dict:
    """One short training run of the model the guides use, shared by every page."""
    folder = tmp_path_factory.mktemp("docs-fitted")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("DEEPECGKIT_HOME", str(data_home))
        patcher.setattr(torch.cuda, "is_available", lambda: False)
        patcher.setattr(torch.backends.mps, "is_available", lambda: False)
        info = get_dataset_info("af-classification")
        data_module = _data_module()
        model = get_model("kanres")(
            input_channels=info["input_channels"], output_size=info["num_classes"]
        )
        trainer = ECGTrainer(model)
        trainer.fit(
            data_module, epochs=1, checkpoint_dir=folder / "checkpoints", log_dir=folder / "logs"
        )
        trainer.validate(data_module)
        val_targets, val_logits = trainer.get_test_results()[1], trainer.get_test_logits()
        trainer.test(data_module)
        y_pred, y_true, y_prob = trainer.get_test_results()
        return {
            "checkpoint": Path(trainer.best_checkpoint_path),
            "metrics": folder / "logs" / "metrics.csv",
            "arrays": {
                "y_pred": y_pred,
                "y_true": y_true,
                "y_prob": y_prob,
                "predictions": y_pred,
                "targets": y_true,
                "probabilities": y_prob,
                "test_targets": y_true,
                "test_probabilities": y_prob,
                "test_logits": trainer.get_test_logits(),
                "val_targets": val_targets,
                "val_logits": val_logits,
            },
        }


@pytest.fixture
def assumed(fitted) -> dict:
    """What a page of fragments takes for granted: a fitted trainer, arrays, files on disk."""
    info = get_dataset_info("af-classification")
    data_module = _data_module()
    trainer = ECGTrainer.load_checkpoint(fitted["checkpoint"])
    dataset = data_module.dataset
    signals = torch.stack([dataset[index][0] for index in range(32)])
    labels = np.array([int(dataset[index][1]) for index in range(32)])
    long_signal = np.random.default_rng(0).standard_normal(300 * 120).astype(np.float32)

    stored = torch.load(fitted["checkpoint"], weights_only=True)
    for relative in CHECKPOINT_PATHS:
        Path(relative).parent.mkdir(parents=True, exist_ok=True)
        torch.save(stored, relative)
    legacy = {key: value for key, value in stored.items() if not key.endswith("_config")}
    torch.save(legacy, "legacy.ckpt")
    torch.save(stored["model_state_dict"], "weights.pt")
    Path("logs").mkdir()
    Path("logs/metrics.csv").write_text(fitted["metrics"].read_text())
    np.save("ecg.npy", long_signal[: 250 * 60])
    np.save("recording.npy", signals[0, 0].numpy())
    wfdb.wrsamp(
        "record",
        fs=300,
        units=["mV"],
        sig_name=["ECG"],
        p_signal=long_signal[:9000, None].astype(np.float64),
        fmt=["16"],
    )

    return {
        **fitted["arrays"],
        "ECGTrainer": ECGTrainer,
        "get_model": get_model,
        "torch": torch,
        "data_module": data_module,
        "dataset": dataset,
        "dataset_info": info,
        "input_channels": info["input_channels"],
        "num_classes": info["num_classes"],
        "model": trainer.model,
        "model_class": get_model("kanres"),
        "trainer": trainer,
        "config": {"learning_rate": 1e-3},
        "train_config": {"learning_rate": 1e-3},
        "test_loader": data_module.test_dataloader(),
        "dataloader": data_module.test_dataloader(),
        "signal": signals[0],
        "signals": signals,
        "labels": labels,
        "record_ids": np.arange(len(labels)) // 4,
        "torch_tensor": signals,
        "numpy_array": signals.numpy(),
        "train_signals": signals[:24].numpy(),
        "train_labels": labels[:24],
        "test_signals": signals[24:].numpy(),
        "test_labels": labels[24:],
        "target": torch.zeros(len(signals), dtype=torch.long),
        "raw_signal": long_signal[: 360 * 30],
        "long_signal": long_signal,
        "annotation": SimpleNamespace(
            sample=np.array([0, 9000, 18000]), aux_note=["(N", "(AFIB", "(N"], fs=300
        ),
    }


def blocks_of(page: str) -> list[tuple[int, str]]:
    """Return each Python block of a page with the line it starts on."""
    text = (DOCS / page).read_text()
    return [
        (text[: match.start()].count("\n") + 2, match.group(1))
        for match in PYTHON_BLOCK.finditer(text)
    ]


def runnable(block: str) -> str:
    """Drop the lines of a block that stand for something the reader supplies."""
    for placeholder in PLACEHOLDER_LINES:
        block = block.replace(placeholder, "")
    return block


@pytest.mark.parametrize("page", PAGES)
def test_python_blocks_run(page, assumed):
    namespace = dict(assumed)
    ran = 0
    for line, block in blocks_of(page):
        if any(phrase in block for phrase in LEFT_OUT):
            continue
        source = runnable(block)
        try:
            exec(compile(source, f"docs/{page}:{line}", "exec"), namespace)
        except Exception as error:
            pytest.fail(f"docs/{page}, block at line {line}: {type(error).__name__}: {error}")
        ran += 1
    assert ran, f"docs/{page} has no Python block left to run; drop it from PAGES"


def test_every_left_out_phrase_still_matches_a_block():
    """A phrase nothing matches any more is a block that is now run, or was deleted."""
    sources = [source for page in PAGES for _, source in blocks_of(page)]

    for phrase in LEFT_OUT:
        assert any(phrase in source for source in sources), phrase
