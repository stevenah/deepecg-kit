"""Regression tests for previously silent correctness bugs.

Each test here corresponds to a defect that produced plausible but wrong output
rather than an error, so the assertions target the specific failure mode.
"""

import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from deepecgkit.datasets.modules import ECGDataModule
from deepecgkit.datasets.unified_af import UnifiedAFDataset, _RemappedDataset
from deepecgkit.evaluation.metrics import calculate_classification_metrics, resolve_average
from deepecgkit.models.factory import (
    build_model,
    build_model_from_checkpoint,
    infer_output_size,
)
from deepecgkit.registry import get_model_names
from deepecgkit.training import ECGTrainer


class _ToyDataset(Dataset):
    """Small in-memory dataset with configurable grouping."""

    def __init__(self, n_records=12, segments_per_record=4, n_classes=2, with_groups=True):
        rng = np.random.default_rng(0)
        self.signals = []
        self.labels = []
        self.record_names = []
        for record in range(n_records):
            label = record % n_classes
            for _ in range(segments_per_record):
                self.signals.append(rng.normal(size=(1, 256)).astype(np.float32))
                self.labels.append(label)
                if with_groups:
                    self.record_names.append(f"rec{record:03d}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return torch.from_numpy(self.signals[idx]), torch.tensor(self.labels[idx])


class _ToyModule(torch.nn.Module):
    def __init__(self, input_channels=1, output_size=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(input_channels * 256, output_size)
        )

    def forward(self, x):
        return self.net(x)


def _make_data_module(dataset):
    module = ECGDataModule(
        dataset=dataset, batch_size=8, num_workers=0, val_split=0.25, test_split=0.25, verbose=False
    )
    module.setup()
    return module


class TestBestCheckpointRestored:
    """fit() must leave the model holding best-validation weights, not final-epoch weights."""

    def test_weights_match_best_checkpoint(self, tmp_path):
        torch.manual_seed(0)
        dataset = _ToyDataset()
        data_module = _make_data_module(dataset)
        trainer = ECGTrainer(
            model=_ToyModule(),
            train_config={"learning_rate": 0.05, "scheduler": {"factor": 0.5, "patience": 1}},
            device="cpu",
        )
        trainer.fit(
            data_module,
            epochs=6,
            early_stopping_patience=6,
            checkpoint_dir=str(tmp_path / "ckpt"),
            progress_bar=False,
        )

        assert trainer.best_checkpoint_path is not None
        saved = torch.load(trainer.best_checkpoint_path, map_location="cpu", weights_only=False)
        for key, tensor in trainer.model.state_dict().items():
            assert torch.allclose(tensor.cpu(), saved["model_state_dict"][key]), (
                f"Parameter {key} does not match the best checkpoint; evaluation would "
                "be run on final-epoch weights."
            )

    def test_restore_can_be_disabled(self, tmp_path):
        torch.manual_seed(0)
        data_module = _make_data_module(_ToyDataset())
        trainer = ECGTrainer(
            model=_ToyModule(),
            train_config={"learning_rate": 0.05, "scheduler": {"factor": 0.5, "patience": 1}},
            device="cpu",
        )
        trainer.fit(
            data_module,
            epochs=4,
            checkpoint_dir=str(tmp_path / "ckpt"),
            progress_bar=False,
            restore_best=False,
        )
        assert trainer.restore_best_checkpoint() is True


class TestGroupedSplitting:
    """Segments of one recording must never be divided across splits."""

    def _split_groups(self, module, dataset):
        out = []
        for subset in (module.train_dataset, module.val_dataset, module.test_dataset):
            out.append({dataset.record_names[i] for i in subset.indices})
        return out

    def test_no_group_spans_splits(self):
        dataset = _ToyDataset()
        module = _make_data_module(dataset)
        train, val, test = self._split_groups(module, dataset)
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)

    def test_missing_groups_warns(self):
        dataset = _ToyDataset(with_groups=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_data_module(dataset)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught), (
            "Sample-level splitting must warn, otherwise leakage is silent."
        )

    def test_group_id_length_mismatch_raises(self):
        dataset = _ToyDataset()
        dataset.record_names = dataset.record_names[:-1]
        with pytest.raises(ValueError, match="group ids"):
            _make_data_module(dataset)


MODEL_INPUT_LENGTHS = {"afmodel": 9000}
DEFAULT_INPUT_LENGTH = 1000


class TestModelFactory:
    """Every registered model must round-trip through a checkpoint unchanged."""

    @pytest.mark.parametrize("name", sorted(get_model_names()))
    @pytest.mark.parametrize("shape", [(1, 4), (12, 5)])
    def test_checkpoint_roundtrip(self, name, shape, tmp_path):
        input_channels, output_size = shape
        torch.manual_seed(0)
        model = build_model(name, input_channels=input_channels, output_size=output_size)
        model.eval()

        path = tmp_path / f"{name}-{input_channels}x{output_size}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "train_config": {},
                "model_config": {
                    "name": name,
                    "input_channels": input_channels,
                    "output_size": output_size,
                },
            },
            path,
        )

        restored, _ = build_model_from_checkpoint(path, name=name)
        restored.eval()

        length = MODEL_INPUT_LENGTHS.get(name, DEFAULT_INPUT_LENGTH)
        x = torch.randn(2, input_channels, length)
        with torch.no_grad():
            assert torch.allclose(model(x), restored(x), atol=1e-6)

    @pytest.mark.parametrize("name", sorted(get_model_names()))
    def test_output_size_inferable(self, name):
        model = build_model(name, input_channels=2, output_size=7)
        assert infer_output_size(model.state_dict()) == 7


class TestMetricAveraging:
    """Two-class precision/recall/F1 are positive-class values, not macro averages."""

    def test_resolve_average_reports_binary(self):
        y_true = np.array([0, 0, 1, 1])
        assert resolve_average(y_true) == "binary"
        assert resolve_average(np.array([0, 1, 2])) == "macro"
        assert resolve_average(y_true, average="macro") == "macro"

    def test_binary_and_macro_differ(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.2])
        binary = calculate_classification_metrics(y_true, y_pred, average="binary")["f1"]
        macro = calculate_classification_metrics(y_true, y_pred, average="macro")["f1"]
        assert binary != macro

    def test_default_is_unchanged(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.2])
        assert (
            calculate_classification_metrics(y_true, y_pred)["f1"]
            == calculate_classification_metrics(y_true, y_pred, average="binary")["f1"]
        )


class _NamedSource(Dataset):
    """Minimal stand-in for a source dataset with record names and labels."""

    def __init__(self, labels, record_names):
        self.labels = labels
        self.record_names = record_names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.zeros(1, 8), torch.tensor(self.labels[idx])


class TestUnifiedAFGrouping:
    """The merged AF dataset must expose record names, or refuse to be built."""

    def test_remapped_dataset_forwards_record_names(self):
        source = _NamedSource([0, 1, 2, 3], ["a", "b", "c", "d"])
        remapped = _RemappedDataset(source, {0: 0, 1: 1, 2: None, 3: 1})

        assert len(remapped) == 3
        assert remapped.record_names == ["a", "b", "d"]
        assert len(remapped.record_names) == len(remapped)

    def test_collect_record_names_namespaces_sources(self):
        merged = UnifiedAFDataset.__new__(UnifiedAFDataset)
        merged.loaded_dataset_names = ["physionet2017", "mitbih_afdb"]
        merged.datasets = [
            _NamedSource([0, 1], ["A00001", "A00002"]),
            _NamedSource([0, 1], ["A00001", "04015"]),
        ]

        merged._collect_record_names()

        assert merged.record_names == [
            "physionet2017/A00001",
            "physionet2017/A00002",
            "mitbih_afdb/A00001",
            "mitbih_afdb/04015",
        ]
        assert len(set(merged.record_names)) == 4

    def test_missing_source_record_names_raises(self):
        class _Unnamed(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return torch.zeros(1, 8), torch.tensor(0)

        merged = UnifiedAFDataset.__new__(UnifiedAFDataset)
        merged.loaded_dataset_names = ["mitbih_afdb"]
        merged.datasets = [_Unnamed()]

        with pytest.raises(RuntimeError, match="leak"):
            merged._collect_record_names()
