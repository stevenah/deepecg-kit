"""Tests for the representation-quality protocol."""

import json
from typing import ClassVar, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation import (
    PROBE_METRICS,
    RANDOM_INIT,
    ProbeResult,
    TransferMatrix,
    build_probe,
    cross_dataset_transfer,
    embed_2d,
    linear_probe,
    probe_encoder,
)
from deepecgkit.evaluation.visualization import plot_embedding, plot_transfer_matrix
from deepecgkit.features import FeatureExtractor, FeatureSet
from deepecgkit.models.factory import build_model

MODEL_NAME = "kanres"
SIGNAL_LENGTH = 512
TOY_SAMPLES = 64
TOY_CLASSES = 4
SEPARABLE_FLOOR = 0.95
CHANCE_CEILING = 0.7
MULTILABEL_FLOOR = 0.7


@pytest.fixture(autouse=True)
def headless_matplotlib():
    """Render without a display and close every figure a test leaves open."""
    matplotlib.use("Agg", force=True)
    yield
    plt.close("all")


class ToyECGDataset(Dataset):
    """A small in-memory single-lead dataset whose class shifts the signal's frequency."""

    CLASS_LABELS: ClassVar[List[str]] = ["slow", "medium", "fast", "very fast"]

    def __init__(self, seed: int, n_samples: int = TOY_SAMPLES, offset: float = 0.0):
        rng = np.random.default_rng(seed)
        self.labels = np.tile(np.arange(TOY_CLASSES), n_samples // TOY_CLASSES)
        self.record_names = [f"toy{index:03d}" for index in range(len(self.labels))]
        time = np.linspace(0.0, 4.0, SIGNAL_LENGTH)
        signals = np.empty((len(self.labels), 1, SIGNAL_LENGTH), dtype=np.float32)
        for index, label in enumerate(self.labels):
            frequency = 2.0 + 1.5 * label + offset
            signals[index, 0] = np.sin(2 * np.pi * frequency * time) + rng.normal(
                0.0, 0.2, SIGNAL_LENGTH
            )
        self.signals = signals

    @property
    def class_names(self) -> List[str]:
        """Names of the four classes."""
        return self.CLASS_LABELS

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Return one signal and its class index."""
        return (
            torch.from_numpy(self.signals[index]),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


def _module(dataset: Dataset) -> ECGDataModule:
    """Wrap an in-memory dataset in a data module with a held-out split."""
    module = ECGDataModule(
        dataset=dataset,
        batch_size=16,
        num_workers=0,
        val_split=0.2,
        test_split=0.25,
        seed=7,
        verbose=False,
    )
    module.setup()
    return module


@pytest.fixture
def toy_modules():
    """Two in-memory corpora that differ only in a frequency offset."""
    return {
        "corpus-a": _module(ToyECGDataset(seed=1)),
        "corpus-b": _module(ToyECGDataset(seed=2, offset=0.6)),
    }


@pytest.fixture
def toy_checkpoint(temp_dir):
    """An ECGTrainer-shaped checkpoint for a single-lead kanres encoder."""
    torch.manual_seed(0)
    model = build_model(MODEL_NAME, input_channels=1, output_size=TOY_CLASSES)
    path = temp_dir / "corpus-a.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "name": MODEL_NAME,
                "input_channels": 1,
                "output_size": TOY_CLASSES,
            },
            "train_config": {},
        },
        str(path),
    )
    return path


class TestLinearProbe:
    """A standardised logistic probe recovers linearly separable structure and nothing else."""

    def test_separable_features_score_high(self, probe_features):
        """Classes that are linearly separable are recovered almost perfectly."""
        train, test = probe_features
        result = linear_probe(train, test)
        assert result.score > SEPARABLE_FLOOR
        assert result.task == "multiclass"
        assert result.n_train == train[0].shape[0]
        assert result.n_test == test[0].shape[0]
        assert result.feature_dim == train[0].shape[1]
        assert result.note is None

    def test_every_protocol_metric_is_reported(self, probe_features):
        """The result carries every metric, not just the headline one."""
        train, test = probe_features
        result = linear_probe(train, test)
        assert set(result.metrics) == set(PROBE_METRICS)
        assert all(np.isfinite(value) for value in result.metrics.values())

    def test_shuffled_labels_land_near_chance(self, probe_features):
        """A probe fitted on shuffled labels learns nothing about the held-out split."""
        (features, labels), test = probe_features
        shuffled = np.random.default_rng(0).permutation(labels)
        result = linear_probe((features, shuffled), test)
        assert result.score < CHANCE_CEILING

    def test_multilabel_returns_a_macro_auroc(self, probe_features_multilabel):
        """Independent sigmoids are scored per label rather than collapsed by an argmax."""
        train, test = probe_features_multilabel
        result = linear_probe(train, test)
        assert result.task == "multilabel"
        assert result.score > MULTILABEL_FLOOR
        assert result.metrics["accuracy"] <= 1.0

    def test_single_class_target_returns_nan_with_a_note(self, probe_features):
        """A training split holding one class yields a recorded failure, not an exception."""
        (features, _), test = probe_features
        result = linear_probe((features, np.zeros(features.shape[0], dtype=int)), test)
        assert not np.isfinite(result.score)
        assert result.note is not None

    def test_headline_metric_is_selectable(self, probe_features):
        """Asking for a different metric moves the score to that metric."""
        train, test = probe_features
        result = linear_probe(train, test, metric="accuracy")
        assert result.metric == "accuracy"
        assert result.score == pytest.approx(result.metrics["accuracy"])

    def test_unknown_metric_raises(self, probe_features):
        """An unrecognised metric is rejected before any fitting happens."""
        train, test = probe_features
        with pytest.raises(ValueError, match="Unknown probe metric"):
            linear_probe(train, test, metric="rmse")

    def test_mismatched_widths_raise(self, probe_features):
        """Two splits from different encoders cannot be probed together."""
        (features, labels), (test_features, test_labels) = probe_features
        with pytest.raises(ValueError, match="Embedding width differs"):
            linear_probe((features, labels), (test_features[:, :4], test_labels))

    def test_featureset_without_labels_raises(self, probe_features):
        """Embeddings extracted from a bare array carry no labels and cannot be probed."""
        (features, labels), _ = probe_features
        with pytest.raises(ValueError, match="needs labels"):
            linear_probe(FeatureSet(features), FeatureSet(features, labels))

    def test_unsupported_input_raises(self, probe_features):
        """Anything but a FeatureSet or a pair of arrays is rejected."""
        train, _ = probe_features
        with pytest.raises(TypeError, match="Unsupported probe input"):
            linear_probe(train, "features.npz")


class TestStandardisation:
    """Standardisation is part of the protocol, not a cosmetic preprocessing step."""

    def test_scaling_lifts_the_probe(self, probe_features):
        """On columns spanning six orders of magnitude the scaler buys real accuracy."""
        (features, labels), (test_features, test_labels) = probe_features
        scaled = linear_probe((features, labels), (test_features, test_labels))
        bare = LogisticRegression(max_iter=1000, random_state=42).fit(features, labels)
        assert scaled.metrics["accuracy"] > bare.score(test_features, test_labels)

    def test_scaler_is_fitted_on_the_training_split_alone(self, probe_features):
        """The held-out split never contributes to the mean the scaler subtracts."""
        (features, labels), _ = probe_features
        pipeline = build_probe()
        pipeline.fit(features, labels)
        assert pipeline.named_steps["scaler"].mean_ == pytest.approx(features.mean(axis=0))

    def test_multilabel_probe_wraps_one_estimator_per_label(self, probe_features_multilabel):
        """Multi-hot targets get a one-vs-rest estimator rather than a single softmax."""
        pipeline = build_probe(multilabel=True)
        assert hasattr(pipeline.named_steps["probe"], "estimator")


class TestEmbed2d:
    """Projection to two dimensions is reproducible and survives small splits."""

    def test_tsne_returns_two_columns(self, probe_features):
        """t-SNE projects any embedding width down to two coordinates."""
        (features, _), _ = probe_features
        assert embed_2d(features).shape == (features.shape[0], 2)

    def test_tsne_is_deterministic(self, probe_features):
        """The same seed reproduces the same projection."""
        (features, _), _ = probe_features
        first = embed_2d(features[:120], random_state=3)
        second = embed_2d(features[:120], random_state=3)
        assert np.allclose(first, second)

    def test_perplexity_is_clamped_for_small_splits(self, probe_features):
        """A split too small for the requested perplexity is projected anyway."""
        (features, _), _ = probe_features
        assert embed_2d(features[:8], perplexity=30.0).shape == (8, 2)

    def test_pca_is_available_as_the_cheap_alternative(self, probe_features):
        """The PCA method returns coordinates of the same shape."""
        (features, _), _ = probe_features
        assert embed_2d(features, method="pca").shape == (features.shape[0], 2)

    def test_unknown_method_raises(self, probe_features):
        """Only the two supported projections are accepted."""
        (features, _), _ = probe_features
        with pytest.raises(ValueError, match="Unknown embedding method"):
            embed_2d(features, method="umap")

    def test_too_few_samples_raise(self):
        """Two points cannot be projected and say so."""
        with pytest.raises(ValueError, match="at least"):
            embed_2d(np.zeros((2, 8)))

    def test_one_dimensional_input_raises(self):
        """A flat array is not an embedding matrix."""
        with pytest.raises(ValueError, match="two-dimensional"):
            embed_2d(np.zeros(16))


class TestProbeEncoder:
    """A frozen encoder can be probed straight from a data module."""

    def test_probe_encoder_reports_on_the_held_out_split(self, toy_modules):
        """Probing a module fits on its train split and reports on its test split."""
        torch.manual_seed(0)
        extractor = FeatureExtractor(
            build_model(MODEL_NAME, input_channels=1, output_size=TOY_CLASSES),
            model_name=MODEL_NAME,
            device="cpu",
            batch_size=16,
        )
        module = toy_modules["corpus-a"]
        result = probe_encoder(extractor, module, target="corpus-a")
        assert isinstance(result, ProbeResult)
        assert result.target == "corpus-a"
        assert result.n_test == len(module.test_dataset)
        assert result.feature_dim == extractor.feature_dim


class TestTransferMatrix:
    """The matrix carries one cell per source and target, plus the untrained floor."""

    def test_shape_includes_the_baseline_row(self, toy_modules, toy_checkpoint):
        """One row per source checkpoint plus one for the random-init floor."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        assert matrix.values.shape == (2, 2)
        assert matrix.sources == [RANDOM_INIT, "corpus-a"]
        assert matrix.targets == ["corpus-a", "corpus-b"]

    def test_random_init_is_the_first_row(self, toy_modules, toy_checkpoint):
        """The floor leads the table so it reads as the baseline for every column."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        assert matrix.to_frame().index[0] == RANDOM_INIT

    def test_to_frame_labels_the_axes(self, toy_modules, toy_checkpoint):
        """Rows are sources and columns are targets."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        frame = matrix.to_frame()
        assert list(frame.index) == matrix.sources
        assert list(frame.columns) == matrix.targets
        assert frame.loc[RANDOM_INIT, "corpus-b"] == pytest.approx(matrix.values[0, 1])

    def test_every_cell_keeps_its_full_result(self, toy_modules, toy_checkpoint):
        """The headline value is a view on a ProbeResult that keeps every metric."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        result = matrix.result_for("corpus-a", "corpus-b")
        assert result is not None
        assert result.model_name == MODEL_NAME
        assert set(result.metrics) == set(PROBE_METRICS)
        assert matrix.result_for("corpus-a", "missing") is None

    def test_embeddings_are_retained_only_when_asked(self, toy_modules, toy_checkpoint):
        """Held-out embeddings cost memory, so they are kept only for the requested cells."""
        common = {
            "checkpoints": {"corpus-a": toy_checkpoint},
            "model_name": MODEL_NAME,
            "targets": sorted(toy_modules),
            "data_modules": toy_modules,
            "device": "cpu",
            "batch_size": 16,
        }
        assert cross_dataset_transfer(**common).embeddings == {}
        kept = cross_dataset_transfer(**common, keep_embeddings="all")
        assert len(kept.embeddings) == 4
        assert kept.embedding_for("corpus-a", "corpus-b") is not None
        diagonal = cross_dataset_transfer(**common, keep_embeddings="diagonal")
        assert set(diagonal.embeddings) == {
            "random-init->corpus-a",
            "random-init->corpus-b",
            "corpus-a->corpus-a",
        }

    def test_class_names_follow_the_target(self, toy_modules, toy_checkpoint):
        """Each target's class names are recorded so a plot can label its legend."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        assert matrix.class_names["corpus-a"] == ToyECGDataset.CLASS_LABELS

    def test_protocol_settings_are_recorded(self, toy_modules, toy_checkpoint):
        """The run's protocol is kept alongside the numbers, for provenance."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
            seed=11,
        )
        assert matrix.protocol["input_channels"] == 1
        assert matrix.protocol["seed"] == 11
        assert matrix.protocol["random_init"] is True

    def test_json_round_trips(self, toy_modules, toy_checkpoint, temp_dir):
        """The JSON form keeps every cell's metrics and the protocol settings."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        path = temp_dir / "matrix.json"
        matrix.save(path)
        payload = json.loads(path.read_text())
        assert payload["sources"] == matrix.sources
        assert payload["targets"] == matrix.targets
        assert len(payload["cells"]) == 4
        assert payload["protocol"]["input_channels"] == 1

    def test_csv_round_trips(self, toy_modules, toy_checkpoint, temp_dir):
        """The CSV form holds the headline table alone."""
        matrix = cross_dataset_transfer(
            {"corpus-a": toy_checkpoint},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        path = temp_dir / "matrix.csv"
        matrix.save(path)
        reloaded = pd.read_csv(path, index_col=0)
        assert list(reloaded.index) == matrix.sources
        assert list(reloaded.columns) == matrix.targets

    def test_unknown_save_format_raises(self, temp_dir):
        """Only the two documented formats are written."""
        matrix = TransferMatrix(sources=["a"], targets=["b"], values=np.zeros((1, 1)))
        with pytest.raises(ValueError, match="Unsupported output format"):
            matrix.save(temp_dir / "matrix.parquet")

    def test_unknown_metric_raises(self, toy_modules):
        """The metric is checked before any dataset is touched."""
        with pytest.raises(ValueError, match="Unknown probe metric"):
            cross_dataset_transfer({}, MODEL_NAME, targets=["corpus-a"], metric="rmse")

    def test_unknown_keep_mode_raises(self, toy_modules):
        """So is the embedding retention mode."""
        with pytest.raises(ValueError, match="Unknown keep_embeddings mode"):
            cross_dataset_transfer({}, MODEL_NAME, targets=["corpus-a"], keep_embeddings="some")

    def test_nothing_to_probe_raises(self):
        """Dropping both the sources and the floor leaves no rows at all."""
        with pytest.raises(ValueError, match="Nothing to probe"):
            cross_dataset_transfer({}, MODEL_NAME, targets=["corpus-a"], include_random_init=False)


class TestRandomInitBaseline:
    """The untrained floor is reproducible, which is what makes it a baseline."""

    def test_same_seed_reproduces_the_floor(self, toy_modules):
        """Two runs at one seed build the same random encoder and score identically."""
        common = {
            "checkpoints": {},
            "model_name": MODEL_NAME,
            "targets": sorted(toy_modules),
            "data_modules": toy_modules,
            "device": "cpu",
            "batch_size": 16,
            "seed": 5,
        }
        first = cross_dataset_transfer(**common)
        second = cross_dataset_transfer(**common)
        assert np.allclose(first.values, second.values)

    def test_a_different_seed_draws_a_different_encoder(self, toy_modules):
        """The floor describes one particular random network, so the seed selects it.

        The score itself can saturate on an easy target, so the embeddings rather
        than the metric are what must move.
        """
        common = {
            "checkpoints": {},
            "model_name": MODEL_NAME,
            "targets": ["corpus-a"],
            "data_modules": toy_modules,
            "device": "cpu",
            "batch_size": 16,
            "keep_embeddings": "all",
        }
        first = cross_dataset_transfer(**common, seed=5)
        second = cross_dataset_transfer(**common, seed=6)
        left = first.embedding_for(RANDOM_INIT, "corpus-a")
        right = second.embedding_for(RANDOM_INIT, "corpus-a")
        assert not np.allclose(left.features, right.features)

    def test_the_floor_is_finite(self, toy_modules):
        """An untrained encoder still produces a scoreable embedding."""
        matrix = cross_dataset_transfer(
            {},
            MODEL_NAME,
            targets=sorted(toy_modules),
            data_modules=toy_modules,
            device="cpu",
            batch_size=16,
        )
        assert matrix.sources == [RANDOM_INIT]
        assert np.isfinite(matrix.values).all()


class TestRepresentationVisualization:
    """Both new plots write a file and leave no figures open."""

    def test_transfer_matrix_plot_is_written(self, temp_dir):
        """The heatmap renders sources against targets."""
        path = temp_dir / "matrix.png"
        plot_transfer_matrix(
            np.array([[0.5, 0.6], [0.8, 0.7]]),
            [RANDOM_INIT, "corpus-a"],
            ["corpus-a", "corpus-b"],
            baseline_rows=1,
            save_path=str(path),
        )
        assert path.exists() and path.stat().st_size > 0
        assert plt.get_fignums() == []

    def test_transfer_matrix_plot_tolerates_unscored_cells(self, temp_dir):
        """A NaN cell renders blank rather than raising."""
        path = temp_dir / "sparse.png"
        plot_transfer_matrix(
            np.array([[0.5, np.nan], [np.nan, 0.7]]),
            ["a", "b"],
            ["a", "b"],
            save_path=str(path),
        )
        assert path.exists() and path.stat().st_size > 0

    def test_embedding_plot_is_written(self, temp_dir, probe_features):
        """The scatter colours points by class and labels them."""
        (features, labels), _ = probe_features
        path = temp_dir / "embedding.png"
        plot_embedding(
            embed_2d(features[:120]),
            labels[:120],
            class_names=["N", "AF", "AFL", "J"],
            save_path=str(path),
        )
        assert path.exists() and path.stat().st_size > 0
        assert plt.get_fignums() == []

    def test_embedding_plot_accepts_multi_hot_labels(self, temp_dir, probe_features_multilabel):
        """Multi-hot rows are coloured by their first positive label."""
        (features, labels), _ = probe_features_multilabel
        path = temp_dir / "multilabel.png"
        plot_embedding(embed_2d(features[:120]), labels[:120], save_path=str(path))
        assert path.exists() and path.stat().st_size > 0

    def test_embedding_plot_accepts_no_labels(self, temp_dir, probe_features):
        """Unlabelled embeddings still plot, in a single colour."""
        (features, _), _ = probe_features
        path = temp_dir / "unlabelled.png"
        plot_embedding(embed_2d(features[:120]), save_path=str(path))
        assert path.exists() and path.stat().st_size > 0

    def test_embedding_plot_rejects_unreduced_input(self, probe_features):
        """A full-width embedding must go through embed_2d first."""
        (features, _), _ = probe_features
        with pytest.raises(ValueError, match="shape"):
            plot_embedding(features)

    def test_embedding_plot_rejects_misaligned_labels(self, probe_features):
        """Labels that do not line up with the points are rejected."""
        (features, labels), _ = probe_features
        with pytest.raises(ValueError, match="rows"):
            plot_embedding(embed_2d(features[:60]), labels[:10])
