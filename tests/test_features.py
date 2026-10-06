"""
Tests for deepecgkit.features module.
"""

import warnings

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from deepecgkit.datasets import ECGDataModule
from deepecgkit.features import FeatureExtractor, FeatureSet, _resolve_device, extract_features
from deepecgkit.models.factory import build_model
from deepecgkit.registry import get_model_names

SIGNAL_FIXTURES = {"afmodel": "sample_batch_ecg_30s"}

LENGTH_LIMITED_MODELS = ["transformer", "dualnet", "medformer"]


class ToyECGDataset(Dataset):
    """Deterministic single-lead dataset yielding (signal, label) pairs."""

    def __init__(self, num_samples: int = 12, length: int = 3000):
        self.num_samples = num_samples
        self.length = length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        index = int(idx)
        generator = torch.Generator().manual_seed(index)
        signal = torch.randn(1, self.length, generator=generator)
        return signal, torch.tensor(index % 4)


@pytest.fixture
def kanres_model():
    """A small registered model with a known feature dimension."""
    return build_model("kanres", input_channels=1, output_size=4)


@pytest.fixture
def cpu_extractor(kanres_model):
    """A CPU extractor over the kanres model."""
    return FeatureExtractor(kanres_model, model_name="kanres", device="cpu", batch_size=4)


@pytest.fixture
def kanres_checkpoint(kanres_model, temp_dir):
    """An ECGTrainer-shaped checkpoint on disk."""
    path = temp_dir / "kanres.ckpt"
    torch.save(
        {
            "model_state_dict": kanres_model.state_dict(),
            "model_config": {"name": "kanres", "input_channels": 1, "output_size": 4},
            "train_config": {},
        },
        str(path),
    )
    return path


class TestFeatureContract:
    """Every registered model must return (batch, feature_dim) embeddings."""

    @pytest.mark.parametrize("model_name", get_model_names())
    def test_extract_features_shape(self, model_name, request):
        """extract_features returns a 2D tensor matching feature_dim."""
        signals = request.getfixturevalue(SIGNAL_FIXTURES.get(model_name, "sample_batch_ecg"))
        model = build_model(model_name, input_channels=1, output_size=4)
        model.eval()

        with torch.no_grad():
            features = model.extract_features(signals)

        assert features.dim() == 2
        assert features.shape == (signals.shape[0], model.feature_dim)
        assert torch.isfinite(features).all()

    @pytest.mark.parametrize("model_name", get_model_names())
    def test_forward_matches_output_size(self, model_name, request):
        """forward returns one row of logits per sample."""
        signals = request.getfixturevalue(SIGNAL_FIXTURES.get(model_name, "sample_batch_ecg"))
        model = build_model(model_name, input_channels=1, output_size=4)
        model.eval()

        with torch.no_grad():
            outputs = model(signals)

        assert outputs.shape == (signals.shape[0], 4)

    @pytest.mark.parametrize("model_name", get_model_names())
    def test_forward_routes_through_extract_features(self, model_name, request):
        """forward must not duplicate the feature path."""
        signals = request.getfixturevalue(SIGNAL_FIXTURES.get(model_name, "sample_batch_ecg"))
        model = build_model(model_name, input_channels=1, output_size=4)
        model.eval()

        calls = []
        original = model.extract_features

        def spy(x):
            calls.append(1)
            return original(x)

        model.extract_features = spy
        with torch.no_grad():
            model(signals)

        assert calls, f"{model_name}.forward does not call extract_features"


class TestLengthConstraints:
    """Architectures with a positional-encoding ceiling reject long signals."""

    @pytest.mark.parametrize("model_name", LENGTH_LIMITED_MODELS)
    def test_rejects_signals_above_ceiling(self, model_name, sample_batch_ecg_30s):
        """Signals longer than 5000 samples raise rather than truncate."""
        model = build_model(model_name, input_channels=1, output_size=4)
        model.eval()

        with pytest.raises(RuntimeError), torch.no_grad():
            model.extract_features(sample_batch_ecg_30s)


class TestModelGuards:
    """Guards against silently wrong embeddings."""

    def test_deep_res_cnn_rejects_wrong_lead_count(self):
        """A lead-count mismatch raises instead of returning a plausible vector."""
        model = build_model("deep-res-cnn", input_channels=12, output_size=4)
        model.eval()

        with torch.no_grad():
            assert model.extract_features(torch.randn(2, 12, 1000)).shape == (2, 128)

        with pytest.raises(ValueError, match="built for 12 leads but received 13"):
            model.extract_features(torch.randn(2, 13, 1000))

    def test_deep_res_cnn_rejects_non_3d_input(self):
        """A missing lead axis is reported clearly."""
        model = build_model("deep-res-cnn", input_channels=12, output_size=4)

        with pytest.raises(ValueError, match="expects input of shape"):
            model.extract_features(torch.randn(12, 1000))

    @pytest.mark.parametrize(
        "recording_length,expected_minimum", [(6, 1393), (10, 1393), (30, 7971)]
    )
    def test_af_model_minimum_input_length(self, recording_length, expected_minimum):
        """The advertised minimum is the true architectural boundary."""
        model = build_model(
            "afmodel", input_channels=1, output_size=4, recording_length=recording_length
        )
        model.eval()

        assert model.min_input_length == expected_minimum

        with torch.no_grad():
            features = model.extract_features(torch.randn(1, 1, expected_minimum))
        assert features.shape == (1, model.feature_dim)

        with pytest.raises(ValueError, match="needs at least"):
            model.extract_features(torch.randn(1, 1, expected_minimum - 1))

    def test_simple_cnn_forward_uses_extract_features(self, sample_batch_ecg):
        """forward equals classifier(extract_features(x))."""
        model = build_model("simple-cnn", input_channels=1, output_size=4)
        model.eval()

        with torch.no_grad():
            direct = model(sample_batch_ecg)
            composed = model.classifier(model.extract_features(sample_batch_ecg))

        assert torch.allclose(direct, composed)


class TestFeatureExtractorInputs:
    """The extractor accepts every input kind the library exposes."""

    def test_numpy_input(self, cpu_extractor):
        """A 3D array is extracted as a batch."""
        result = cpu_extractor.extract(np.random.randn(6, 1, 3000).astype(np.float32))
        assert result.features.shape == (6, 64)
        assert result.labels is None

    def test_tensor_input(self, cpu_extractor):
        """A tensor is accepted like an array."""
        assert cpu_extractor.extract(torch.randn(6, 1, 3000)).features.shape == (6, 64)

    def test_one_dimensional_input(self, cpu_extractor):
        """A bare signal becomes a single single-lead sample."""
        assert cpu_extractor.extract(np.random.randn(3000)).features.shape == (1, 64)

    def test_two_dimensional_input(self, cpu_extractor):
        """Rows of a 2D array become independent single-lead samples."""
        assert cpu_extractor.extract(np.random.randn(5, 3000)).features.shape == (5, 64)

    def test_file_input(self, cpu_extractor, temp_dir):
        """A .npy file is read and extracted."""
        path = temp_dir / "signals.npy"
        np.save(str(path), np.random.randn(4, 1, 3000))
        assert cpu_extractor.extract(str(path)).features.shape == (4, 64)

    def test_dataset_input_keeps_labels_aligned(self, cpu_extractor):
        """Labels come back in dataset order."""
        result = cpu_extractor.extract(ToyECGDataset(12))
        assert result.features.shape == (12, 64)
        assert np.array_equal(result.labels, np.array([i % 4 for i in range(12)]))

    def test_dataloader_input(self, cpu_extractor):
        """An existing loader is used as supplied."""
        loader = DataLoader(ToyECGDataset(12), batch_size=5, shuffle=False)
        assert cpu_extractor.extract(loader).features.shape == (12, 64)

    def test_shuffled_dataloader_warns(self, cpu_extractor):
        """A shuffling loader scrambles row order, so it warns."""
        loader = DataLoader(ToyECGDataset(8), batch_size=4, shuffle=True)
        with pytest.warns(RuntimeWarning, match="shuffles"):
            cpu_extractor.extract(loader)

    @pytest.mark.parametrize("split", ["train", "val", "test", "all"])
    def test_data_module_splits(self, cpu_extractor, split):
        """Every split can be extracted and carries labels."""
        module = ECGDataModule(
            dataset=ToyECGDataset(40),
            batch_size=8,
            val_split=0.2,
            test_split=0.2,
            num_workers=0,
            verbose=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module.setup()
            result = cpu_extractor.extract(module, split=split)

        assert result.features.shape[1] == 64
        assert result.labels.shape[0] == result.features.shape[0]

    def test_data_module_train_split_order_is_stable(self, cpu_extractor):
        """The train split must not inherit the module's shuffling loader."""
        module = ECGDataModule(
            dataset=ToyECGDataset(40), batch_size=8, num_workers=0, verbose=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module.setup()
            first = cpu_extractor.extract(module, split="train")
            second = cpu_extractor.extract(module, split="train")

        assert np.array_equal(first.labels, second.labels)
        assert np.array_equal(first.features, second.features)

    def test_data_module_is_set_up_lazily(self, cpu_extractor):
        """A module that has not been set up is prepared on demand."""
        module = ECGDataModule(
            dataset=ToyECGDataset(10), batch_size=5, num_workers=0, verbose=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cpu_extractor.extract(module, split="all")

        assert result.features.shape == (10, 64)

    def test_empty_split_returns_empty_features(self, cpu_extractor):
        """An empty subset yields a correctly shaped empty array."""
        result = cpu_extractor.extract(Subset(ToyECGDataset(10), []))
        assert result.features.shape == (0, 64)
        assert result.labels is None

    def test_unsupported_input_type(self, cpu_extractor):
        """An unusable input names the accepted kinds."""
        with pytest.raises(TypeError, match="Unsupported input type"):
            cpu_extractor.extract(42)


class TestFeatureExtractorBehaviour:
    """Batching, precision, and non-interference with the caller's model."""

    @pytest.mark.parametrize("batch_size", [1, 3, 6, 64])
    def test_batch_size_does_not_change_results(self, kanres_model, batch_size):
        """Results are independent of how the data is chunked."""
        signals = np.random.randn(6, 1, 3000).astype(np.float32)
        reference = FeatureExtractor(kanres_model, device="cpu", batch_size=6).extract(signals)
        chunked = FeatureExtractor(kanres_model, device="cpu", batch_size=batch_size).extract(
            signals
        )
        assert np.allclose(reference.features, chunked.features, atol=1e-6)

    def test_float16_output(self, kanres_model):
        """The requested precision is honoured."""
        extractor = FeatureExtractor(kanres_model, device="cpu", dtype="float16")
        assert extractor.extract(np.random.randn(4, 1, 3000)).features.dtype == np.float16

    def test_unknown_dtype(self, cpu_extractor):
        """An unsupported dtype is rejected before any work happens."""
        cpu_extractor.dtype = "float64"
        with pytest.raises(ValueError, match="Unknown dtype"):
            cpu_extractor.extract(np.random.randn(2, 1, 3000))

    def test_unknown_split(self, cpu_extractor):
        """An unsupported split is rejected."""
        with pytest.raises(ValueError, match="Unknown split"):
            cpu_extractor.extract(np.random.randn(2, 1, 3000), split="holdout")

    def test_training_mode_is_restored(self, kanres_model):
        """A model left in training mode stays that way."""
        kanres_model.train()
        FeatureExtractor(kanres_model, device="cpu").extract(np.random.randn(2, 1, 3000))
        assert kanres_model.training

    def test_eval_mode_is_preserved(self, kanres_model):
        """A model in eval mode is not switched to training."""
        kanres_model.eval()
        FeatureExtractor(kanres_model, device="cpu").extract(np.random.randn(2, 1, 3000))
        assert not kanres_model.training

    def test_gradients_are_not_tracked(self, kanres_model):
        """Extraction never builds a graph or populates gradients."""
        FeatureExtractor(kanres_model, device="cpu").extract(np.random.randn(2, 1, 3000))
        assert all(p.grad is None for p in kanres_model.parameters())
        assert all(p.requires_grad for p in kanres_model.parameters())

    def test_extraction_is_deterministic(self, cpu_extractor):
        """Two passes over the same data agree exactly."""
        signals = np.random.randn(6, 1, 3000).astype(np.float32)
        assert np.array_equal(
            cpu_extractor.extract(signals).features, cpu_extractor.extract(signals).features
        )

    def test_non_two_dimensional_features_are_rejected(self):
        """A model breaking the contract is reported, not silently reshaped."""

        class BadModel(nn.Module):
            feature_dim = 8

            def extract_features(self, x):
                return torch.randn(x.shape[0], 4, 2)

        extractor = FeatureExtractor(BadModel(), device="cpu")
        with pytest.raises(RuntimeError, match="2-D"):
            extractor.extract(np.random.randn(2, 1, 3000))

    def test_feature_dim_requires_declaration(self):
        """A model without feature_dim cannot be used for extraction."""

        class Bare(nn.Module):
            def extract_features(self, x):
                return x

        with pytest.raises(AttributeError, match="does not declare feature_dim"):
            _ = FeatureExtractor(Bare(), device="cpu").feature_dim

    def test_input_channels_inferred(self, cpu_extractor):
        """The lead count is recovered from convolutional weights."""
        assert cpu_extractor.input_channels == 1

    @pytest.mark.parametrize(
        "name,expected",
        [("cpu", "cpu"), ("auto", None), ("gpu", "cuda")],
    )
    def test_resolve_device(self, name, expected, monkeypatch):
        """Device names, including the CLI accelerator vocabulary, resolve."""
        if name == "gpu":
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        resolved = _resolve_device(name)
        if expected is not None:
            assert resolved.type == expected
        else:
            assert resolved.type in {"cpu", "cuda", "mps"}


class TestFeatureExtractorLoading:
    """Building an extractor from stored weights."""

    def test_from_checkpoint_matches_in_memory_model(self, kanres_model, kanres_checkpoint):
        """Loaded weights reproduce the original model's embeddings."""
        signals = np.random.randn(4, 1, 3000).astype(np.float32)
        reference = FeatureExtractor(kanres_model, device="cpu").extract(signals)
        loaded = FeatureExtractor.from_checkpoint(
            kanres_checkpoint, "kanres", device="cpu"
        ).extract(signals)

        assert np.allclose(reference.features, loaded.features, atol=1e-6)
        assert loaded.model_name == "kanres"

    def test_from_checkpoint_reads_recorded_shapes(self, kanres_checkpoint):
        """input_channels and output_size come from the recorded config."""
        extractor = FeatureExtractor.from_checkpoint(kanres_checkpoint, "kanres", device="cpu")
        assert extractor.feature_dim == 64

    def test_from_pretrained_with_local_weights(self, kanres_model, temp_dir):
        """A bare state dict on disk is accepted."""
        path = temp_dir / "weights.pt"
        torch.save(kanres_model.state_dict(), str(path))
        extractor = FeatureExtractor.from_pretrained(path, model_name="kanres", device="cpu")
        assert extractor.feature_dim == 64

    def test_from_pretrained_requires_resolvable_model(self, temp_dir):
        """An unregistered weight name without a model name is rejected."""
        path = temp_dir / "unknown.pt"
        torch.save({}, str(path))
        with pytest.raises(ValueError, match="Could not determine which model"):
            FeatureExtractor.from_pretrained(path)


class TestExtractFeaturesFunction:
    """The one-call convenience wrapper."""

    def test_with_model(self, kanres_model):
        """An already-built model is used directly."""
        result = extract_features(np.random.randn(4, 1, 3000), model=kanres_model, device="cpu")
        assert result.features.shape == (4, 64)

    def test_with_checkpoint(self, kanres_checkpoint):
        """A checkpoint path is loaded."""
        result = extract_features(
            np.random.randn(4, 1, 3000),
            checkpoint=kanres_checkpoint,
            model_name="kanres",
            device="cpu",
        )
        assert result.features.shape == (4, 64)

    def test_checkpoint_supplies_its_own_model_name(self, kanres_checkpoint):
        """A checkpoint that records its architecture needs no model name."""
        result = extract_features(
            np.random.randn(2, 1, 3000), checkpoint=kanres_checkpoint, device="cpu"
        )
        assert result.features.shape == (2, 64)
        assert result.model_name == "kanres"

    def test_checkpoint_without_a_recorded_name_requires_one(self, kanres_model, temp_dir):
        """Bare weights do not say which architecture they belong to."""
        path = temp_dir / "bare.pt"
        torch.save(kanres_model.state_dict(), str(path))
        with pytest.raises(ValueError, match="does not record which architecture"):
            extract_features(np.random.randn(2, 1, 3000), checkpoint=path)

    def test_checkpoint_rejects_a_contradicting_model_name(self, kanres_checkpoint):
        """Naming the wrong architecture is reported as that, in one line."""
        with pytest.raises(ValueError, match="was trained as 'kanres', not 'resnet'"):
            extract_features(
                np.random.randn(2, 1, 3000), checkpoint=kanres_checkpoint, model_name="resnet"
            )

    def test_a_plain_module_is_rejected_with_a_pointer_to_the_base_class(self):
        """A model without a feature path explains what it is missing."""
        plain = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3000, 4))
        with pytest.raises(TypeError, match="ECGModelBase"):
            extract_features(np.random.randn(2, 1, 3000), model=plain, device="cpu")

    def test_requires_exactly_one_source(self, kanres_model, kanres_checkpoint):
        """Zero or several weight sources are rejected."""
        with pytest.raises(ValueError, match="exactly one"):
            extract_features(np.random.randn(2, 1, 3000))
        with pytest.raises(ValueError, match="exactly one"):
            extract_features(
                np.random.randn(2, 1, 3000), model=kanres_model, checkpoint=kanres_checkpoint
            )


class TestSklearnCompatibility:
    """The extractor behaves as a scikit-learn transformer."""

    def test_pipeline_with_classifier(self, cpu_extractor):
        """A linear probe trains on the embeddings."""
        signals = np.random.randn(16, 1, 3000).astype(np.float32)
        labels = np.array([0, 1, 2, 3] * 4)
        pipeline = Pipeline([("ecg", cpu_extractor), ("clf", LogisticRegression(max_iter=100))])
        pipeline.fit(signals, labels)
        assert pipeline.predict(signals).shape == (16,)

    def test_fit_returns_self(self, cpu_extractor):
        """fit is a no-op returning the estimator."""
        assert cpu_extractor.fit(None) is cpu_extractor

    def test_transform_returns_array(self, cpu_extractor):
        """transform yields a plain array."""
        features = cpu_extractor.transform(np.random.randn(4, 1, 3000))
        assert isinstance(features, np.ndarray)
        assert features.shape == (4, 64)

    def test_fit_transform(self, cpu_extractor):
        """fit_transform comes from TransformerMixin."""
        assert cpu_extractor.fit_transform(np.random.randn(4, 1, 3000)).shape == (4, 64)

    def test_get_params_round_trip(self, cpu_extractor):
        """Constructor arguments are stored unmodified, as sklearn requires."""
        params = cpu_extractor.get_params()
        assert params["device"] == "cpu"
        assert params["batch_size"] == 4
        assert clone(cpu_extractor).get_params()["device"] == "cpu"

    def test_get_feature_names_out(self, cpu_extractor):
        """Feature names are prefixed with the model name."""
        names = cpu_extractor.get_feature_names_out()
        assert len(names) == 64
        assert names[0] == "kanres_0"


class TestFeatureSet:
    """The result object and its serialisation."""

    @pytest.fixture
    def feature_set(self):
        """A small populated result."""
        return FeatureSet(
            np.arange(12, dtype=np.float32).reshape(3, 4),
            np.array([0, 1, 2]),
            "kanres",
        )

    def test_properties(self, feature_set):
        """Dimensions are derived from the array."""
        assert feature_set.feature_dim == 4
        assert len(feature_set) == 3

    def test_array_protocol(self, feature_set):
        """The object stands in for its features array."""
        assert np.array_equal(np.asarray(feature_set), feature_set.features)
        assert np.asarray(feature_set, dtype=np.float64).dtype == np.float64

    def test_save_npz(self, feature_set, temp_dir):
        """The npz form keeps labels and metadata."""
        path = temp_dir / "out.npz"
        feature_set.save(path)
        loaded = np.load(str(path), allow_pickle=True)
        assert np.array_equal(loaded["features"], feature_set.features)
        assert np.array_equal(loaded["labels"], feature_set.labels)
        assert str(loaded["model_name"]) == "kanres"
        assert int(loaded["feature_dim"]) == 4

    def test_save_npy(self, feature_set, temp_dir):
        """The npy form stores features only."""
        path = temp_dir / "out.npy"
        feature_set.save(path)
        assert np.array_equal(np.load(str(path)), feature_set.features)

    def test_save_csv(self, feature_set, temp_dir):
        """The csv form names the columns and appends the label."""
        import pandas as pd

        path = temp_dir / "out.csv"
        feature_set.save(path)
        frame = pd.read_csv(str(path))
        assert list(frame.columns) == ["f0", "f1", "f2", "f3", "label"]
        assert len(frame) == 3

    def test_save_multi_label_csv(self, temp_dir):
        """Multi-label targets become one column each."""
        result = FeatureSet(np.zeros((2, 3), dtype=np.float32), np.ones((2, 2)), "ptbxl")
        path = temp_dir / "multi.csv"
        result.save(path)

        import pandas as pd

        assert list(pd.read_csv(str(path)).columns) == ["f0", "f1", "f2", "label_0", "label_1"]

    def test_save_without_labels(self, temp_dir):
        """Missing labels are simply omitted."""
        FeatureSet(np.zeros((2, 3), dtype=np.float32)).save(temp_dir / "bare.npz")
        assert "labels" not in np.load(str(temp_dir / "bare.npz")).files

    def test_save_unsupported_format(self, feature_set, temp_dir):
        """An unknown suffix names the supported ones."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            feature_set.save(temp_dir / "out.parquet")
