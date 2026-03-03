"""
Tests for deepecgkit.datasets module.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from deepecgkit.datasets import (
    AFClassificationDataset,
    BaseECGDataset,
    ECGDataModule,
    ECGSegmenter,
    ECGStandardizer,
    LTAFDBDataset,
    MITBIHAFDBDataset,
    RhythmAnnotationExtractor,
    UnifiedAFDataset,
)
from deepecgkit.datasets.base import BaseECGDataset as BaseECGDatasetDirect


class TestBaseECGDataset:
    """Test the BaseECGDataset abstract class."""

    def test_get_default_data_dir(self):
        """Test the default data directory path."""
        data_dir = BaseECGDataset.get_default_data_dir()

        assert isinstance(data_dir, Path)
        assert data_dir.name == "baseecgdataset"
        assert ".deepecgkit" in str(data_dir)

    def test_base_dataset_init_params(self):
        """Test BaseECGDataset initialization parameters."""

        class TestDataset(BaseECGDataset):
            def download(self):
                pass

            def _load_data(self):
                pass

            def __getitem__(self, idx):
                return torch.tensor([1.0]), torch.tensor([0])

            def __len__(self):
                return 1

            @property
            def num_classes(self):
                return 2

            @property
            def class_names(self):
                return ["Class0", "Class1"]

        dataset = TestDataset(sampling_rate=500, leads=["I", "II", "III"], download=False)

        assert dataset.sampling_rate == 500
        assert dataset.leads == ["I", "II", "III"]

    def test_base_dataset_metadata(self):
        """Test BaseECGDataset metadata functionality."""

        class TestDataset(BaseECGDataset):
            def download(self):
                pass

            def _load_data(self):
                pass

            def __getitem__(self, idx):
                return torch.tensor([1.0]), torch.tensor([0])

            def __len__(self):
                return 100

            @property
            def num_classes(self):
                return 4

            @property
            def class_names(self):
                return ["N", "A", "O", "~"]

        dataset = TestDataset(sampling_rate=300, leads=["ECG"], download=False)

        metadata = dataset.get_metadata()

        assert metadata["sampling_rate"] == 300
        assert metadata["num_leads"] == 1
        assert metadata["lead_names"] == ["ECG"]
        assert metadata["num_classes"] == 4
        assert metadata["class_names"] == ["N", "A", "O", "~"]
        assert metadata["dataset_size"] == 100


class TestAFClassificationDataset:
    """Test the AFClassificationDataset class."""

    def test_af_dataset_constants(self):
        """Test AF dataset class constants."""
        assert AFClassificationDataset.CLASS_LABELS == ["Normal", "AF", "Other", "Noisy"]
        assert AFClassificationDataset.REFERENCE_FILE == "REFERENCE-v3.csv"
        assert AFClassificationDataset.LABEL_MAPPING == {"N": 0, "A": 1, "O": 2, "~": 3}
        assert AFClassificationDataset.LEADS == ["ECG"]
        assert AFClassificationDataset.NATIVE_SAMPLING_RATE == 300

    def test_af_dataset_init_validation(self, mock_dataset_files):
        """Test AF dataset initialization with validation."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )
        assert dataset.subset == "training"

        with pytest.raises(ValueError, match="subset must be either 'training' or 'validation'"):
            AFClassificationDataset(
                data_dir=str(mock_dataset_files), subset="invalid", download=False
            )

    def test_af_dataset_properties(self, mock_dataset_files):
        """Test AF dataset properties."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        assert dataset.num_classes == 4
        assert dataset.class_names == ["Normal", "AF", "Other", "Noisy"]

    @patch("deepecgkit.datasets.af_classification.download_file")
    def test_af_dataset_download(self, mock_download, mock_dataset_files):
        """Test AF dataset download functionality."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        with patch("deepecgkit.datasets.af_classification.zipfile.ZipFile"):
            dataset.download()

        assert mock_download.called
        _call_args = mock_download.call_args

    def test_af_dataset_with_mock_data(self, mock_dataset_files):
        """Test AF dataset with mock data files."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        assert dataset is not None
        assert dataset.num_classes == 4

        assert dataset.reference_data is not None

    def test_af_dataset_getitem(self, mock_dataset_files):
        """Test AF dataset __getitem__ method."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        if len(dataset) > 0:
            signal, label = dataset[0]

            assert isinstance(signal, torch.Tensor)
            assert isinstance(label, torch.Tensor)
            assert signal.dim() >= 1
            assert label.dim() == 0 or label.dim() == 1
            if label.dim() == 0:
                assert 0 <= label.item() < 4
            elif len(label) == 1:
                assert 0 <= label[0].item() < 4

    def test_af_dataset_transforms(self, mock_dataset_files):
        """Test AF dataset with transforms."""

        def dummy_transform(x):
            return x * 2

        def dummy_target_transform(y):
            return y + 1

        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files),
            subset="training",
            transform=dummy_transform,
            target_transform=dummy_target_transform,
            download=False,
        )

        assert dataset.transform is not None
        assert dataset.target_transform is not None

    def test_af_dataset_class_distribution(self, mock_dataset_files):
        """Test getting class distribution."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        if len(dataset) > 0:
            distribution = dataset.get_class_distribution()

            assert isinstance(distribution, dict)
            for class_name in dataset.class_names:
                if class_name in distribution:
                    assert isinstance(distribution[class_name], int)
                    assert distribution[class_name] >= 0


class TestECGDataModule:
    """Test the ECGDataModule class."""

    def test_ecg_datamodule_init(self):
        """Test ECGDataModule initialization."""
        datamodule = ECGDataModule(
            dataset_class=AFClassificationDataset, data_dir="test_dir", batch_size=32, num_workers=2
        )

        assert datamodule.dataset_class == AFClassificationDataset
        assert datamodule.data_dir == "test_dir"
        assert datamodule.batch_size == 32
        assert datamodule.num_workers == 2

    def test_ecg_datamodule_setup(self, mock_dataset_files):
        """Test ECGDataModule setup method."""
        datamodule = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(mock_dataset_files),
            batch_size=2,
        )

        datamodule.setup(stage="fit")
        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "val_dataset")

        datamodule.setup(stage="test")
        assert hasattr(datamodule, "test_dataset")

    def test_ecg_datamodule_dataloaders(self, mock_dataset_files):
        """Test ECGDataModule dataloader methods."""
        datamodule = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(mock_dataset_files),
            batch_size=2,
            num_workers=0,
        )

        datamodule.setup(stage="fit")

        train_loader = datamodule.train_dataloader()
        assert train_loader is not None
        assert train_loader.batch_size == 2

        val_loader = datamodule.val_dataloader()
        assert val_loader is not None
        assert val_loader.batch_size == 2

        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()
        assert test_loader is not None

    def test_ecg_datamodule_with_different_configs(self):
        """Test ECGDataModule with different configurations."""
        configs = [
            {"batch_size": 16, "num_workers": 1},
            {"batch_size": 64, "num_workers": 0},
            {"batch_size": 8, "num_workers": 2},
        ]

        for config in configs:
            datamodule = ECGDataModule(
                dataset_class=AFClassificationDataset, data_dir="test_dir", **config
            )

            assert datamodule.batch_size == config["batch_size"]
            assert datamodule.num_workers == config["num_workers"]


class TestDatasetUtilities:
    """Test dataset utility functions."""

    def test_dataset_imports(self):
        """Test that all dataset classes can be imported."""
        assert AFClassificationDataset is not None
        assert BaseECGDataset is not None
        assert ECGDataModule is not None

    def test_dataset_inheritance(self):
        """Test dataset inheritance hierarchy."""
        assert issubclass(AFClassificationDataset, BaseECGDataset)

        required_methods = ["download", "_load_data", "__getitem__", "__len__"]
        for method in required_methods:
            assert hasattr(AFClassificationDataset, method)
            assert callable(getattr(AFClassificationDataset, method))

    def test_dataset_type_checking(self, mock_dataset_files):
        """Test dataset type checking and validation."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        metadata = dataset.get_metadata()
        assert isinstance(metadata["sampling_rate"], int)
        assert isinstance(metadata["num_classes"], int)
        assert isinstance(metadata["class_names"], list)
        assert isinstance(metadata["dataset_size"], int)

        assert isinstance(dataset.num_classes, int)
        assert isinstance(dataset.class_names, list)
        assert len(dataset.class_names) == dataset.num_classes


class TestECGStandardizer:
    def test_standardizer_init(self):
        standardizer = ECGStandardizer(
            target_sampling_rate=300,
            target_duration_seconds=10.0,
            normalization="zscore",
        )

        assert standardizer.target_sampling_rate == 300
        assert standardizer.target_length == 3000
        assert standardizer.normalization == "zscore"

    def test_resample_no_change(self):
        standardizer = ECGStandardizer(target_sampling_rate=300)
        signal = np.random.randn(1, 3000)

        resampled = standardizer.resample(signal, 300)

        assert resampled.shape == signal.shape
        np.testing.assert_array_almost_equal(resampled, signal)

    def test_resample_downsample(self):
        standardizer = ECGStandardizer(target_sampling_rate=150)
        signal = np.random.randn(1, 3000)

        resampled = standardizer.resample(signal, 300)

        assert resampled.shape[0] == 1
        assert resampled.shape[1] == 1500

    def test_resample_upsample(self):
        standardizer = ECGStandardizer(target_sampling_rate=600)
        signal = np.random.randn(1, 3000)

        resampled = standardizer.resample(signal, 300)

        assert resampled.shape[0] == 1
        assert resampled.shape[1] == 6000

    def test_normalize_zscore(self):
        standardizer = ECGStandardizer(normalization="zscore")
        signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        normalized = standardizer.normalize(signal)

        assert normalized.shape == signal.shape
        np.testing.assert_almost_equal(np.mean(normalized), 0.0, decimal=6)
        np.testing.assert_almost_equal(np.std(normalized), 1.0, decimal=6)

    def test_normalize_minmax(self):
        standardizer = ECGStandardizer(normalization="minmax")
        signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        normalized = standardizer.normalize(signal)

        assert normalized.shape == signal.shape
        np.testing.assert_almost_equal(np.min(normalized), 0.0, decimal=6)
        np.testing.assert_almost_equal(np.max(normalized), 1.0, decimal=6)

    def test_normalize_none(self):
        standardizer = ECGStandardizer(normalization="none")
        signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        normalized = standardizer.normalize(signal)

        np.testing.assert_array_equal(normalized, signal)

    def test_clip_or_pad_center_clip(self):
        standardizer = ECGStandardizer(target_length=100)
        signal = np.random.randn(1, 200)

        clipped = standardizer.clip_or_pad(signal)

        assert clipped.shape == (1, 100)

    def test_clip_or_pad_center_pad(self):
        standardizer = ECGStandardizer(target_length=200)
        signal = np.random.randn(1, 100)

        padded = standardizer.clip_or_pad(signal)

        assert padded.shape == (1, 200)

    def test_full_pipeline(self):
        standardizer = ECGStandardizer(
            target_sampling_rate=300,
            target_duration_seconds=10.0,
            normalization="zscore",
            clip_method="center",
        )

        signal = np.random.randn(2, 5000)

        processed = standardizer(signal, original_sampling_rate=250)

        assert processed.shape[0] == 2
        assert processed.shape[1] == 3000


class TestECGSegmenter:
    def test_segmenter_init(self):
        segmenter = ECGSegmenter(segment_duration_seconds=10.0, sampling_rate=300, overlap=0.5)

        assert segmenter.segment_duration_seconds == 10.0
        assert segmenter.sampling_rate == 300
        assert segmenter.overlap == 0.5
        assert segmenter.segment_length == 3000
        assert segmenter.stride == 1500

    def test_segment_no_overlap(self):
        segmenter = ECGSegmenter(segment_duration_seconds=1.0, sampling_rate=100, overlap=0.0)

        signal = np.random.randn(1, 1000)
        segments, start_indices = segmenter.segment(signal)

        assert segments.shape == (10, 1, 100)
        assert len(start_indices) == 10
        assert start_indices[0] == 0
        assert start_indices[-1] == 900

    def test_segment_with_overlap(self):
        segmenter = ECGSegmenter(segment_duration_seconds=1.0, sampling_rate=100, overlap=0.5)

        signal = np.random.randn(1, 1000)
        segments, start_indices = segmenter.segment(signal)

        assert segments.shape[0] == 19
        assert segments.shape[1:] == (1, 100)
        assert start_indices[1] - start_indices[0] == 50

    def test_segment_short_signal(self):
        segmenter = ECGSegmenter(segment_duration_seconds=10.0, sampling_rate=100, overlap=0.0)

        signal = np.random.randn(1, 500)
        segments, start_indices = segmenter.segment(signal)

        assert len(segments) == 0
        assert len(start_indices) == 0


class TestRhythmAnnotationExtractor:
    def test_extractor_init(self):
        extractor = RhythmAnnotationExtractor(sampling_rate=300, binary_classification=False)

        assert extractor.sampling_rate == 300
        assert extractor.binary_classification is False

    def test_rhythm_map(self):
        extractor = RhythmAnnotationExtractor(sampling_rate=300)

        assert extractor.RHYTHM_MAP["(AFIB"] == 1
        assert extractor.RHYTHM_MAP["(AFL"] == 2
        assert extractor.RHYTHM_MAP["(J"] == 3
        assert extractor.RHYTHM_MAP["(N"] == 0

    def test_segment_with_labels(self):
        extractor = RhythmAnnotationExtractor(sampling_rate=300)

        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        segment_start_indices = np.array([0, 3, 6, 9])
        segment_length = 3

        segment_labels = extractor.segment_with_labels(
            labels, segment_start_indices, segment_length
        )

        assert len(segment_labels) == 4
        assert segment_labels[0] == 0
        assert segment_labels[1] == 1
        assert segment_labels[2] == 2
        assert segment_labels[3] == 3


class TestMITBIHAFDBDataset:
    def test_dataset_constants(self):
        assert MITBIHAFDBDataset.CLASS_LABELS == ["Normal", "AF", "AFL", "J"]
        assert MITBIHAFDBDataset.SAMPLING_RATE == 250
        assert MITBIHAFDBDataset.LEADS == ["ECG1", "ECG2"]
        assert len(MITBIHAFDBDataset.RECORD_NAMES) == 23

    def test_dataset_init_no_data_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MITBIHAFDBDataset(
                data_dir=tmp_path,
                sampling_rate=300,
                segment_duration_seconds=10.0,
                download=False,
                verbose=False,
            )

    def test_dataset_properties(self):
        assert MITBIHAFDBDataset.CLASS_LABELS == ["Normal", "AF", "AFL", "J"]
        assert MITBIHAFDBDataset.SAMPLING_RATE == 250
        assert MITBIHAFDBDataset.LEADS == ["ECG1", "ECG2"]

    def test_dataset_binary_class_names(self, tmp_path, monkeypatch):
        def mock_load_data(self):
            self.signals = []
            self.labels = []
            self.record_names = []

        monkeypatch.setattr(MITBIHAFDBDataset, "_load_data", mock_load_data)

        dataset = MITBIHAFDBDataset(
            data_dir=tmp_path,
            binary_classification=True,
            download=False,
            verbose=False,
        )

        assert dataset.num_classes == 2
        assert dataset.class_names == ["Non-AF", "AF"]


class TestLTAFDBDataset:
    def test_dataset_constants(self):
        assert LTAFDBDataset.CLASS_LABELS == ["Normal", "AF", "AFL", "J"]
        assert LTAFDBDataset.SAMPLING_RATE == 128
        assert LTAFDBDataset.LEADS == ["ECG1", "ECG2"]

    def test_dataset_init_no_data_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LTAFDBDataset(
                data_dir=tmp_path,
                sampling_rate=300,
                segment_duration_seconds=10.0,
                download=False,
                verbose=False,
            )

    def test_dataset_properties(self):
        assert LTAFDBDataset.DOWNLOAD_URL == "https://physionet.org/content/ltafdb/get-zip/1.0.0/"
        assert LTAFDBDataset.LABEL_MAPPING == {"(N": 0, "(AFIB": 1, "(AFL": 2, "(J": 3}


class TestUnifiedAFDataset:
    def test_dataset_constants(self):
        assert UnifiedAFDataset.CLASS_LABELS == ["Normal", "AF", "AFL", "J"]
        assert "physionet2017" in UnifiedAFDataset.AVAILABLE_DATASETS
        assert "mitbih_afdb" in UnifiedAFDataset.AVAILABLE_DATASETS
        assert "ltafdb" in UnifiedAFDataset.AVAILABLE_DATASETS

    def test_dataset_init(self, tmp_path):
        with pytest.raises(RuntimeError, match="No datasets were successfully loaded"):
            UnifiedAFDataset(
                data_dir=tmp_path,
                datasets=["physionet2017"],
                download=False,
                verbose=False,
            )

    def test_dataset_invalid_name(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            UnifiedAFDataset(
                data_dir=tmp_path,
                datasets=["invalid_dataset"],
                download=False,
                verbose=False,
            )

    def test_dataset_properties(self, tmp_path, monkeypatch):
        def mock_load_data(self):
            self.datasets = []
            self.dataset_sizes = []
            self.concat_dataset = None

        monkeypatch.setattr(UnifiedAFDataset, "_load_data", mock_load_data)

        dataset = UnifiedAFDataset(
            data_dir=tmp_path,
            datasets=["physionet2017"],
            download=False,
            verbose=False,
        )

        assert dataset.num_classes == 4
        assert dataset.class_names == ["Normal", "AF", "AFL", "J"]

    def test_dataset_binary_classification(self, tmp_path, monkeypatch):
        def mock_load_data(self):
            self.datasets = []
            self.dataset_sizes = []
            self.concat_dataset = None

        monkeypatch.setattr(UnifiedAFDataset, "_load_data", mock_load_data)

        dataset = UnifiedAFDataset(
            data_dir=tmp_path,
            datasets=["physionet2017"],
            binary_classification=True,
            download=False,
            verbose=False,
        )

        assert dataset.num_classes == 2
        assert dataset.class_names == ["Non-AF", "AF"]


class TestDatasetIntegration:
    def test_dataset_inheritance(self):
        assert issubclass(MITBIHAFDBDataset, BaseECGDatasetDirect)
        assert issubclass(LTAFDBDataset, BaseECGDatasetDirect)
        assert issubclass(MITBIHAFDBDataset, Dataset)
        assert issubclass(LTAFDBDataset, Dataset)

    def test_all_datasets_have_required_methods(self):
        datasets = [AFClassificationDataset, MITBIHAFDBDataset, LTAFDBDataset, UnifiedAFDataset]
        required_methods = ["download", "_load_data", "__getitem__", "__len__"]

        for dataset_class in datasets:
            for method in required_methods:
                assert hasattr(dataset_class, method)
                assert callable(getattr(dataset_class, method))

    def test_all_datasets_have_properties(self):
        datasets = [MITBIHAFDBDataset, LTAFDBDataset, UnifiedAFDataset]

        for dataset_class in datasets:
            assert hasattr(dataset_class, "num_classes")
            assert hasattr(dataset_class, "class_names")
