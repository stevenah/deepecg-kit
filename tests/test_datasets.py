"""
Tests for deepecgkit.datasets module.
"""

import zipfile
from pathlib import Path
from typing import ClassVar
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
from deepecgkit.datasets.io import (
    SUPPORTED_ECG_SUFFIXES,
    as_model_input,
    load_recordings,
    prepare_recording,
    read_ecg_recording,
    read_ecg_recording_with_rate,
)


class TestBaseECGDataset:
    """Test the BaseECGDataset abstract class."""

    def test_get_default_data_dir(self, monkeypatch):
        """Test the default data directory path."""
        monkeypatch.delenv("DEEPECGKIT_HOME", raising=False)
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
        """Download fetches the label file, skipping archives that are already extracted."""
        dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        dataset.download()

        fetched = [Path(call.args[1]).name for call in mock_download.call_args_list]
        assert fetched == ["REFERENCE-v3.csv"]
        mirrors = mock_download.call_args_list[0].args[0]
        assert all(url.endswith("REFERENCE-v3.csv") for url in mirrors)
        assert len(mirrors) == len(AFClassificationDataset.DOWNLOAD_MIRRORS)

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

    @staticmethod
    def _annotation(entries):
        """A stand-in for a wfdb annotation: (sample, aux_note) pairs in order."""
        from types import SimpleNamespace

        return SimpleNamespace(
            sample=np.array([sample for sample, _ in entries]),
            aux_note=[note for _, note in entries],
        )

    def test_rhythm_only_annotations_label_each_span(self):
        """MIT-BIH AFDB style: every annotation is a rhythm change."""
        extractor = RhythmAnnotationExtractor(sampling_rate=250)
        annotation = self._annotation([(0, "(N"), (100, "(AFIB"), (300, "(AFL"), (400, "(N")])

        labels = extractor.extract_labels(annotation, 500, 250)

        assert labels[:100].tolist() == [0] * 100
        assert labels[100:300].tolist() == [1] * 200
        assert labels[300:400].tolist() == [2] * 100
        assert labels[400:].tolist() == [0] * 100

    def test_beat_annotations_do_not_end_an_episode(self):
        """LTAFDB style: a beat annotation sits between every pair of rhythm changes.

        Each beat used to reset the rhythm to Normal, which left an AF episode
        one beat long and the dataset labelled almost entirely Normal.
        """
        extractor = RhythmAnnotationExtractor(sampling_rate=128, binary_classification=True)
        beats = [(sample, "") for sample in range(50, 1000, 50)]
        rhythm = [(100, "(AFIB"), (600, "(N")]
        annotation = self._annotation(sorted(beats + rhythm))

        labels = extractor.extract_labels(annotation, 1000, 128)

        assert labels[:100].sum() == 0
        assert labels[100:600].tolist() == [1] * 500
        assert labels[600:].sum() == 0

    def test_comments_and_padded_notes_are_handled(self):
        """Pause and missed-beat notes are not rhythms; NUL padding is not part of one."""
        extractor = RhythmAnnotationExtractor(sampling_rate=128, binary_classification=True)
        annotation = self._annotation(
            [(0, "\x01 Aux"), (10, "(AFIB\x00"), (40, "PSE"), (60, "MISSB"), (80, "(SVTA")]
        )

        labels = extractor.extract_labels(annotation, 100, 128)

        assert labels[10:80].tolist() == [1] * 70
        assert labels[80:].sum() == 0

    def test_annotation_positions_follow_the_resampled_signal(self):
        extractor = RhythmAnnotationExtractor(sampling_rate=300)
        annotation = self._annotation([(0, "(N"), (128, "(AFIB"), (256, "(N")])

        labels = extractor.extract_labels(annotation, 900, original_sampling_rate=128)

        assert labels[:300].sum() == 0
        assert labels[300:600].tolist() == [1] * 300
        assert labels[600:].sum() == 0

    def test_binary_mode_keeps_only_atrial_fibrillation(self):
        extractor = RhythmAnnotationExtractor(sampling_rate=250, binary_classification=True)
        annotation = self._annotation([(0, "(AFL"), (10, "(AFIB"), (20, "(J")])

        assert (
            extractor.extract_labels(annotation, 30, 250).tolist() == [0] * 10 + [1] * 10 + [0] * 10
        )

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
        assert RhythmAnnotationExtractor.RHYTHM_MAP == {"(N": 0, "(AFIB": 1, "(AFL": 2, "(J": 3}


class TestUnifiedAFDataset:
    def test_dataset_constants(self):
        assert UnifiedAFDataset.CLASS_LABELS == ["Normal", "AF", "AFL", "J"]
        assert "physionet2017" in UnifiedAFDataset.AVAILABLE_DATASETS
        assert "mitbih_afdb" in UnifiedAFDataset.AVAILABLE_DATASETS
        assert "ltafdb" in UnifiedAFDataset.AVAILABLE_DATASETS

    def test_dataset_init(self, tmp_path):
        with pytest.raises(RuntimeError, match="Could not load 'physionet2017'"):
            UnifiedAFDataset(
                data_dir=tmp_path,
                datasets=["physionet2017"],
                download=False,
                verbose=False,
            )

    def test_a_missing_source_is_an_error_rather_than_a_smaller_dataset(self, temp_dir):
        """Two sources out of three used to be trained on under the name of all three."""
        from tests.conftest import write_af_dataset

        write_af_dataset(temp_dir / AFClassificationDataset.get_default_data_dir().name)

        with pytest.raises(RuntimeError, match="Could not load 'mitbih_afdb'"):
            UnifiedAFDataset(
                data_dir=temp_dir,
                datasets=["physionet2017", "mitbih_afdb"],
                download=False,
                verbose=False,
            )

    def test_unavailable_sources_can_be_skipped_on_request(self, temp_dir):
        from tests.conftest import write_af_dataset

        write_af_dataset(temp_dir / AFClassificationDataset.get_default_data_dir().name)

        dataset = UnifiedAFDataset(
            data_dir=temp_dir,
            datasets=["physionet2017", "mitbih_afdb"],
            download=False,
            verbose=False,
            skip_unavailable=True,
        )

        assert dataset.loaded_dataset_names == ["physionet2017"]

    def test_nothing_to_load_is_an_error_even_when_skipping(self, tmp_path):
        with pytest.raises(RuntimeError, match="No datasets were successfully loaded"):
            UnifiedAFDataset(
                data_dir=tmp_path,
                datasets=["physionet2017"],
                download=False,
                verbose=False,
                skip_unavailable=True,
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

    def test_sources_are_read_from_an_explicit_data_dir(self, temp_dir):
        """data_dir used to be accepted and ignored, every source coming from its default."""
        from tests.conftest import write_af_dataset

        write_af_dataset(temp_dir / AFClassificationDataset.get_default_data_dir().name)

        dataset = UnifiedAFDataset(
            data_dir=temp_dir, datasets=["physionet2017"], download=False, verbose=False
        )

        assert len(dataset) > 0
        assert dataset.loaded_dataset_names == ["physionet2017"]

    def test_is_downloaded_looks_under_an_explicit_data_dir(self, temp_dir):
        assert not UnifiedAFDataset.is_downloaded(temp_dir)

    def test_forced_download_clears_the_sources_and_nothing_else(self, temp_dir, monkeypatch):
        """force_download used to delete data_dir whole and then fetch nothing."""
        from tests.conftest import write_af_dataset

        mine = temp_dir / "notes.txt"
        mine.write_text("mine")
        source = write_af_dataset(temp_dir / AFClassificationDataset.get_default_data_dir().name)
        fetched = []

        def record_fetch(cls, data_dir=None, verbose=True):
            fetched.append(Path(data_dir))

        monkeypatch.setattr(AFClassificationDataset, "download_to", classmethod(record_fetch))

        with pytest.raises(RuntimeError, match="Could not load 'physionet2017'"):
            UnifiedAFDataset(
                data_dir=temp_dir, datasets=["physionet2017"], force_download=True, verbose=False
            )

        assert mine.read_text() == "mine"
        assert not source.exists()
        assert fetched == [source]


class TestForcedDownloadGuard:
    """A forced download deletes the directory first, so it has to be the dataset's own."""

    def test_a_directory_holding_something_else_is_left_alone(self, temp_dir):
        (temp_dir / "thesis.txt").write_text("irreplaceable")

        with pytest.raises(RuntimeError, match="Refusing to clear"):
            AFClassificationDataset(data_dir=temp_dir, force_download=True, verbose=False)

        assert (temp_dir / "thesis.txt").read_text() == "irreplaceable"

    @pytest.mark.parametrize("state", ["empty", "complete", "interrupted"])
    def test_a_recognised_directory_may_be_cleared(self, state, temp_dir):
        from deepecgkit.utils.download import IN_PROGRESS_MARKER
        from tests.conftest import write_af_dataset

        target = temp_dir / "data"
        target.mkdir()
        if state == "complete":
            write_af_dataset(target)
        elif state == "interrupted":
            (target / IN_PROGRESS_MARKER).touch()
            (target / "training2017.zip.part").write_bytes(b"partial")

        AFClassificationDataset._require_clearable(target)

    def test_the_default_directory_is_the_packages_own(self, temp_dir, monkeypatch):
        """Whatever an older version left there, the default directory is safe to replace."""
        monkeypatch.setenv("DEEPECGKIT_HOME", str(temp_dir))
        default = AFClassificationDataset.get_default_data_dir()
        default.mkdir(parents=True)
        (default / "challenge-2017-1.0.0.zip").write_bytes(b"PK\x03\x04truncated")

        AFClassificationDataset._require_clearable(default)


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


class TestECGFileIO:
    """Reading ECG recordings from files and reshaping them for models."""

    def test_read_npy(self, temp_dir):
        """A .npy file round-trips."""
        path = temp_dir / "signal.npy"
        data = np.random.randn(2, 3000)
        np.save(str(path), data)
        assert np.allclose(read_ecg_recording(path), data)

    def test_read_csv(self, temp_dir):
        """A .csv file is read through its header row."""
        import pandas as pd

        path = temp_dir / "signal.csv"
        pd.DataFrame({"I": np.arange(10.0), "II": np.arange(10.0)}).to_csv(str(path), index=False)
        assert read_ecg_recording(path).shape == (10, 2)

    def test_read_unsupported_suffix(self, temp_dir):
        """An unknown suffix names the supported formats."""
        path = temp_dir / "signal.txt"
        path.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported input format"):
            read_ecg_recording(path)

    def test_supported_suffixes(self):
        """The suffix tuple is what the CLI advertises."""
        assert SUPPORTED_ECG_SUFFIXES == (".npy", ".csv", ".dat", ".hea")

    def test_one_dimensional_becomes_single_sample(self):
        """A bare signal becomes one single-lead recording."""
        assert as_model_input(np.zeros(3000)).shape == (1, 1, 3000)

    def test_two_dimensional_leads_first(self):
        """A leads-first array keeps its orientation."""
        assert as_model_input(np.zeros((12, 3000)), input_channels=12).shape == (1, 12, 3000)

    def test_two_dimensional_time_first(self):
        """A time-first array is transposed."""
        assert as_model_input(np.zeros((3000, 12)), input_channels=12).shape == (1, 12, 3000)

    def test_two_dimensional_without_lead_count(self):
        """Without a lead count, rows are independent single-lead recordings."""
        assert as_model_input(np.zeros((5, 3000))).shape == (5, 1, 3000)

    def test_two_dimensional_incompatible(self):
        """A mismatch carries the message the predict command reports."""
        with pytest.raises(ValueError, match="is not compatible with a model expecting 12 leads"):
            as_model_input(np.zeros((5, 3000)), input_channels=12)

    def test_three_dimensional_passes_through(self):
        """An already-batched array is returned unchanged."""
        data = np.zeros((4, 2, 3000))
        assert as_model_input(data).shape == (4, 2, 3000)

    def test_four_dimensional_rejected(self):
        """Too many dimensions are reported rather than passed to the model."""
        with pytest.raises(ValueError, match="expected at most three"):
            as_model_input(np.zeros((2, 2, 2, 2)))


class TestBaseDatasetDownloadBehaviour:
    """Regression tests for the download plumbing on BaseECGDataset."""

    @staticmethod
    def _minimal_dataset_class():
        from deepecgkit.datasets.base import BaseECGDataset

        class MiniDataset(BaseECGDataset):
            CLASS_LABELS: ClassVar[list[str]] = ["A", "B"]
            downloads: ClassVar[list[Path]] = []

            @classmethod
            def _download(cls, data_dir, verbose=True):
                cls.downloads.append(Path(data_dir))
                Path(data_dir).mkdir(parents=True, exist_ok=True)
                (Path(data_dir) / "marker.txt").write_text("ok")

            def _load_data(self):
                self.signals = [np.zeros((1, 10), dtype=np.float32)]
                self.labels = [0]
                self.record_names = ["r0"]

        return MiniDataset

    def test_download_triggers_for_existing_empty_dir(self, temp_dir):
        """download=True used to be ignored when the directory existed but was
        empty, failing later in _load_data."""
        cls = self._minimal_dataset_class()
        empty = temp_dir / "empty"
        empty.mkdir()

        cls(data_dir=str(empty), download=True)
        assert cls.downloads == [empty]

    def test_download_skipped_for_populated_dir(self, temp_dir):
        cls = self._minimal_dataset_class()
        populated = temp_dir / "populated"
        populated.mkdir()
        (populated / "existing.dat").write_text("x")

        cls(data_dir=str(populated), download=True)
        assert cls.downloads == []

    def test_download_to_fetches_without_loading(self, temp_dir):
        """download_to replaces the object.__new__ hack unified-af used to
        reach a sub-dataset's download without running _load_data."""
        cls = self._minimal_dataset_class()
        target = temp_dir / "dl"

        returned = cls.download_to(target, verbose=False)
        assert returned == target
        assert (target / "marker.txt").exists()

    def test_base_accessors_from_standard_storage(self, temp_dir):
        cls = self._minimal_dataset_class()
        populated = temp_dir / "data"
        populated.mkdir()
        (populated / "existing.dat").write_text("x")

        ds = cls(data_dir=str(populated))
        assert len(ds) == 1
        signal, label = ds[0]
        assert signal.shape == (1, 10)
        assert int(label) == 0
        assert ds.num_classes == 2
        assert ds.class_names == ["A", "B"]
        assert ds.get_class_distribution() == {"A": 1}
        assert ds.get_record_info(0)["record_name"] == "r0"


class TestLTAFDBCacheAndMemmap:
    """Regression tests for LTAFDB subsampling reproducibility and cache growth."""

    @staticmethod
    def _bare_instance(**attrs):
        ds = object.__new__(LTAFDBDataset)
        defaults = {
            "sampling_rate": 300,
            "segment_duration_seconds": 10.0,
            "segment_overlap": 0.0,
            "binary_classification": False,
            "use_both_leads": False,
            "max_segments_per_record": None,
            "subsample_seed": 0,
        }
        defaults.update(attrs)
        for key, value in defaults.items():
            setattr(ds, key, value)

        class _Std:
            normalization = "zscore"

        ds.standardizer = _Std()
        return ds

    def test_cache_key_ignores_seed_without_subsampling(self):
        """Full caches do not depend on the subsampling seed."""
        a = self._bare_instance(subsample_seed=0)
        b = self._bare_instance(subsample_seed=123)
        assert a._cache_key() == b._cache_key()

    def test_cache_key_includes_seed_when_subsampling(self):
        """A subsampled cache is only reusable for the same seed."""
        a = self._bare_instance(max_segments_per_record=10, subsample_seed=0)
        b = self._bare_instance(max_segments_per_record=10, subsample_seed=123)
        assert a._cache_key() != b._cache_key()

    def test_caches_from_before_the_labelling_fix_are_pointed_out(self, temp_dir, caplog):
        """A folder of mislabelled segments is never read, but it is several gigabytes."""
        import logging

        instance = self._bare_instance()
        instance.data_dir = temp_dir
        old = temp_dir / "cache_0123456789ab"
        old.mkdir()
        (old / "signals.npy").write_bytes(b"x" * 1024)
        current = temp_dir / "cache_current"
        current.mkdir()
        (current / "signals.npy").write_bytes(b"x")
        tagged = temp_dir / "cache_other_parameters"
        tagged.mkdir()
        (tagged / "signals.npy").write_bytes(b"x")
        (tagged / LTAFDBDataset.CACHE_INFO_FILE).write_text("{}")

        with caplog.at_level(logging.WARNING, logger="deepecgkit.datasets.ltafdb"):
            instance._warn_about_stale_caches(current)

        assert "cache_0123456789ab" in caplog.text
        assert "cache_other_parameters" not in caplog.text
        assert "cache_current" not in caplog.text

    def test_cache_key_changes_with_the_cache_version(self, monkeypatch):
        """Caches written before a labelling fix must not be read back."""
        instance = self._bare_instance()
        current = instance._cache_key()
        monkeypatch.setattr(LTAFDBDataset, "CACHE_VERSION", LTAFDBDataset.CACHE_VERSION + 1)
        assert instance._cache_key() != current

    def test_grow_memmap_preserves_content_and_grows(self, temp_dir):
        """open_memmap(mode='r+') ignores a passed shape, so the old 'grow'
        silently kept the original capacity; the helper must actually grow."""
        path = temp_dir / "signals_tmp.npy"
        writer = np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=(2, 1, 5))
        writer[0] = 1.0
        writer[1] = 2.0

        grown = LTAFDBDataset._grow_memmap(path, writer, 6)
        assert grown.shape == (6, 1, 5)
        assert np.all(grown[0] == 1.0) and np.all(grown[1] == 2.0)

        grown[5] = 5.0  # writing past the old capacity must succeed
        grown.flush()
        reread = np.load(path, mmap_mode="r")
        assert reread.shape == (6, 1, 5)
        assert np.all(reread[5] == 5.0)


class TestRecordLoadErrorPolicy:
    """Skipped source records must be loud, bounded, and escapable."""

    @staticmethod
    def _bare_base():
        class _Concrete(BaseECGDatasetDirect):
            def _load_data(self):
                pass

        ds = object.__new__(_Concrete)
        ds.verbose = False
        return ds

    def test_strict_reraises(self):
        ds = self._bare_base()
        ds.strict = True
        with pytest.raises(ValueError, match="corrupt"):
            ds._record_load_error("rec01", ValueError("corrupt"))

    def test_partial_failure_warns_unconditionally(self):
        ds = self._bare_base()
        ds._record_load_error("rec01", ValueError("corrupt"))
        with pytest.warns(RuntimeWarning, match=r"Skipped 1/10 records"):
            ds._finish_load(loaded_records=9, total_records=10)

    def test_total_failure_raises(self):
        ds = self._bare_base()
        ds._record_load_error("rec01", ValueError("corrupt"))
        with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="All 1 records"):
            ds._finish_load(loaded_records=0, total_records=1)

    def test_clean_load_is_silent(self, recwarn):
        ds = self._bare_base()
        ds._finish_load(loaded_records=5, total_records=5)
        assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


def _zip_folder(folder: Path, archive: Path, prefix: str) -> None:
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, f"{prefix}/{path.relative_to(folder)}")


class TestSetupSplitsOnce:
    """The trainer calls setup() for every stage; only the first call should do the work."""

    def _module(self, data_dir, **kwargs):
        return ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(data_dir),
            verbose=False,
            dataset_kwargs={"verbose": False},
            **kwargs,
        )

    def test_a_repeated_setup_keeps_the_split_it_made(self, af_dataset_dir):
        module = self._module(af_dataset_dir)
        module.setup(stage="fit")
        first = (module.train_dataset, module.val_dataset, module.test_dataset)

        module.setup(stage="test")

        assert (module.train_dataset, module.val_dataset, module.test_dataset) == first

    def test_a_changed_setting_splits_again(self, af_dataset_dir):
        module = self._module(af_dataset_dir, seed=1)
        module.setup()
        before = list(module.test_dataset.indices)

        module.seed = 2
        module.setup()

        assert list(module.test_dataset.indices) != before


class _SignalsMustNotBeRead:
    """Stands in for a dataset's signals where reading one would be a mistake."""

    def __getitem__(self, index):
        raise AssertionError("a signal was read to get at a label")


class TestSplitLabels:
    """The split is stratified on labels taken from the dataset, not read sample by sample."""

    def _dataset(self, data_dir, **kwargs):
        return AFClassificationDataset(data_dir=str(data_dir), verbose=False, **kwargs)

    def test_direct_labels_are_what_the_samples_carry(self, af_dataset_dir):
        dataset = self._dataset(af_dataset_dir)
        from_samples = torch.stack([dataset[i][1] for i in range(len(dataset))]).numpy()

        direct = dataset.stratification_labels()

        assert direct.dtype == from_samples.dtype
        np.testing.assert_array_equal(direct, from_samples)

    def test_the_split_does_not_read_a_signal_for_them(self, af_dataset_dir):
        """On a memory-mapped dataset of gigabytes that paged all of it in."""
        dataset = self._dataset(af_dataset_dir)
        expected = dataset.stratification_labels()
        dataset.signals = _SignalsMustNotBeRead()
        module = ECGDataModule(dataset=dataset, verbose=False)

        np.testing.assert_array_equal(module._collect_labels(), expected)

    def test_the_split_is_the_one_reading_every_sample_gave(self, af_dataset_dir, monkeypatch):
        def split_indices():
            module = ECGDataModule(dataset=self._dataset(af_dataset_dir), verbose=False, seed=3)
            module.setup()
            return [
                list(part.indices)
                for part in (module.train_dataset, module.val_dataset, module.test_dataset)
            ]

        direct = split_indices()
        monkeypatch.setattr(AFClassificationDataset, "stratification_labels", lambda self: None)

        assert split_indices() == direct

    def test_a_target_transform_sends_it_back_to_the_samples(self, af_dataset_dir):
        """The transform decides the label, so only the samples know it."""
        dataset = self._dataset(af_dataset_dir, target_transform=lambda label: label * 0)

        assert dataset.stratification_labels() is None
        module = ECGDataModule(dataset=dataset, verbose=False)
        assert set(module._collect_labels().tolist()) == {0}

    def test_the_unified_dataset_joins_its_sources_labels(self, temp_dir):
        from tests.conftest import write_af_dataset

        write_af_dataset(temp_dir / AFClassificationDataset.get_default_data_dir().name)
        dataset = UnifiedAFDataset(
            data_dir=temp_dir, datasets=["physionet2017"], download=False, verbose=False
        )
        from_samples = torch.stack([dataset[i][1] for i in range(len(dataset))]).numpy()

        np.testing.assert_array_equal(dataset.stratification_labels(), from_samples)


class TestSamplingRateResolution:
    """A dataset is resampled only when someone asks for it."""

    def _module(self, data_dir, **kwargs):
        return ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(data_dir),
            verbose=False,
            dataset_kwargs={"verbose": False},
            **kwargs,
        )

    def test_data_module_keeps_the_dataset_default_rate(self, af_dataset_dir):
        module = self._module(af_dataset_dir)
        assert module.sampling_rate is None

        module.setup()

        assert module.dataset.sampling_rate == AFClassificationDataset.NATIVE_SAMPLING_RATE
        assert module.sampling_rate == 300
        signal, _ = module.dataset[0]
        assert signal.shape == (1, 3000)

    def test_an_explicit_rate_reaches_the_dataset(self, af_dataset_dir):
        module = self._module(af_dataset_dir, sampling_rate=150)
        module.setup()

        assert module.dataset.sampling_rate == 150
        signal, _ = module.dataset[0]
        assert signal.shape == (1, 1500)

    def test_data_module_defaults_to_no_loader_workers(self):
        assert ECGDataModule(dataset_class=AFClassificationDataset).num_workers == 0

    def test_describe_reports_what_a_checkpoint_needs(self, af_dataset_dir):
        module = self._module(af_dataset_dir, seed=7)
        module.setup()

        description = module.describe()

        assert description["dataset"] == "af-classification"
        assert description["preprocessing"] == {
            "sampling_rate": 300,
            "segment_duration_seconds": 10.0,
            "normalization": "zscore",
            "leads": ["ECG"],
        }
        assert description["class_names"] == ["Normal", "AF", "Other", "Noisy"]
        assert description["input_channels"] == 1
        assert description["input_length"] == 3000
        assert description["split"] == {"seed": 7, "val_split": 0.2, "test_split": 0.1}

    def test_describe_requires_setup(self):
        with pytest.raises(RuntimeError, match="setup"):
            ECGDataModule(dataset_class=AFClassificationDataset).describe()


class TestDatasetAvailability:
    """An interrupted download is noticed and repaired instead of loaded."""

    def test_complete_layout_counts_as_downloaded(self, mock_dataset_files):
        assert AFClassificationDataset.is_downloaded(mock_dataset_files)

    def test_missing_directory_is_not_downloaded(self, temp_dir):
        assert not AFClassificationDataset.is_downloaded(temp_dir / "absent")

    def test_truncated_archive_is_not_a_dataset(self, temp_dir):
        (temp_dir / "challenge-2017-1.0.0.zip").write_bytes(b"PK\x03\x04" + b"\0" * 2048)
        assert not AFClassificationDataset.is_downloaded(temp_dir)

    def test_truncated_inner_archive_is_not_a_dataset(self, mock_dataset_files):
        (mock_dataset_files / "training2017.zip").write_bytes(b"PK\x03\x04truncated")
        assert not AFClassificationDataset.is_downloaded(mock_dataset_files)

    def test_missing_dataset_error_names_the_directory_and_the_fix(self, temp_dir):
        target = temp_dir / "absent"
        with pytest.raises(FileNotFoundError) as excinfo:
            AFClassificationDataset(data_dir=target, download=False, verbose=False)

        message = str(excinfo.value)
        assert str(target) in message
        assert "download=True" in message
        assert "DEEPECGKIT_HOME" in message

    def test_truncated_inner_archive_reports_itself(self, mock_dataset_files):
        (mock_dataset_files / "training2017.zip").write_bytes(b"PK\x03\x04truncated")
        with pytest.raises(FileNotFoundError, match="truncated or corrupted"):
            AFClassificationDataset(data_dir=mock_dataset_files, download=False, verbose=False)

    def test_download_true_recovers_from_an_interrupted_download(
        self, temp_dir, af_dataset_dir, monkeypatch
    ):
        """The leftovers of a cut-short download must not block the next attempt."""
        from deepecgkit.datasets import af_classification as af_module

        target = temp_dir / "cache"
        target.mkdir()
        leftover = target / AFClassificationDataset.LEGACY_ARCHIVE
        leftover.write_bytes(b"PK\x03\x04" + b"\0" * 4096)
        fetched = []

        def fake_download(urls, file_path, desc=None, force=False, max_retries=3):
            file_path = Path(file_path)
            fetched.append(file_path.name)
            if file_path.name == "REFERENCE-v3.csv":
                file_path.write_text((af_dataset_dir / "REFERENCE-v3.csv").read_text())
            elif file_path.name == "training2017.zip":
                _zip_folder(af_dataset_dir / "training2017", file_path, "training2017")
            else:
                with zipfile.ZipFile(file_path, "w") as zf:
                    zf.writestr("sample2017/validation/REFERENCE.csv", "")

        monkeypatch.setattr(af_module, "download_file", fake_download)

        dataset = AFClassificationDataset(data_dir=target, download=True, verbose=False)

        assert fetched == list(AFClassificationDataset.DOWNLOAD_FILES)
        assert not leftover.exists()
        assert len(dataset) == 120
        assert not (target / "training2017.zip").exists()
        assert AFClassificationDataset.is_downloaded(target)

    def test_download_is_skipped_when_the_dataset_is_present(self, af_dataset_dir, monkeypatch):
        from deepecgkit.datasets import af_classification as af_module

        def unexpected(*args, **kwargs):
            raise AssertionError("nothing should be downloaded")

        monkeypatch.setattr(af_module, "download_file", unexpected)
        dataset = AFClassificationDataset(data_dir=af_dataset_dir, download=True, verbose=False)
        assert len(dataset) == 120

    def test_leftover_archive_rebuilds_a_half_extracted_folder(self, temp_dir, af_dataset_dir):
        target = temp_dir / "cache"
        (target / "training2017").mkdir(parents=True)
        (target / "REFERENCE-v3.csv").write_text((af_dataset_dir / "REFERENCE-v3.csv").read_text())
        _zip_folder(af_dataset_dir / "training2017", target / "training2017.zip", "training2017")

        dataset = AFClassificationDataset(data_dir=target, download=False, verbose=False)

        assert len(dataset) == 120
        assert not (target / "training2017.zip").exists()

    def test_validation_subset_reads_the_nested_sample_folder(self, temp_dir, af_dataset_dir):
        target = temp_dir / "cache"
        validation = target / "sample2017" / "validation"
        validation.mkdir(parents=True)
        for path in sorted((af_dataset_dir / "training2017").glob("A0000*.mat")):
            (validation / path.name).write_bytes(path.read_bytes())
        (validation / "REFERENCE.csv").write_text(
            "\n".join(f"A0000{i},N" for i in range(10)) + "\n"
        )

        dataset = AFClassificationDataset(
            data_dir=target, subset="validation", download=False, verbose=False
        )

        assert len(dataset) == 30

    def test_signals_are_stored_in_single_precision(self, af_dataset_dir):
        dataset = AFClassificationDataset(data_dir=af_dataset_dir, verbose=False)
        assert dataset.signals[0].dtype == np.float32

    def test_record_datasets_ignore_unfinished_downloads(self, temp_dir):
        for dataset_class in (MITBIHAFDBDataset, LTAFDBDataset):
            folder = temp_dir / dataset_class.__name__
            folder.mkdir()
            assert not dataset_class.is_downloaded(folder)

            (folder / "records.zip.part").write_bytes(b"x")
            assert not dataset_class.is_downloaded(folder)

            (folder / "04015.hea").write_text("header")
            (folder / "04015.atr").write_bytes(b"annotation")
            assert not dataset_class.is_downloaded(folder)

            (folder / "records.zip.part").unlink()
            assert dataset_class.is_downloaded(folder)

    def test_ptbxl_needs_its_metadata_table_and_no_leftover_archive(self, temp_dir):
        from deepecgkit.datasets import PTBXLDataset

        assert not PTBXLDataset.is_downloaded(temp_dir)
        (temp_dir / "ptbxl_database.csv").write_text("ecg_id\n")
        assert PTBXLDataset.is_downloaded(temp_dir)
        (temp_dir / "ptb-xl-1.0.3.zip").write_bytes(b"x")
        assert not PTBXLDataset.is_downloaded(temp_dir)

    def test_unimplemented_download_explains_itself(self, temp_dir):
        class LocalOnly(BaseECGDataset):
            def _load_data(self):
                self.signals, self.labels, self.record_names = [], [], []

        with pytest.raises(NotImplementedError, match="does not implement _download"):
            LocalOnly(data_dir=temp_dir / "empty", download=True)

    def test_unified_dataset_forwards_its_normalization(self):
        dataset = UnifiedAFDataset.__new__(UnifiedAFDataset)
        dataset.sampling_rate = 300
        dataset.transform = None
        dataset.target_transform = None
        dataset.segment_duration_seconds = 10.0
        dataset.binary_classification = False
        dataset.normalization = "minmax"
        dataset.dataset_kwargs = {}

        for name in UnifiedAFDataset.AVAILABLE_DATASETS:
            assert dataset._get_dataset_kwargs(name)["normalization"] == "minmax"


class TestRecordingPreparation:
    """A recording file goes through the same steps as the training data."""

    PREPROCESSING: ClassVar[dict] = {
        "sampling_rate": 300,
        "segment_duration_seconds": 10.0,
        "normalization": "zscore",
    }

    def test_csv_keeps_every_sample_with_or_without_a_header(self, temp_dir):
        samples = np.arange(1.0, 6.0)
        with_header = temp_dir / "with_header.csv"
        with_header.write_text("ecg\n" + "\n".join(str(value) for value in samples) + "\n")
        without_header = temp_dir / "without_header.csv"
        without_header.write_text("\n".join(str(value) for value in samples) + "\n")

        assert read_ecg_recording(with_header).ravel().tolist() == samples.tolist()
        assert read_ecg_recording(without_header).ravel().tolist() == samples.tolist()

    def test_wfdb_records_report_their_sampling_rate(self, temp_dir):
        import wfdb

        wfdb.wrsamp(
            "rec",
            fs=250,
            units=["mV"],
            sig_name=["ECG"],
            p_signal=np.random.randn(1000, 1),
            write_dir=str(temp_dir),
        )

        signal, rate = read_ecg_recording_with_rate(temp_dir / "rec.hea")

        assert rate == 250.0
        assert signal.shape == (1, 1000)

    def test_array_files_carry_no_sampling_rate(self, temp_dir):
        path = temp_dir / "ecg.npy"
        np.save(str(path), np.zeros(100))
        assert read_ecg_recording_with_rate(path)[1] is None

    def test_matches_what_the_dataset_loader_produces(self, af_dataset_dir):
        """The property that makes prediction on a raw file meaningful."""
        import scipy.io

        dataset = AFClassificationDataset(data_dir=af_dataset_dir, verbose=False)
        raw = scipy.io.loadmat(str(af_dataset_dir / "training2017" / "A00000.mat"))["val"]

        prepared = prepare_recording(raw, dataset.get_preprocessing(), source_sampling_rate=300)

        assert prepared.segments.shape == (3, 1, 3000)
        np.testing.assert_allclose(prepared.segments, np.stack(dataset.signals[:3]), atol=1e-6)
        assert prepared.start_seconds.tolist() == [0.0, 10.0, 20.0]
        assert prepared.sampling_rate == 300.0

    def test_amplitude_scale_does_not_matter_after_normalisation(self):
        raw = np.random.default_rng(0).standard_normal(9000)
        small = prepare_recording(raw, self.PREPROCESSING, 300)
        large = prepare_recording(raw * 1000.0, self.PREPROCESSING, 300)
        np.testing.assert_allclose(small.segments, large.segments, atol=1e-4)

    def test_resamples_to_the_training_rate(self):
        prepared = prepare_recording(np.random.randn(4500), self.PREPROCESSING, 150)
        assert prepared.segments.shape == (3, 1, 3000)
        assert any("resampled from 150 Hz to 300 Hz" in note for note in prepared.notes)

    def test_unknown_rate_is_assumed_and_reported(self):
        prepared = prepare_recording(np.random.randn(3000), self.PREPROCESSING, None)
        assert prepared.segments.shape == (1, 1, 3000)
        assert any("assumed to be sampled at 300 Hz" in note for note in prepared.notes)

    def test_a_short_recording_is_used_whole_and_says_so(self):
        prepared = prepare_recording(np.random.randn(900), self.PREPROCESSING, 300)
        assert prepared.segments.shape == (1, 1, 900)
        assert any("shorter than the 10 s segments" in note for note in prepared.notes)

    def test_without_preprocessing_the_recording_passes_through(self):
        raw = np.random.randn(2, 500)
        prepared = prepare_recording(raw, None, 250.0)
        assert prepared.notes == []
        np.testing.assert_allclose(prepared.segments[0], raw, atol=1e-6)


class TestMirroredDownloads:
    """Record datasets come from the mirror file by file, with the archive as fallback."""

    def test_afdb_requests_exactly_the_files_its_loader_reads(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import base as base_module

        seen = {}

        def fake_download_files(mirrors, names, dest_dir, *, desc, workers=1, **kwargs):
            seen.update(mirrors=list(mirrors), names=list(names), workers=workers)

        monkeypatch.setattr(base_module, "download_files", fake_download_files)
        MITBIHAFDBDataset.download_to(temp_dir, verbose=False)

        assert seen["mirrors"] == [MITBIHAFDBDataset.DOWNLOAD_MIRROR]
        assert len(seen["names"]) == 3 * len(MITBIHAFDBDataset.RECORD_NAMES)
        assert {"04015.hea", "04015.atr", "04015.dat"} <= set(seen["names"])
        assert not (temp_dir / ".download-in-progress").exists()

    def test_ltafdb_reads_its_file_list_from_the_project_index(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import base as base_module
        from deepecgkit.datasets import ltafdb as ltafdb_module

        def fake_index(url, file_path, **kwargs):
            Path(file_path).write_text("00\n01\n")

        seen = {}

        def fake_download_files(mirrors, names, dest_dir, *, desc, workers=1, **kwargs):
            seen["names"] = list(names)

        monkeypatch.setattr(ltafdb_module, "download_file", fake_index)
        monkeypatch.setattr(base_module, "download_files", fake_download_files)
        LTAFDBDataset.download_to(temp_dir, verbose=False)

        assert seen["names"] == ["00.hea", "00.atr", "00.dat", "01.hea", "01.atr", "01.dat"]

    def test_ptbxl_fetches_metadata_and_both_resolutions_in_parallel(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import PTBXLDataset
        from deepecgkit.datasets import base as base_module
        from deepecgkit.datasets import ptbxl as ptbxl_module

        def fake_index(url, file_path, **kwargs):
            if Path(file_path).name == "ptbxl_database.csv":
                Path(file_path).write_text(
                    "ecg_id,filename_lr,filename_hr\n"
                    "1,records100/00000/00001_lr,records500/00000/00001_hr\n"
                )
            else:
                Path(file_path).write_text("")

        seen = {}

        def fake_download_files(mirrors, names, dest_dir, *, desc, workers=1, **kwargs):
            seen.update(names=list(names), workers=workers)

        monkeypatch.setattr(ptbxl_module, "download_file", fake_index)
        monkeypatch.setattr(base_module, "download_files", fake_download_files)
        PTBXLDataset.download_to(temp_dir, verbose=False)

        assert seen["names"] == [
            "ptbxl_database.csv",
            "scp_statements.csv",
            "records100/00000/00001_lr.hea",
            "records100/00000/00001_lr.dat",
            "records500/00000/00001_hr.hea",
            "records500/00000/00001_hr.dat",
        ]
        assert seen["workers"] == PTBXLDataset.DOWNLOAD_WORKERS

    def test_an_index_that_cannot_be_fetched_falls_back_to_the_archive(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import base as base_module
        from deepecgkit.datasets import ltafdb as ltafdb_module

        archive_calls = []

        def unreachable(*args, **kwargs):
            raise RuntimeError("Failed to download: mirror unreachable")

        def fake_archive(url, dest_dir, *, zip_name, desc, verbose=True, **kwargs):
            archive_calls.append(zip_name)

        monkeypatch.setattr(ltafdb_module, "download_file", unreachable)
        monkeypatch.setattr(base_module, "download_and_extract_zip", fake_archive)
        LTAFDBDataset.download_to(temp_dir, verbose=False)

        assert archive_calls == ["ltafdb-1.0.0.zip"]

    def test_a_finished_mirror_download_clears_an_abandoned_archive(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import base as base_module

        (temp_dir / "afdb-1.0.0.zip.part").write_bytes(b"abandoned on an earlier run")

        def fake_download_files(mirrors, names, dest_dir, *, desc, workers=1, **kwargs):
            for name in names:
                (Path(dest_dir) / name).write_bytes(b"x")

        monkeypatch.setattr(base_module, "download_files", fake_download_files)
        MITBIHAFDBDataset.download_to(temp_dir, verbose=False)

        assert not (temp_dir / "afdb-1.0.0.zip.part").exists()
        assert MITBIHAFDBDataset.is_downloaded(temp_dir)

    def test_an_unusable_mirror_falls_back_to_the_archive(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import base as base_module

        archive_calls = []

        def unreachable(*args, **kwargs):
            raise RuntimeError("Failed to download: mirror unreachable")

        def fake_archive(url, dest_dir, *, zip_name, desc, verbose=True, **kwargs):
            archive_calls.append((url, zip_name))

        monkeypatch.setattr(base_module, "download_files", unreachable)
        monkeypatch.setattr(base_module, "download_and_extract_zip", fake_archive)
        MITBIHAFDBDataset.download_to(temp_dir, verbose=False)

        assert archive_calls == [(MITBIHAFDBDataset.DOWNLOAD_URL, "afdb-1.0.0.zip")]

    def test_a_download_cut_short_between_files_is_not_a_dataset(self, temp_dir, monkeypatch):
        from deepecgkit.datasets import base as base_module

        def dies_part_way(mirrors, names, dest_dir, *, desc, workers=1, **kwargs):
            (Path(dest_dir) / "04015.hea").write_text("header")
            raise KeyboardInterrupt

        monkeypatch.setattr(base_module, "download_files", dies_part_way)
        with pytest.raises(KeyboardInterrupt):
            MITBIHAFDBDataset.download_to(temp_dir, verbose=False)

        assert (temp_dir / "04015.hea").exists()
        assert not MITBIHAFDBDataset.is_downloaded(temp_dir)


class TestLoadRecordings:
    """A recording file becomes a batch shaped for the model that will read it."""

    def _write_wfdb(self, folder, leads):
        import wfdb

        wfdb.wrsamp(
            "rec",
            fs=250,
            units=["mV"] * leads,
            sig_name=[f"ECG{i + 1}" for i in range(leads)],
            p_signal=np.random.randn(1000, leads),
            write_dir=str(folder),
        )
        return folder / "rec.hea"

    def test_extra_wfdb_leads_are_dropped_and_reported(self, temp_dir):
        recordings, rate, notes = load_recordings(self._write_wfdb(temp_dir, 2), input_channels=1)

        assert recordings.shape == (1, 1, 1000)
        assert rate == 250.0
        assert notes == ["the record has 2 leads, of which the first 1 is used"]

    def test_matching_wfdb_leads_are_kept_as_one_recording(self, temp_dir):
        recordings, _, notes = load_recordings(self._write_wfdb(temp_dir, 2), input_channels=2)

        assert recordings.shape == (1, 2, 1000)
        assert notes == []

    def test_array_rows_are_separate_recordings_for_a_single_lead_model(self, temp_dir):
        path = temp_dir / "batch.npy"
        np.save(str(path), np.zeros((3, 500)))

        recordings, rate, notes = load_recordings(path, input_channels=1)

        assert recordings.shape == (3, 1, 500)
        assert rate is None
        assert notes == []
