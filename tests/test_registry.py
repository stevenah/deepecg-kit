"""
Tests for deepecgkit.registry module.
"""

import pytest

import deepecgkit.datasets
import deepecgkit.models  # noqa: F401
from deepecgkit.registry import (
    get_dataset,
    get_dataset_info,
    get_dataset_names,
    get_model,
    get_model_info,
    get_model_names,
)


class TestModelRegistry:
    def test_all_models_registered(self):
        names = get_model_names()
        assert len(names) == 21
        expected = [
            "afmodel",
            "convnext-v2",
            "crnn",
            "deep-res-cnn",
            "dualnet",
            "fcn-wang",
            "gru",
            "inception-time",
            "kanres",
            "kanres-deep",
            "lstm",
            "mamba",
            "medformer",
            "resnet",
            "resnet-wang",
            "se-resnet",
            "simple-cnn",
            "tcn",
            "transformer",
            "xresnet",
            "xresnet1d-benchmark",
        ]
        assert names == expected

    def test_get_model_returns_class(self):
        cls = get_model("kanres")
        assert cls.__name__ == "KanResWideX"

    def test_get_model_afmodel(self):
        cls = get_model("afmodel")
        assert cls.__name__ == "AFModel"

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent")

    def test_model_info_has_description(self):
        info = get_model_info("kanres")
        assert len(info["description"]) > 0

    def test_model_info_has_class(self):
        for name in get_model_names():
            info = get_model_info(name)
            assert "class" in info
            assert "description" in info

    def test_afmodel_has_default_kwargs(self):
        info = get_model_info("afmodel")
        assert "recording_length" in info["default_kwargs"]


class TestDatasetRegistry:
    def test_all_datasets_registered(self):
        names = get_dataset_names()
        assert len(names) == 5
        expected = ["af-classification", "ltafdb", "mitbih-afdb", "ptbxl", "unified-af"]
        assert names == expected

    def test_get_dataset_returns_class(self):
        cls = get_dataset("af-classification")
        assert cls.__name__ == "AFClassificationDataset"

    def test_get_dataset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            get_dataset("nonexistent")

    def test_dataset_info_has_channels(self):
        info = get_dataset_info("ptbxl")
        assert info["input_channels"] == 12

    def test_dataset_info_af_classification(self):
        info = get_dataset_info("af-classification")
        assert info["input_channels"] == 1
        assert info["num_classes"] == 4

    def test_dataset_info_has_num_classes(self):
        for name in get_dataset_names():
            info = get_dataset_info(name)
            assert info["num_classes"] > 0

    def test_dataset_input_channels(self):
        expected = {
            "af-classification": 1,
            "ltafdb": 2,
            "mitbih-afdb": 2,
            "ptbxl": 12,
            "unified-af": 1,
        }
        for name, channels in expected.items():
            info = get_dataset_info(name)
            assert info["input_channels"] == channels
