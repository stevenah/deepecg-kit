"""
Tests for deepecgkit.cli module.
"""

import json
import re
import sys
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import torch

import deepecgkit.cli.commands.predict  # noqa: F401
from deepecgkit.__main__ import main as main_entry
from deepecgkit.cli import (
    DATASET_INPUT_CHANNELS,
    DATASET_NAMES,
    DATASET_NUM_CLASSES,
    MODEL_DESCRIPTIONS,
    MODEL_NAMES,
    CLILogger,
    evaluate,
    explain,
    list_datasets,
    list_models,
    load_config,
    main,
    predict,
    probe,
    show_info,
    train,
)
from deepecgkit.cli.commands.evaluate import (
    _dataset_kwargs_for_checkpoint,
    _read_train_config,
)
from deepecgkit.cli.commands.probe import parse_source_checkpoints
from deepecgkit.cli.registry import _get_dataset_registry, _get_model_registry
from deepecgkit.models.factory import build_model

predict_module = sys.modules["deepecgkit.cli.commands.predict"]


class TestCLILogger:
    """Test the CLILogger class."""

    def test_logger_default_init(self):
        """Test default logger initialization."""
        logger = CLILogger()
        assert logger.verbose is False
        assert logger.quiet is False

    def test_logger_verbose_init(self):
        """Test verbose logger initialization."""
        logger = CLILogger(verbose=True)
        assert logger.verbose is True
        assert logger.quiet is False

    def test_logger_quiet_init(self):
        """Test quiet logger initialization."""
        logger = CLILogger(quiet=True)
        assert logger.verbose is False
        assert logger.quiet is True

    def test_logger_info_normal(self, capsys):
        """Test info output in normal mode."""
        logger = CLILogger()
        logger.info("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_logger_info_quiet(self, capsys):
        """Test info output in quiet mode."""
        logger = CLILogger(quiet=True)
        logger.info("test message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_logger_debug_verbose(self, capsys):
        """Test debug output in verbose mode."""
        logger = CLILogger(verbose=True)
        logger.debug("debug message")
        captured = capsys.readouterr()
        assert "[DEBUG] debug message" in captured.out

    def test_logger_debug_not_verbose(self, capsys):
        """Test debug output when not verbose."""
        logger = CLILogger(verbose=False)
        logger.debug("debug message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_logger_debug_quiet_overrides_verbose(self, capsys):
        """Test that quiet mode overrides verbose for debug."""
        logger = CLILogger(verbose=True, quiet=True)
        logger.debug("debug message")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_logger_error(self, capsys):
        """Test error output."""
        logger = CLILogger()
        logger.error("error message")
        captured = capsys.readouterr()
        assert "[ERROR] error message" in captured.err

    def test_logger_error_in_quiet_mode(self, capsys):
        """Test error output in quiet mode (should still print)."""
        logger = CLILogger(quiet=True)
        logger.error("error message")
        captured = capsys.readouterr()
        assert "[ERROR] error message" in captured.err

    def test_logger_warning(self, capsys):
        """Test warning output."""
        logger = CLILogger()
        logger.warning("warning message")
        captured = capsys.readouterr()
        assert "[WARNING] warning message" in captured.err

    def test_logger_warning_quiet(self, capsys):
        """Test warning output in quiet mode."""
        logger = CLILogger(quiet=True)
        logger.warning("warning message")
        captured = capsys.readouterr()
        assert captured.err == ""


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_json_config(self, temp_dir):
        """Test loading JSON configuration."""
        config_file = temp_dir / "config.json"
        config_data = {"verbose": True, "train": {"epochs": 100}}
        config_file.write_text(json.dumps(config_data))

        loaded = load_config(str(config_file))
        assert loaded == config_data

    def test_load_yaml_config(self, temp_dir):
        """Test loading YAML configuration."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("verbose: true\ntrain:\n  epochs: 100\n")

        loaded = load_config(str(config_file))
        assert loaded["verbose"] is True
        assert loaded["train"]["epochs"] == 100

    def test_load_yml_config(self, temp_dir):
        """Test loading .yml configuration."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("model: kanres\n")

        loaded = load_config(str(config_file))
        assert loaded["model"] == "kanres"

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.json")

    def test_load_config_unsupported_format(self, temp_dir):
        """Test loading unsupported config format."""
        config_file = temp_dir / "config.txt"
        config_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported config format"):
            load_config(str(config_file))


class TestConstants:
    """Test CLI constants."""

    def test_model_names_not_empty(self):
        """Test that MODEL_NAMES contains models."""
        assert len(MODEL_NAMES) > 0
        assert "kanres" in MODEL_NAMES
        assert "afmodel" in MODEL_NAMES

    def test_dataset_names_not_empty(self):
        """Test that DATASET_NAMES contains datasets."""
        assert len(DATASET_NAMES) > 0
        assert "af-classification" in DATASET_NAMES

    def test_dataset_num_classes_matches_names(self):
        """Test that all datasets have num_classes defined."""
        for name in DATASET_NAMES:
            assert name in DATASET_NUM_CLASSES
            assert DATASET_NUM_CLASSES[name] > 0

    def test_dataset_input_channels_matches_names(self):
        """Test that all datasets have input_channels defined."""
        for name in DATASET_NAMES:
            assert name in DATASET_INPUT_CHANNELS
            assert DATASET_INPUT_CHANNELS[name] > 0

    def test_model_descriptions_exist(self):
        """Test that all models have descriptions."""
        for name in MODEL_NAMES:
            assert name in MODEL_DESCRIPTIONS
            assert len(MODEL_DESCRIPTIONS[name]) > 0


class TestListModels:
    """Test the list_models function."""

    def test_list_models_output(self, capsys):
        """Test list_models prints model information."""
        list_models()
        captured = capsys.readouterr()

        assert "Available models:" in captured.out
        for model_name in MODEL_NAMES:
            assert model_name in captured.out

    def test_list_models_includes_descriptions(self, capsys):
        """Test list_models includes descriptions."""
        list_models()
        captured = capsys.readouterr()

        assert "Description:" in captured.out

    def test_list_models_lists_only_published_weights(self, capsys, monkeypatch):
        """Weights are advertised once there is something to download, not before."""
        from deepecgkit.utils import weights as weights_module

        list_models()
        assert "Pretrained weights:" not in capsys.readouterr().out

        monkeypatch.setitem(
            weights_module.WEIGHTS_REGISTRY,
            "kanres-published",
            {
                "url": "https://example.org/kanres-published.pt",
                "model_class": "KanResWideX",
                "model_kwargs": {"input_channels": 1, "output_size": 4},
                "description": "published",
            },
        )
        list_models()
        assert "Pretrained weights: kanres-published" in capsys.readouterr().out

    def test_list_models_shows_input_length_limits(self, capsys):
        """A model that cannot take every signal length says so in the listing."""
        list_models()
        assert "Input length: at least 7971 samples per lead" in capsys.readouterr().out


class TestListDatasets:
    """Test the list_datasets function."""

    def test_list_datasets_output(self, capsys):
        """Test list_datasets prints dataset information."""
        list_datasets()
        captured = capsys.readouterr()

        assert "Available datasets:" in captured.out
        for dataset_name in DATASET_NAMES:
            assert dataset_name in captured.out

    def test_list_datasets_includes_classes(self, capsys):
        """Test list_datasets includes class count."""
        list_datasets()
        captured = capsys.readouterr()

        assert "Classes:" in captured.out


class TestShowInfo:
    """Test the show_info function."""

    def test_show_info_unknown_model(self, capsys):
        """Test show_info with unknown model."""
        result = show_info("unknown-model")

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown model" in captured.err

    def test_show_info_valid_model(self, capsys):
        """Test show_info with valid model."""
        result = show_info("kanres")

        assert result == 0
        captured = capsys.readouterr()
        assert "Model: kanres" in captured.out
        assert "Total parameters:" in captured.out
        assert "Architecture:" in captured.out


class TestMainCLI:
    """Test the main CLI entry point."""

    def test_main_no_command(self, capsys):
        """Test main with no command shows help."""
        result = main([])

        assert result == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "deepecg" in captured.out.lower()

    def test_main_help_flag(self, capsys):
        """Test main with --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0

    def test_main_list_models_command(self, capsys):
        """Test main with list-models command."""
        result = main(["list-models"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Available models:" in captured.out

    def test_main_list_datasets_command(self, capsys):
        """Test main with list-datasets command."""
        result = main(["list-datasets"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Available datasets:" in captured.out

    def test_main_info_command(self, capsys):
        """Test main with info command."""
        result = main(["info", "-m", "kanres"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Model: kanres" in captured.out

    def test_main_info_command_unknown_model(self, capsys):
        """Test main with info command for unknown model."""
        with pytest.raises(SystemExit):
            main(["info", "-m", "unknown-model"])

    def test_main_verbose_flag(self):
        """Test main with --verbose flag."""
        with patch("deepecgkit.cli.list_models") as mock_list:
            main(["--verbose", "list-models"])
            mock_list.assert_called_once()

    def test_main_quiet_flag(self):
        """Test main with --quiet flag."""
        with patch("deepecgkit.cli.list_models") as mock_list:
            main(["--quiet", "list-models"])
            mock_list.assert_called_once()

    def test_main_config_file(self, temp_dir, capsys):
        """Test main with config file."""
        config_file = temp_dir / "config.json"
        config_file.write_text(json.dumps({"verbose": True}))

        result = main(["--config", str(config_file), "list-models"])

        assert result == 0

    def test_main_config_file_not_found(self, capsys):
        """Test main with non-existent config file."""
        result = main(["--config", "/nonexistent/config.json", "list-models"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to load config" in captured.err


class TestTrainCommand:
    """Test the train command."""

    def test_train_missing_required_args(self):
        """Test train command with missing required arguments."""
        with pytest.raises(SystemExit):
            main(["train"])

    def test_train_missing_dataset(self):
        """Test train command with missing dataset."""
        with pytest.raises(SystemExit):
            main(["train", "-m", "kanres"])

    def test_train_missing_model(self):
        """Test train command with missing model."""
        with pytest.raises(SystemExit):
            main(["train", "-d", "af-classification"])

    def test_train_invalid_model(self):
        """Test train command with invalid model."""
        with pytest.raises(SystemExit):
            main(["train", "-m", "invalid-model", "-d", "af-classification"])

    def test_train_invalid_dataset(self):
        """Test train command with invalid dataset."""
        with pytest.raises(SystemExit):
            main(["train", "-m", "kanres", "-d", "invalid-dataset"])


class TestEvaluateCommand:
    """Test the evaluate command."""

    def test_evaluate_missing_checkpoint(self):
        """Test evaluate command with missing checkpoint."""
        with pytest.raises(SystemExit):
            main(["evaluate", "-d", "af-classification"])

    def test_evaluate_missing_dataset(self):
        """Test evaluate command with missing dataset."""
        with pytest.raises(SystemExit):
            main(["evaluate", "--checkpoint", "model.ckpt", "-m", "kanres"])

    def test_evaluate_checkpoint_not_found(self, capsys):
        """Test evaluate command with non-existent checkpoint."""
        result = main(
            [
                "evaluate",
                "--checkpoint",
                "/nonexistent/model.ckpt",
                "-m",
                "kanres",
                "-d",
                "af-classification",
            ]
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err


class TestPredictCommand:
    """Test the predict command."""

    def test_predict_missing_checkpoint(self):
        """Test predict command with missing checkpoint."""
        with pytest.raises(SystemExit):
            main(["predict", "-i", "input.npy"])

    def test_predict_missing_input(self):
        """Test predict command with missing input."""
        with pytest.raises(SystemExit):
            main(["predict", "--checkpoint", "model.ckpt", "-m", "kanres"])

    def test_predict_checkpoint_not_found(self, capsys):
        """Test predict command with non-existent checkpoint."""
        result = main(
            [
                "predict",
                "--checkpoint",
                "/nonexistent/model.ckpt",
                "-m",
                "kanres",
                "-i",
                "input.npy",
            ]
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err

    def test_predict_input_not_found(self, temp_dir, capsys):
        """Test predict command with non-existent input file."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        result = main(
            [
                "predict",
                "--checkpoint",
                str(checkpoint_file),
                "-m",
                "kanres",
                "-i",
                "/nonexistent/input.npy",
            ]
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Input file not found" in captured.err


class TestTrainFunction:
    """Test the train function directly."""

    def test_train_unknown_model(self, capsys):
        """Test train with unknown model name."""
        result = train(
            model_name="unknown-model",
            dataset_name="af-classification",
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown model" in captured.err

    def test_train_unknown_dataset(self, capsys):
        """Test train with unknown dataset name."""
        result = train(
            model_name="kanres",
            dataset_name="unknown-dataset",
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown dataset" in captured.err


class TestEvaluateFunction:
    """Test the evaluate function directly."""

    def test_evaluate_checkpoint_not_found(self, capsys):
        """Test evaluate with non-existent checkpoint."""
        result = evaluate(
            checkpoint="/nonexistent/model.ckpt",
            dataset_name="af-classification",
            model_name="kanres",
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err

    def test_evaluate_unknown_dataset(self, temp_dir, capsys):
        """Test evaluate with unknown dataset."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        result = evaluate(
            checkpoint=str(checkpoint_file),
            dataset_name="unknown-dataset",
            model_name="kanres",
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown dataset" in captured.err


class TestEvaluateCalibrationWiring:
    """Test how the evaluate command reconstructs a checkpoint's task shape."""

    def test_multi_label_checkpoint_rebuilds_ptbxl_multi_label(self):
        """PTB-XL must be rebuilt multi-label or its targets come back the wrong rank."""
        kwargs, multi_label = _dataset_kwargs_for_checkpoint(
            "ptbxl", {"multi_label": True}, False, None
        )

        assert multi_label is True
        assert kwargs["multi_label"] is True
        assert kwargs["task"] == "superclass"

    def test_binary_checkpoint_rebuilds_af_dataset_collapsed(self):
        """A one-logit head was trained against collapsed AF labels."""
        kwargs, multi_label = _dataset_kwargs_for_checkpoint(
            "mitbih-afdb", {"binary_classification": True}, False, None
        )

        assert multi_label is False
        assert kwargs["binary_classification"] is True

    def test_multiclass_checkpoint_asks_for_no_special_labelling(self):
        """The four-class case needs no flags, which is why it worked before."""
        kwargs, multi_label = _dataset_kwargs_for_checkpoint("af-classification", {}, False, None)

        assert multi_label is False
        assert "multi_label" not in kwargs
        assert "binary_classification" not in kwargs

    def test_data_dir_and_force_download_are_forwarded(self):
        """The flags that already worked must keep working."""
        kwargs, _ = _dataset_kwargs_for_checkpoint("ptbxl", {}, True, "/data/ptbxl")

        assert kwargs["force_download"] is True
        assert kwargs["data_dir"] == "/data/ptbxl"

    def test_train_config_is_recovered_from_a_checkpoint(self, temp_dir):
        """The shape flags come from the checkpoint, so no new CLI flags are needed."""
        path = temp_dir / "model.pt"
        torch.save({"train_config": {"multi_label": True}, "model_state_dict": {}}, path)

        assert _read_train_config(str(path)) == {"multi_label": True}

    def test_checkpoint_without_train_config_reads_as_empty(self, temp_dir):
        """A bare state dict must not crash the shape lookup."""
        path = temp_dir / "bare.pt"
        torch.save({"model_state_dict": {}}, path)

        assert _read_train_config(str(path)) == {}

    def test_invalid_split_is_rejected(self, temp_dir, capsys):
        """An unusable split should fail before any dataset work happens."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        result = evaluate(
            checkpoint=str(checkpoint_file),
            dataset_name="af-classification",
            model_name="kanres",
            split="train",
        )

        assert result == 1
        assert "Invalid split" in capsys.readouterr().err


class TestPredictFunction:
    """Test the predict function directly."""

    def test_predict_checkpoint_not_found(self, capsys):
        """Test predict with non-existent checkpoint."""
        result = predict(
            checkpoint="/nonexistent/model.ckpt",
            input_path="input.npy",
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err

    def test_predict_input_not_found(self, temp_dir, capsys):
        """Test predict with non-existent input file."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        result = predict(
            checkpoint=str(checkpoint_file),
            input_path="/nonexistent/input.npy",
        )

        assert result == 1
        captured = capsys.readouterr()
        assert "Input file not found" in captured.err

    def test_predict_unsupported_format(self, temp_dir, capsys):
        """Test predict with unsupported input format."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        input_file = temp_dir / "input.xyz"
        input_file.write_text("data")

        with (
            patch.object(predict_module, "torch") as mock_torch,
            patch.object(predict_module, "require_model", return_value=None),
        ):
            mock_torch.load.return_value = {"model_state_dict": {}}

            result = predict(
                checkpoint=str(checkpoint_file),
                input_path=str(input_file),
                model_name="kanres",
            )

        assert result == 1
        captured = capsys.readouterr()
        assert "Unsupported input format" in captured.err


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_model_registry_lazy_loading(self):
        """Test that model registry is lazily loaded."""
        registry = _get_model_registry()
        assert "kanres" in registry
        assert "afmodel" in registry

    def test_dataset_registry_lazy_loading(self):
        """Test that dataset registry is lazily loaded."""
        registry = _get_dataset_registry()
        assert "af-classification" in registry

    def test_main_module_entry_point(self):
        """Test that __main__ module imports correctly."""
        assert main_entry is not None


class TestCLIParserDefaults:
    """Test CLI argument parser defaults."""

    def test_train_default_values(self):
        """Test train command default argument values."""
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            main(["train", "-m", "kanres", "-d", "af-classification"])

            call_kwargs = mock_train.call_args[1]
            assert call_kwargs["epochs"] == 50
            assert call_kwargs["batch_size"] == 32
            assert call_kwargs["learning_rate"] == 1e-3
            assert call_kwargs["val_split"] == 0.2
            assert call_kwargs["test_split"] == 0.1
            assert call_kwargs["num_workers"] == 0
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["devices"] == 1
            assert call_kwargs["early_stopping_patience"] == 10
            assert call_kwargs["seed"] == 42
            assert call_kwargs["sampling_rate"] is None
            assert call_kwargs["segment_duration"] is None

    def test_probe_default_values(self):
        """Test probe command default argument values."""
        with patch("deepecgkit.cli.probe") as mock_probe:
            mock_probe.return_value = 0
            main(["probe", "-m", "kanres", "-o", "probe/"])

            call_kwargs = mock_probe.call_args[1]
            assert call_kwargs["metric"] == "auroc_macro"
            assert call_kwargs["input_channels"] == 1
            assert call_kwargs["sampling_rate"] == 300
            assert call_kwargs["embedding_plots"] == "diagonal"
            assert call_kwargs["random_init"] is True
            assert call_kwargs["plots"] is True
            assert call_kwargs["latex"] is True
            assert call_kwargs["source_checkpoints"] == []
            assert call_kwargs["targets"] == []
            assert call_kwargs["val_split"] == 0.2
            assert call_kwargs["test_split"] == 0.1
            assert call_kwargs["seed"] == 42
            assert call_kwargs["batch_size"] == 32
            assert call_kwargs["num_workers"] == 0
            assert call_kwargs["accelerator"] == "auto"

    def test_probe_custom_values(self):
        """Test probe command with custom argument values."""
        with patch("deepecgkit.cli.probe") as mock_probe:
            mock_probe.return_value = 0
            main(
                [
                    "probe",
                    "-m",
                    "resnet",
                    "-o",
                    "out/",
                    "--source-checkpoint",
                    "ptbxl=a.pt",
                    "--source-checkpoint",
                    "ltafdb=b.pt",
                    "--target",
                    "ptbxl",
                    "--metric",
                    "accuracy",
                    "--no-random-init",
                    "--no-plots",
                    "--no-latex",
                    "--embedding-plots",
                    "all",
                ]
            )

            call_kwargs = mock_probe.call_args[1]
            assert call_kwargs["source_checkpoints"] == ["ptbxl=a.pt", "ltafdb=b.pt"]
            assert call_kwargs["targets"] == ["ptbxl"]
            assert call_kwargs["metric"] == "accuracy"
            assert call_kwargs["random_init"] is False
            assert call_kwargs["plots"] is False
            assert call_kwargs["latex"] is False
            assert call_kwargs["embedding_plots"] == "all"

    def test_train_custom_values(self):
        """Test train command with custom argument values."""
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            main(
                [
                    "train",
                    "-m",
                    "kanres",
                    "-d",
                    "af-classification",
                    "--epochs",
                    "100",
                    "--batch-size",
                    "64",
                    "--learning-rate",
                    "0.01",
                    "--val-split",
                    "0.15",
                    "--test-split",
                    "0.05",
                    "--num-workers",
                    "8",
                    "--accelerator",
                    "gpu",
                    "--devices",
                    "2",
                    "--early-stopping-patience",
                    "20",
                    "--seed",
                    "123",
                ]
            )

            call_kwargs = mock_train.call_args[1]
            assert call_kwargs["epochs"] == 100
            assert call_kwargs["batch_size"] == 64
            assert call_kwargs["learning_rate"] == 0.01
            assert call_kwargs["val_split"] == 0.15
            assert call_kwargs["test_split"] == 0.05
            assert call_kwargs["num_workers"] == 8
            assert call_kwargs["accelerator"] == "gpu"
            assert call_kwargs["devices"] == 2
            assert call_kwargs["early_stopping_patience"] == 20
            assert call_kwargs["seed"] == 123

    def test_evaluate_default_values(self, temp_dir):
        """Test evaluate command default argument values."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        with patch("deepecgkit.cli.evaluate") as mock_eval:
            mock_eval.return_value = 0
            main(
                [
                    "evaluate",
                    "--checkpoint",
                    str(checkpoint_file),
                    "-m",
                    "kanres",
                    "-d",
                    "af-classification",
                ]
            )

            call_kwargs = mock_eval.call_args[1]
            assert call_kwargs["batch_size"] == 32
            assert call_kwargs["num_workers"] == 0
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["devices"] == 1
            assert call_kwargs["split"] == "test"
            assert call_kwargs["sampling_rate"] is None

    def test_predict_default_values(self, temp_dir):
        """Test predict command default argument values."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()
        input_file = temp_dir / "input.npy"
        np.save(str(input_file), np.random.randn(3000))

        with patch("deepecgkit.cli.predict") as mock_predict:
            mock_predict.return_value = 0
            main(
                [
                    "predict",
                    "--checkpoint",
                    str(checkpoint_file),
                    "-m",
                    "kanres",
                    "-i",
                    str(input_file),
                ]
            )

            call_kwargs = mock_predict.call_args[1]
            assert call_kwargs["batch_size"] == 32
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["output_path"] is None


class TestProbeCommand:
    """Test the probe command."""

    def test_model_is_required(self):
        """The model architecture must be given."""
        with pytest.raises(SystemExit):
            main(["probe", "-o", "probe/"])

    def test_output_is_required(self):
        """The output directory must be given."""
        with pytest.raises(SystemExit):
            main(["probe", "-m", "kanres"])

    def test_unknown_model(self, capsys, temp_dir):
        """An unregistered model is a usage error from argparse."""
        with pytest.raises(SystemExit):
            main(["probe", "-m", "not-a-model", "-o", str(temp_dir)])

    def test_unknown_model_reaches_the_command(self, capsys, temp_dir):
        """Calling the command directly with an unknown model exits non-zero."""
        exit_code = probe(str(temp_dir), "not-a-model", [])
        assert exit_code == 1
        assert "Unknown model" in capsys.readouterr().err

    def test_malformed_source_checkpoint(self, capsys, temp_dir):
        """A source without a separator exits non-zero with a clear message."""
        exit_code = main(
            [
                "probe",
                "-m",
                "kanres",
                "-o",
                str(temp_dir),
                "--source-checkpoint",
                "af-classification",
            ]
        )
        assert exit_code == 1
        assert "Malformed source" in capsys.readouterr().err

    def test_checkpoint_not_found(self, capsys, temp_dir):
        """A missing checkpoint exits non-zero with a clear message."""
        exit_code = main(
            [
                "probe",
                "-m",
                "kanres",
                "-o",
                str(temp_dir),
                "--source-checkpoint",
                f"af-classification={temp_dir / 'absent.pt'}",
            ]
        )
        assert exit_code == 1
        assert "Checkpoint not found" in capsys.readouterr().err

    def test_unknown_source_dataset(self, capsys, temp_dir):
        """A source naming an unregistered dataset exits non-zero."""
        exit_code = probe(str(temp_dir), "kanres", ["not-a-dataset=some.pt"])
        assert exit_code == 1
        assert "Unknown source dataset" in capsys.readouterr().err

    def test_nothing_to_probe(self, capsys, temp_dir):
        """Dropping the floor with no sources leaves nothing to run."""
        exit_code = probe(str(temp_dir), "kanres", [], random_init=False)
        assert exit_code == 1
        assert "Nothing to probe" in capsys.readouterr().err

    def test_unknown_metric(self, capsys, temp_dir):
        """An unrecognised metric exits non-zero before any data is touched."""
        exit_code = probe(str(temp_dir), "kanres", [], metric="rmse")
        assert exit_code == 1
        assert "Unknown metric" in capsys.readouterr().err

    def test_unknown_embedding_mode(self, capsys, temp_dir):
        """An unrecognised embedding plot mode exits non-zero."""
        exit_code = probe(str(temp_dir), "kanres", [], embedding_plots="some")
        assert exit_code == 1
        assert "Unknown embedding plot mode" in capsys.readouterr().err


class TestParseSourceCheckpoints:
    """Repeated DATASET=PATH arguments are parsed into a mapping."""

    def test_parses_pairs(self):
        """Each entry becomes one source."""
        parsed = parse_source_checkpoints(["ptbxl=a.pt", "ltafdb=b.pt"])
        assert parsed == {"ptbxl": "a.pt", "ltafdb": "b.pt"}

    def test_tolerates_surrounding_space(self):
        """Whitespace around either half is stripped."""
        assert parse_source_checkpoints([" ptbxl = a.pt "]) == {"ptbxl": "a.pt"}

    def test_keeps_a_path_containing_a_separator(self):
        """Only the first separator splits the entry, so paths may contain one."""
        assert parse_source_checkpoints(["ptbxl=runs/a=1/best.pt"]) == {"ptbxl": "runs/a=1/best.pt"}

    def test_rejects_a_missing_separator(self):
        """An entry with no separator is rejected."""
        with pytest.raises(ValueError, match="Malformed source"):
            parse_source_checkpoints(["ptbxl"])

    def test_rejects_an_empty_half(self):
        """Both the dataset and the path are required."""
        with pytest.raises(ValueError, match="Malformed source"):
            parse_source_checkpoints(["ptbxl="])

    def test_rejects_a_repeated_dataset(self):
        """One checkpoint per source dataset."""
        with pytest.raises(ValueError, match="more than once"):
            parse_source_checkpoints(["ptbxl=a.pt", "ptbxl=b.pt"])


class TestExtractFeaturesCommand:
    """Test the extract-features command."""

    def test_missing_weight_source(self):
        """Neither a checkpoint nor pretrained weights is a usage error."""
        with pytest.raises(SystemExit):
            main(["extract-features", "-m", "kanres", "-i", "input.npy", "-o", "out.npz"])

    def test_missing_data_source(self):
        """Neither a dataset nor an input file is a usage error."""
        with pytest.raises(SystemExit):
            main(["extract-features", "--checkpoint", "m.ckpt", "-m", "kanres", "-o", "out.npz"])

    def test_missing_output(self):
        """The output path is required."""
        with pytest.raises(SystemExit):
            main(["extract-features", "--checkpoint", "m.ckpt", "-m", "kanres", "-i", "in.npy"])

    def test_checkpoint_and_weights_are_exclusive(self):
        """Only one weight source may be given."""
        with pytest.raises(SystemExit):
            main(
                [
                    "extract-features",
                    "--checkpoint",
                    "m.ckpt",
                    "-w",
                    "kanres-af-30s",
                    "-m",
                    "kanres",
                    "-i",
                    "in.npy",
                    "-o",
                    "out.npz",
                ]
            )

    def test_checkpoint_not_found(self, capsys, temp_dir):
        """A missing checkpoint exits non-zero with a clear message."""
        input_file = temp_dir / "in.npy"
        np.save(str(input_file), np.random.randn(1, 1, 3000))

        exit_code = main(
            [
                "extract-features",
                "--checkpoint",
                str(temp_dir / "absent.ckpt"),
                "-m",
                "kanres",
                "-i",
                str(input_file),
                "-o",
                str(temp_dir / "out.npz"),
            ]
        )

        assert exit_code == 1
        assert "Checkpoint not found" in capsys.readouterr().err

    def test_input_not_found(self, capsys, temp_dir):
        """A missing input file exits non-zero with a clear message."""
        checkpoint = temp_dir / "m.ckpt"
        model = build_model("kanres", input_channels=1, output_size=4)
        torch.save({"model_state_dict": model.state_dict()}, str(checkpoint))

        exit_code = main(
            [
                "extract-features",
                "--checkpoint",
                str(checkpoint),
                "-m",
                "kanres",
                "-i",
                str(temp_dir / "absent.npy"),
                "-o",
                str(temp_dir / "out.npz"),
            ]
        )

        assert exit_code == 1
        assert "Input file not found" in capsys.readouterr().err

    def test_weights_model_mismatch(self, capsys, temp_dir):
        """Pretrained weights belonging to another model are rejected early."""
        input_file = temp_dir / "in.npy"
        np.save(str(input_file), np.random.randn(1, 1, 3000))

        exit_code = main(
            [
                "extract-features",
                "-w",
                "kanres-af-30s",
                "-m",
                "simple-cnn",
                "-i",
                str(input_file),
                "-o",
                str(temp_dir / "out.npz"),
            ]
        )

        assert exit_code == 1
        assert "belong to model 'kanres'" in capsys.readouterr().err

    def test_end_to_end_writes_features(self, temp_dir):
        """A checkpoint and a .npy file produce a readable .npz."""
        checkpoint = temp_dir / "m.ckpt"
        model = build_model("kanres", input_channels=1, output_size=4)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {"name": "kanres", "input_channels": 1, "output_size": 4},
            },
            str(checkpoint),
        )
        input_file = temp_dir / "in.npy"
        np.save(str(input_file), np.random.randn(6, 1, 3000))
        output_file = temp_dir / "features.npz"

        exit_code = main(
            [
                "--quiet",
                "extract-features",
                "--checkpoint",
                str(checkpoint),
                "-m",
                "kanres",
                "-i",
                str(input_file),
                "-o",
                str(output_file),
                "--accelerator",
                "cpu",
            ]
        )

        assert exit_code == 0
        assert output_file.exists()

        loaded = np.load(str(output_file), allow_pickle=True)
        assert loaded["features"].shape == (6, 64)
        assert int(loaded["feature_dim"]) == 64
        assert str(loaded["model_name"]) == "kanres"


class TestExplainCommand:
    """Test the explain command."""

    def _checkpoint(self, temp_dir, leads=1, output_size=4):
        """Write a kanres checkpoint the command can load."""
        path = temp_dir / "m.ckpt"
        model = build_model("kanres", input_channels=leads, output_size=output_size)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "name": "kanres",
                    "input_channels": leads,
                    "output_size": output_size,
                },
                "train_config": {},
            },
            str(path),
        )
        return path

    def _recording(self, temp_dir, leads=1, length=3000, samples=1):
        """Write a synthetic ECG with detectable beats to a .npy file."""
        time = np.arange(length) / 500.0
        trace = np.zeros(length)
        index = 0
        while index + 0.5 < length / 500.0:
            centre = index + 0.5
            for offset, amplitude, width in (
                (-0.20, 0.12, 0.020),
                (0.0, 1.00, 0.008),
                (0.22, 0.25, 0.040),
            ):
                trace += amplitude * np.exp(-((time - (centre + offset)) ** 2) / (2 * width**2))
            index += 1
        batch = np.stack([np.stack([trace] * leads) for _ in range(samples)])
        path = temp_dir / "in.npy"
        np.save(str(path), batch)
        return path

    def test_missing_weight_source(self):
        """Neither a checkpoint nor pretrained weights is a usage error."""
        with pytest.raises(SystemExit):
            main(["explain", "-m", "kanres", "-i", "in.npy", "-o", "out"])

    def test_missing_data_source(self):
        """Neither a dataset nor an input file is a usage error."""
        with pytest.raises(SystemExit):
            main(["explain", "--checkpoint", "m.ckpt", "-m", "kanres", "-o", "out"])

    def test_missing_output(self):
        """The output directory is required."""
        with pytest.raises(SystemExit):
            main(["explain", "--checkpoint", "m.ckpt", "-m", "kanres", "-i", "in.npy"])

    def test_checkpoint_and_weights_are_exclusive(self):
        """Only one weight source may be given."""
        with pytest.raises(SystemExit):
            main(
                [
                    "explain",
                    "--checkpoint",
                    "m.ckpt",
                    "-w",
                    "kanres-af-30s",
                    "-m",
                    "kanres",
                    "-i",
                    "in.npy",
                    "-o",
                    "out",
                ]
            )

    def test_dataset_and_input_are_exclusive(self):
        """Only one data source may be given."""
        with pytest.raises(SystemExit):
            main(
                [
                    "explain",
                    "--checkpoint",
                    "m.ckpt",
                    "-m",
                    "kanres",
                    "-d",
                    "af-classification",
                    "-i",
                    "in.npy",
                    "-o",
                    "out",
                ]
            )

    def test_an_unknown_method_is_rejected_by_argparse(self):
        """The method list is enforced at parse time."""
        with pytest.raises(SystemExit):
            main(
                [
                    "explain",
                    "--checkpoint",
                    "m.ckpt",
                    "-m",
                    "kanres",
                    "-i",
                    "in.npy",
                    "-o",
                    "out",
                    "--method",
                    "gradcam",
                ]
            )

    def test_checkpoint_not_found(self, capsys, temp_dir):
        """A missing checkpoint exits non-zero with a clear message."""
        exit_code = explain(
            output_dir=str(temp_dir / "out"),
            model_name="kanres",
            checkpoint=str(temp_dir / "absent.ckpt"),
            input_path=str(self._recording(temp_dir)),
        )

        assert exit_code == 1
        assert "Checkpoint not found" in capsys.readouterr().err

    def test_input_not_found(self, capsys, temp_dir):
        """A missing input file exits non-zero with a clear message."""
        exit_code = explain(
            output_dir=str(temp_dir / "out"),
            model_name="kanres",
            checkpoint=str(self._checkpoint(temp_dir)),
            input_path=str(temp_dir / "absent.npy"),
        )

        assert exit_code == 1
        assert "Input file not found" in capsys.readouterr().err

    def test_unknown_model_is_reported(self, capsys, temp_dir):
        """An unregistered model name is refused before any work."""
        exit_code = explain(
            output_dir=str(temp_dir / "out"),
            model_name="not-a-model",
            checkpoint=str(self._checkpoint(temp_dir)),
            input_path=str(self._recording(temp_dir)),
        )

        assert exit_code == 1
        assert "Unknown model" in capsys.readouterr().err

    def test_invalid_target_class_is_rejected(self, capsys, temp_dir):
        """A target that is neither an integer nor 'all' is named."""
        exit_code = explain(
            output_dir=str(temp_dir / "out"),
            model_name="kanres",
            checkpoint=str(self._checkpoint(temp_dir)),
            input_path=str(self._recording(temp_dir)),
            target_class="middle",
        )

        assert exit_code == 1
        assert "must be an integer or 'all'" in capsys.readouterr().err

    def test_non_positive_max_samples_is_rejected(self, capsys, temp_dir):
        """Explaining zero recordings is a usage error, not an empty result."""
        exit_code = explain(
            output_dir=str(temp_dir / "out"),
            model_name="kanres",
            checkpoint=str(self._checkpoint(temp_dir)),
            input_path=str(self._recording(temp_dir)),
            max_samples=0,
        )

        assert exit_code == 1
        assert "--max-samples must be at least 1" in capsys.readouterr().err

    def test_a_signal_too_short_for_the_model_is_explained(self, capsys, temp_dir):
        """AFModel's floor is reported with the reason cropping cannot fix it."""
        path = temp_dir / "af.ckpt"
        model = build_model("afmodel", input_channels=1, output_size=4)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {"name": "afmodel", "input_channels": 1, "output_size": 4},
                "train_config": {},
            },
            str(path),
        )

        exit_code = explain(
            output_dir=str(temp_dir / "out"),
            model_name="afmodel",
            checkpoint=str(path),
            input_path=str(self._recording(temp_dir)),
            accelerator="cpu",
        )

        assert exit_code == 1
        assert "needs at least 7971" in capsys.readouterr().err

    def test_end_to_end_writes_every_artifact(self, temp_dir):
        """A checkpoint and a .npy file produce arrays, a summary and three plots."""
        output_dir = temp_dir / "out"

        exit_code = main(
            [
                "--quiet",
                "explain",
                "--checkpoint",
                str(self._checkpoint(temp_dir)),
                "-m",
                "kanres",
                "-i",
                str(self._recording(temp_dir)),
                "-o",
                str(output_dir),
                "--accelerator",
                "cpu",
                "--sampling-rate",
                "500",
            ]
        )

        assert exit_code == 0
        for name in (
            "attribution.npz",
            "summary.json",
            "signal_attribution.png",
            "lead_importance.png",
            "beat_profile.png",
        ):
            assert (output_dir / name).stat().st_size > 0

        loaded = np.load(str(output_dir / "attribution.npz"), allow_pickle=False)
        assert loaded["values"].shape == (1, 1, 3000)
        assert loaded["signal"].shape == (1, 1, 3000)
        assert str(loaded["method"]) == "saliency"

        summary = json.loads((output_dir / "summary.json").read_text())
        assert summary["model_name"] == "kanres"
        assert summary["target_class"] == "predicted"
        assert sum(summary["lead_importance"].values()) == pytest.approx(1.0, abs=1e-5)
        assert summary["beat_count"] > 0
        assert set(summary["pqrst_fractions"]) == {"P", "QRS", "T", "baseline"}

    def test_no_plots_writes_only_the_arrays_and_summary(self, temp_dir):
        """The figures are skipped on request."""
        output_dir = temp_dir / "out"

        exit_code = main(
            [
                "--quiet",
                "explain",
                "--checkpoint",
                str(self._checkpoint(temp_dir)),
                "-m",
                "kanres",
                "-i",
                str(self._recording(temp_dir)),
                "-o",
                str(output_dir),
                "--accelerator",
                "cpu",
                "--no-plots",
            ]
        )

        assert exit_code == 0
        assert (output_dir / "attribution.npz").exists()
        assert (output_dir / "summary.json").exists()
        assert not (output_dir / "signal_attribution.png").exists()

    def test_target_class_all_adds_a_class_axis(self, temp_dir):
        """Every class is attributed and stored under its own axis."""
        output_dir = temp_dir / "out"

        exit_code = main(
            [
                "--quiet",
                "explain",
                "--checkpoint",
                str(self._checkpoint(temp_dir)),
                "-m",
                "kanres",
                "-i",
                str(self._recording(temp_dir)),
                "-o",
                str(output_dir),
                "--accelerator",
                "cpu",
                "--no-plots",
                "--target-class",
                "all",
            ]
        )

        assert exit_code == 0
        loaded = np.load(str(output_dir / "attribution.npz"), allow_pickle=False)
        assert loaded["values_by_class"].shape == (4, 1, 1, 3000)
        assert json.loads((output_dir / "summary.json").read_text())["target_class"] == "all"

    def test_explain_defaults_are_forwarded(self):
        """The parser hands the command the documented defaults."""
        with patch("deepecgkit.cli.explain") as mock_explain:
            mock_explain.return_value = 0
            main(["explain", "--checkpoint", "m.ckpt", "-m", "kanres", "-i", "in.npy", "-o", "out"])

        call_kwargs = mock_explain.call_args[1]
        assert call_kwargs["method"] == "saliency"
        assert call_kwargs["max_samples"] == 8
        assert call_kwargs["ig_steps"] == 64
        assert call_kwargs["sampling_rate"] is None
        assert call_kwargs["preprocess"] is True
        assert call_kwargs["plots"] is True
        assert call_kwargs["target_class"] is None


class TestConfigMerge:
    """--config values must actually reach the command (regression).

    The old merge compared against the top-level parser's defaults (always
    None), silently discarding every config key whose CLI default was
    non-None: epochs, batch_size, learning_rate, seed, val_split, ...
    """

    def test_config_overrides_non_none_defaults(self, temp_dir):
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            "train:\n  epochs: 100\n  batch_size: 64\n  learning_rate: 0.01\n  seed: 7\n"
        )
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            result = main(
                ["--config", str(config_file), "train", "-m", "kanres", "-d", "af-classification"]
            )
        assert result == 0
        kwargs = mock_train.call_args[1]
        assert kwargs["epochs"] == 100
        assert kwargs["batch_size"] == 64
        assert kwargs["learning_rate"] == 0.01
        assert kwargs["seed"] == 7

    def test_explicit_flag_beats_config(self, temp_dir):
        config_file = temp_dir / "config.yaml"
        config_file.write_text("train:\n  epochs: 100\n")
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            main(
                [
                    "--config",
                    str(config_file),
                    "train",
                    "-m",
                    "kanres",
                    "-d",
                    "af-classification",
                    "--epochs",
                    "7",
                ]
            )
        assert mock_train.call_args[1]["epochs"] == 7

    def test_unknown_config_key_warns(self, temp_dir, capsys):
        config_file = temp_dir / "config.yaml"
        config_file.write_text("train:\n  epochz: 3\n")
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            main(["--config", str(config_file), "train", "-m", "kanres", "-d", "af-classification"])
        assert "epochz" in capsys.readouterr().err

    def test_config_applies_to_other_commands(self, temp_dir):
        """--config was previously honoured only by train."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("evaluate:\n  batch_size: 16\n  seed: 9\n")
        with patch("deepecgkit.cli.evaluate") as mock_evaluate:
            mock_evaluate.return_value = 0
            main(
                [
                    "--config",
                    str(config_file),
                    "evaluate",
                    "--checkpoint",
                    "model.pt",
                    "-m",
                    "kanres",
                    "-d",
                    "af-classification",
                ]
            )
        kwargs = mock_evaluate.call_args[1]
        assert kwargs["batch_size"] == 16
        assert kwargs["seed"] == 9

    def test_null_config_file_is_handled(self, temp_dir):
        """A YAML file holding just 'null' used to crash with AttributeError."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("null\n")
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            result = main(
                ["--config", str(config_file), "train", "-m", "kanres", "-d", "af-classification"]
            )
        assert result == 0


class TestRunDirContract:
    """run_metadata.json ties `deepecg train` output to the analysis pipeline."""

    def test_write_run_metadata_merges(self, temp_dir):
        from deepecgkit.cli.run_dir import write_run_metadata

        write_run_metadata(temp_dir, {"model": "kanres", "seed": 7})
        write_run_metadata(temp_dir, {"returncode": 0})
        payload = json.loads((temp_dir / "run_metadata.json").read_text())
        assert payload["model"] == "kanres"
        assert payload["seed"] == 7
        assert payload["returncode"] == 0
        assert payload["schema_version"] == 2

    def test_load_run_context_from_metadata(self, temp_dir):
        from deepecgkit.cli.run_dir import load_run_context, write_run_metadata

        run_dir = temp_dir / "run"
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        checkpoint = checkpoints / "epoch=01-val_loss=0.5.pt"
        checkpoint.touch()
        write_run_metadata(run_dir, {"seed": 11, "val_split": 0.25, "test_split": 0.15})

        context = load_run_context(checkpoint)
        assert context.source == "run_metadata"
        assert context.seed == 11
        assert context.val_split == 0.25
        assert context.test_split == 0.15
        assert context.run_dir == run_dir.resolve()

    def test_load_run_context_falls_back_to_split_info(self, temp_dir):
        from deepecgkit.cli.run_dir import load_run_context

        run_dir = temp_dir / "run"
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        checkpoint = checkpoints / "model.pt"
        checkpoint.touch()
        (run_dir / "split_info.json").write_text('{"seed": 3, "n_train": 10}')

        context = load_run_context(checkpoint)
        assert context.source == "split_info"
        assert context.seed == 3

    def test_load_run_context_without_metadata(self, temp_dir):
        from deepecgkit.cli.run_dir import load_run_context

        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()
        assert load_run_context(checkpoint).source == "none"

    def test_verify_split_warns_on_mismatch(self, temp_dir, capsys):
        from deepecgkit.cli.run_dir import verify_split

        (temp_dir / "split_info.json").write_text('{"n_train": 100, "n_val": 20, "n_test": 10}')

        class _Module:
            train_dataset: ClassVar[list[int]] = list(range(90))
            val_dataset: ClassVar[list[int]] = list(range(20))
            test_dataset: ClassVar[list[int]] = list(range(10))

        verify_split(_Module(), temp_dir, CLILogger())
        assert "does not match" in capsys.readouterr().err


class TestAnalysisCommands:
    """deepecg aggregate / rank wrap the analysis pipeline."""

    def test_aggregate_reports_no_runs(self, temp_dir, capsys):
        result = main(["aggregate", "--runs-dir", str(temp_dir)])
        assert result == 1
        assert "No usable runs" in capsys.readouterr().err

    def test_rank_reports_no_runs(self, temp_dir, capsys):
        result = main(["rank", "--runs-dir", str(temp_dir)])
        assert result == 1
        assert "No usable runs" in capsys.readouterr().err

    def test_aggregate_consumes_seeded_run_dirs(self, temp_dir, capsys):
        for seed in (1, 2):
            run_dir = temp_dir / f"kanres-af-classification-seed{seed}"
            run_dir.mkdir()
            (run_dir / "classification_metrics.csv").write_text(
                "model,dataset,accuracy,precision,recall,f1,auroc,mcc,average\n"
                f"kanres,af-classification,0.9,0.8,0.7,0.7{seed},0.95,0.6,macro\n"
            )
        result = main(["aggregate", "--runs-dir", str(temp_dir)])
        assert result == 0
        out = capsys.readouterr().out
        assert "2 runs across 1 dataset(s)" in out
        assert "f1" in out


class TestGlobalOptionPlacement:
    """--verbose, --quiet and --config work on either side of the command."""

    def test_version_is_reported(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert capsys.readouterr().out.startswith("deepecg ")

    def test_config_after_the_command(self, temp_dir):
        config_file = temp_dir / "config.yaml"
        config_file.write_text("train:\n  epochs: 100\n")
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            result = main(
                ["train", "-m", "kanres", "-d", "af-classification", "--config", str(config_file)]
            )
        assert result == 0
        assert mock_train.call_args[1]["epochs"] == 100

    def test_verbosity_flags_after_the_command(self):
        for flag, attribute in (("-v", "verbose"), ("--quiet", "quiet")):
            with patch("deepecgkit.cli.train") as mock_train:
                mock_train.return_value = 0
                main(["train", "-m", "kanres", "-d", "af-classification", flag])
            assert getattr(mock_train.call_args[1]["logger"], attribute) is True

    def test_a_flag_before_the_command_survives_the_subcommand(self):
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            main(["-v", "train", "-m", "kanres", "-d", "af-classification"])
        assert mock_train.call_args[1]["logger"].verbose is True

    def test_config_can_supply_required_arguments(self, temp_dir):
        config_file = temp_dir / "config.yaml"
        config_file.write_text("train:\n  model: resnet\n  dataset: ptbxl\n  epochs: 3\n")
        with patch("deepecgkit.cli.train") as mock_train:
            mock_train.return_value = 0
            result = main(["train", "--config", str(config_file)])
        assert result == 0
        kwargs = mock_train.call_args[1]
        assert (kwargs["model_name"], kwargs["dataset_name"], kwargs["epochs"]) == (
            "resnet",
            "ptbxl",
            3,
        )

    def test_required_arguments_are_still_required_without_a_config(self):
        with pytest.raises(SystemExit):
            main(["train", "-m", "kanres"])

    def test_model_is_optional_where_the_checkpoint_can_supply_it(self, temp_dir):
        checkpoint_file = temp_dir / "model.pt"
        checkpoint_file.touch()
        with patch("deepecgkit.cli.evaluate") as mock_evaluate:
            mock_evaluate.return_value = 0
            main(["evaluate", "--checkpoint", str(checkpoint_file), "-d", "af-classification"])
        assert mock_evaluate.call_args[1]["model_name"] is None

    def test_help_does_not_list_every_model(self, capsys):
        with pytest.raises(SystemExit):
            main(["train", "--help"])
        assert capsys.readouterr().out.count("xresnet1d-benchmark") == 0


@pytest.fixture
def clean_registries():
    """Undo any model or dataset registered while a test ran."""
    from deepecgkit import registry

    # Looked up first so that the built-ins are registered before the copy is taken.
    registry.get_model_names()
    registry.get_dataset_names()
    models, datasets = dict(registry._MODEL_REGISTRY), dict(registry._DATASET_REGISTRY)
    modules = set(sys.modules)
    path = list(sys.path)
    try:
        yield
    finally:
        registry._MODEL_REGISTRY.clear()
        registry._MODEL_REGISTRY.update(models)
        registry._DATASET_REGISTRY.clear()
        registry._DATASET_REGISTRY.update(datasets)
        for name in set(sys.modules) - modules:
            sys.modules.pop(name)
        sys.path[:] = path


PLUGIN_SOURCE = """
import torch
from deepecgkit.models import ECGModelBase
from deepecgkit.registry import register_model


@register_model("plugin-model", description="A model registered from outside the package")
class PluginModel(ECGModelBase):
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.backbone = torch.nn.Conv1d(input_channels, 8, kernel_size=7, padding=3)
        self._feature_dim = 8
        self.classifier = torch.nn.Linear(8, output_size)

    def extract_features(self, x):
        return self.backbone(x).mean(dim=-1)
"""


class TestPluginImport:
    """Models and datasets registered outside the package are usable by name."""

    def test_import_makes_a_registered_model_available(self, temp_dir, capsys, clean_registries):
        plugin = temp_dir / "my_plugin_models.py"
        plugin.write_text(PLUGIN_SOURCE)

        assert main(["--import", str(plugin), "info", "-m", "plugin-model"]) == 0
        assert "A model registered from outside the package" in capsys.readouterr().out

    def test_import_is_accepted_after_the_command(self, temp_dir, capsys, clean_registries):
        plugin = temp_dir / "my_plugin_models.py"
        plugin.write_text(PLUGIN_SOURCE)

        assert main(["info", "-m", "plugin-model", "--import", str(plugin)]) == 0

    def test_environment_variable_names_plugins(
        self, temp_dir, capsys, monkeypatch, clean_registries
    ):
        plugin = temp_dir / "my_plugin_models.py"
        plugin.write_text(PLUGIN_SOURCE)
        monkeypatch.setenv("DEEPECGKIT_PLUGINS", str(plugin))

        assert main(["list-models"]) == 0
        assert "plugin-model" in capsys.readouterr().out

    def test_a_missing_plugin_is_reported(self, capsys):
        assert main(["--import", "no_such_plugin_module_xyz", "list-models"]) == 1
        assert "Failed to import 'no_such_plugin_module_xyz'" in capsys.readouterr().err

    def test_an_unregistered_model_is_still_rejected(self):
        with pytest.raises(SystemExit):
            main(["info", "-m", "plugin-model"])


class TestDatasetOptionValidation:
    """A flag the dataset cannot honour is refused, not recorded and ignored."""

    @pytest.mark.parametrize(
        ("dataset", "flags", "expected"),
        [
            ("af-classification", ["--binary-classification"], "--binary-classification"),
            ("af-classification", ["--multi-label"], "--multi-label"),
            ("ptbxl", ["--segment-duration", "30"], "--segment-duration"),
            ("ptbxl", ["--binary-classification"], "--binary-classification"),
            ("mitbih-afdb", ["--multi-label"], "--multi-label"),
        ],
    )
    def test_unsupported_flag_fails_before_anything_is_loaded(
        self, dataset, flags, expected, temp_dir, capsys, monkeypatch
    ):
        monkeypatch.chdir(temp_dir)
        result = main(["train", "-m", "simple-cnn", "-d", dataset, *flags])

        assert result == 1
        error = capsys.readouterr().err
        assert f"{expected} does not apply to the '{dataset}' dataset" in error
        assert not (temp_dir / "runs").exists()

    def test_every_dataset_accepts_a_sampling_rate(self):
        from deepecgkit.cli._common import dataset_kwargs_for

        for name in DATASET_NAMES:
            assert dataset_kwargs_for(name, sampling_rate=250)["sampling_rate"] == 250

    def test_options_left_unset_keep_the_dataset_defaults(self):
        from deepecgkit.cli._common import dataset_kwargs_for

        kwargs = dataset_kwargs_for("af-classification")
        assert kwargs == {"force_download": False}

    def test_segment_duration_and_normalization_are_forwarded(self):
        from deepecgkit.cli._common import dataset_kwargs_for

        kwargs = dataset_kwargs_for("ltafdb", segment_duration=30.0, normalization="minmax")
        assert kwargs["segment_duration_seconds"] == 30.0
        assert kwargs["normalization"] == "minmax"

    def test_an_inapplicable_option_is_refused_by_the_builder_too(self):
        from deepecgkit.cli._common import dataset_kwargs_for

        with pytest.raises(ValueError, match="--binary-classification does not apply"):
            dataset_kwargs_for("af-classification", binary_classification=True)

    def test_deprecated_devices_flag_says_it_does_nothing(self, temp_dir, capsys, monkeypatch):
        from deepecgkit.datasets import af_classification as af_module

        def offline(*args, **kwargs):
            raise RuntimeError("offline")

        monkeypatch.setattr(af_module, "download_file", offline)
        monkeypatch.chdir(temp_dir)

        main(
            [
                "train",
                "-m",
                "simple-cnn",
                "-d",
                "af-classification",
                "--devices",
                "2",
                "--data-dir",
                str(temp_dir / "absent"),
            ]
        )

        assert "--devices 2 has no effect" in capsys.readouterr().err

    def test_out_of_memory_errors_come_with_what_to_change(self, capsys):
        from deepecgkit.cli._common import report_shape_error

        error = RuntimeError("MPS backend out of memory (MPS allocated: 37.04 GiB)")
        assert report_shape_error(error, "transformer", "Training", CLILogger()) == 1

        message = capsys.readouterr().err
        assert "ran out of memory" in message
        assert "Lower --batch-size" in message


class TestCheckpointResolution:
    """A run directory stands for the best checkpoint inside it."""

    def _run(self, temp_dir, losses):
        folder = temp_dir / "run" / "checkpoints"
        folder.mkdir(parents=True)
        for epoch, loss in enumerate(losses, 1):
            (folder / f"epoch={epoch:02d}-val_loss={loss:.4f}.pt").write_bytes(b"x")
        return temp_dir / "run"

    def test_run_directory_resolves_to_the_lowest_validation_loss(self, temp_dir):
        from deepecgkit.cli.run_dir import resolve_checkpoint

        run = self._run(temp_dir, [0.93, 0.41, 0.57])
        assert resolve_checkpoint(run).name == "epoch=02-val_loss=0.4100.pt"
        assert resolve_checkpoint(run / "checkpoints").name == "epoch=02-val_loss=0.4100.pt"

    def test_a_file_is_returned_unchanged(self, temp_dir):
        from deepecgkit.cli.run_dir import resolve_checkpoint

        run = self._run(temp_dir, [0.93, 0.41])
        chosen = run / "checkpoints" / "epoch=01-val_loss=0.9300.pt"
        assert resolve_checkpoint(chosen) == chosen

    def test_a_directory_without_checkpoints_is_reported(self, temp_dir, capsys):
        result = evaluate(checkpoint=str(temp_dir), dataset_name="af-classification")

        assert result == 1
        assert "No checkpoint files" in capsys.readouterr().err


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """One short CLI training run on synthetic data, shared by the workflow tests."""
    from tests.conftest import write_af_dataset

    root = tmp_path_factory.mktemp("workflow")
    data_dir = write_af_dataset(root / "data")
    run_dir = root / "run"
    result = main(
        [
            "train",
            "-m",
            "simple-cnn",
            "-d",
            "af-classification",
            "--data-dir",
            str(data_dir),
            "--epochs",
            "1",
            "--accelerator",
            "cpu",
            "-o",
            str(run_dir),
            "--quiet",
        ]
    )
    assert result == 0
    return {"run": run_dir, "data": data_dir, "root": root}


class TestTrainedRunWorkflow:
    """Train once, then use the run the way the documentation says to."""

    def _raw_recording(self, trained_run):
        import scipy.io

        path = trained_run["data"] / "training2017" / "A00001.mat"
        return scipy.io.loadmat(str(path))["val"].ravel()

    def test_metadata_records_the_values_that_were_in_effect(self, trained_run):
        metadata = json.loads((trained_run["run"] / "run_metadata.json").read_text())

        assert metadata["schema_version"] == 2
        assert metadata["returncode"] == 0
        assert metadata["sampling_rate"] == 300
        assert metadata["normalization"] == "zscore"
        assert metadata["segment_duration_seconds"] == 10.0
        assert metadata["input_length"] == 3000
        assert metadata["class_names"] == ["Normal", "AF", "Other", "Noisy"]
        assert metadata["best_checkpoint"].endswith(".pt")

    def _train_from(self, trained_run, model, weights, output_name):
        return main(
            [
                "train",
                "-m",
                model,
                "-d",
                "af-classification",
                "--data-dir",
                str(trained_run["data"]),
                "--epochs",
                "1",
                "--accelerator",
                "cpu",
                "--weights",
                str(weights),
                "-o",
                str(trained_run["root"] / output_name),
            ]
        )

    def test_training_can_start_from_an_earlier_checkpoint(self, trained_run, capsys):
        """A checkpoint passed as --weights used to load nothing and be reported as loaded."""
        result = self._train_from(trained_run, "simple-cnn", trained_run["run"], "from-weights")
        streams = capsys.readouterr()
        output = streams.out + streams.err

        assert result == 0
        assert re.search(r"Loaded (\d+) of the model's \1 weight tensors", output)
        assert "Left as initialised" not in output

    def test_weights_of_another_architecture_are_refused(self, trained_run, capsys):
        from deepecgkit.cli.run_dir import resolve_checkpoint

        checkpoint = resolve_checkpoint(trained_run["run"])
        result = self._train_from(trained_run, "kanres", checkpoint, "from-foreign-weights")

        assert result == 1
        assert "belong to a different architecture" in capsys.readouterr().err

    def test_predictions_name_the_recording_of_each_sample(self, trained_run):
        """Without it, predictions cannot be grouped by recording afterwards."""
        lines = (trained_run["run"] / "predictions.txt").read_text().splitlines()
        header, rows = lines[0].split("\t"), [line.split("\t") for line in lines[1:]]
        split_info = json.loads((trained_run["run"] / "split_info.json").read_text())

        assert header[-1] == "record"
        assert all(len(row) == len(header) for row in rows)
        assert all(row[-1].startswith("A0") for row in rows)
        assert len({row[-1] for row in rows}) == split_info["n_test_groups"]

    def test_checkpoint_carries_its_architecture_and_preprocessing(self, trained_run):
        from deepecgkit.cli.run_dir import resolve_checkpoint

        checkpoint = torch.load(resolve_checkpoint(trained_run["run"]), weights_only=False)

        assert checkpoint["model_config"]["name"] == "simple-cnn"
        assert checkpoint["data_config"]["dataset"] == "af-classification"
        assert checkpoint["data_config"]["preprocessing"]["sampling_rate"] == 300

    def test_evaluate_needs_only_the_run_directory(self, trained_run, capsys):
        result = main(
            [
                "evaluate",
                "--checkpoint",
                str(trained_run["run"]),
                "-d",
                "af-classification",
                "--data-dir",
                str(trained_run["data"]),
                "--accelerator",
                "cpu",
            ]
        )

        output = capsys.readouterr().out
        assert result == 0
        assert "Macro F1" in output
        assert "AUROC" in output
        assert "Per-Class Results" in output

    def test_evaluate_reports_a_contradicting_model_in_one_line(self, trained_run, capsys):
        result = main(
            [
                "evaluate",
                "--checkpoint",
                str(trained_run["run"]),
                "-d",
                "af-classification",
                "-m",
                "kanres",
                "--data-dir",
                str(trained_run["data"]),
                "--accelerator",
                "cpu",
            ]
        )

        error = capsys.readouterr().err
        assert result == 1
        assert "trained as 'simple-cnn', but --model kanres was given" in error
        assert "state_dict" not in error

    def test_predict_preprocesses_a_raw_recording(self, trained_run, temp_dir):
        recording = temp_dir / "raw.npy"
        np.save(str(recording), self._raw_recording(trained_run))
        output = temp_dir / "prediction.json"

        result = main(
            [
                "predict",
                "--checkpoint",
                str(trained_run["run"]),
                "-i",
                str(recording),
                "--sampling-rate",
                "300",
                "-o",
                str(output),
                "--accelerator",
                "cpu",
            ]
        )

        assert result == 0
        payload = json.loads(output.read_text())
        assert payload["model"] == "simple-cnn"
        assert payload["class_names"] == ["Normal", "AF", "Other", "Noisy"]
        assert payload["labels"][0] in payload["class_names"]
        assert len(payload["predictions"]) == 1
        assert len(payload["segments"]) == 3
        assert [segment["start_seconds"] for segment in payload["segments"]] == [0.0, 10.0, 20.0]
        assert max(payload["probabilities"][0]) < 0.999

    def test_prediction_does_not_depend_on_the_recording_scale(self, trained_run, temp_dir):
        raw = self._raw_recording(trained_run)
        outputs = []
        for name, scale in (("millivolts.npy", 0.001), ("counts.npy", 1000.0)):
            recording, output = temp_dir / name, temp_dir / f"{name}.json"
            np.save(str(recording), raw * scale)
            main(
                [
                    "predict",
                    "--checkpoint",
                    str(trained_run["run"]),
                    "-i",
                    str(recording),
                    "-o",
                    str(output),
                    "--accelerator",
                    "cpu",
                    "--quiet",
                ]
            )
            outputs.append(json.loads(output.read_text())["probabilities"][0])

        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-4)

    def test_predict_resamples_a_recording_at_another_rate(self, trained_run, temp_dir, capsys):
        recording = temp_dir / "half_rate.npy"
        np.save(str(recording), self._raw_recording(trained_run)[::2])
        output = temp_dir / "prediction.json"

        result = main(
            [
                "predict",
                "--checkpoint",
                str(trained_run["run"]),
                "-i",
                str(recording),
                "--sampling-rate",
                "150",
                "-o",
                str(output),
                "--accelerator",
                "cpu",
            ]
        )

        assert result == 0
        assert "resampled from 150 Hz to 300 Hz" in capsys.readouterr().out
        assert len(json.loads(output.read_text())["segments"]) == 3

    def test_predict_names_the_class_on_the_console(self, trained_run, temp_dir, capsys):
        recording = temp_dir / "raw.npy"
        np.save(str(recording), self._raw_recording(trained_run))

        main(
            [
                "predict",
                "--checkpoint",
                str(trained_run["run"]),
                "-i",
                str(recording),
                "--accelerator",
                "cpu",
            ]
        )

        output = capsys.readouterr().out
        assert "Recording 0: " in output
        assert "averaged over 3 segments" in output
        assert "Classes: Normal, AF, Other, Noisy" in output

    def test_extract_features_gives_one_row_per_recording(self, trained_run, temp_dir):
        recording = temp_dir / "raw.npy"
        np.save(str(recording), self._raw_recording(trained_run))
        output = temp_dir / "features.npz"

        result = main(
            [
                "extract-features",
                "--checkpoint",
                str(trained_run["run"]),
                "-i",
                str(recording),
                "-o",
                str(output),
                "--accelerator",
                "cpu",
                "--quiet",
            ]
        )

        assert result == 0
        assert np.load(str(output))["features"].shape[0] == 1

    def test_explain_places_beats_at_the_training_rate(self, trained_run, temp_dir):
        recording = temp_dir / "raw.npy"
        np.save(str(recording), self._raw_recording(trained_run))
        output = temp_dir / "explanation"

        result = main(
            [
                "explain",
                "--checkpoint",
                str(trained_run["run"]),
                "-i",
                str(recording),
                "-o",
                str(output),
                "--accelerator",
                "cpu",
                "--no-plots",
                "--quiet",
            ]
        )

        assert result == 0
        summary = json.loads((output / "summary.json").read_text())
        assert summary["sampling_rate"] == 300.0
        assert summary["timesteps"] == 3000

    def test_a_requested_sampling_rate_is_applied_and_recorded(self, trained_run):
        run_dir = trained_run["root"] / "run-150"
        result = main(
            [
                "train",
                "-m",
                "simple-cnn",
                "-d",
                "af-classification",
                "--sampling-rate",
                "150",
                "--data-dir",
                str(trained_run["data"]),
                "--epochs",
                "1",
                "--accelerator",
                "cpu",
                "-o",
                str(run_dir),
                "--quiet",
            ]
        )

        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        assert result == 0
        assert metadata["sampling_rate"] == 150
        assert metadata["input_length"] == 1500

    def test_a_model_that_cannot_take_the_segments_fails_early_with_a_fix(
        self, trained_run, capsys
    ):
        run_dir = trained_run["root"] / "run-afmodel"
        result = main(
            [
                "train",
                "-m",
                "afmodel",
                "-d",
                "af-classification",
                "-o",
                str(run_dir),
                "--data-dir",
                str(trained_run["data"]),
                "--epochs",
                "1",
                "--accelerator",
                "cpu",
            ]
        )

        error = capsys.readouterr().err
        assert result == 1
        assert "'afmodel' needs at least 7971 samples per lead" in error
        assert "3000 samples per lead (10 s at 300 Hz)" in error
        assert "--segment-duration 27" in error
        assert not run_dir.exists()

    def test_a_missing_dataset_leaves_no_run_directory(self, trained_run, capsys, monkeypatch):
        from deepecgkit.datasets import af_classification as af_module

        def offline(*args, **kwargs):
            raise RuntimeError("Failed to download: network unreachable")

        monkeypatch.setattr(af_module, "download_file", offline)
        run_dir = trained_run["root"] / "run-missing"
        result = main(
            [
                "train",
                "-m",
                "simple-cnn",
                "-d",
                "af-classification",
                "-o",
                str(run_dir),
                "--data-dir",
                str(trained_run["root"] / "no-data"),
                "--accelerator",
                "cpu",
            ]
        )

        assert result == 1
        assert "network unreachable" in capsys.readouterr().err
        assert not run_dir.exists()

    def test_legacy_checkpoint_is_evaluated_at_the_legacy_rate(self, trained_run, capsys):
        """Checkpoints from before preprocessing was recorded were trained at 500 Hz."""
        from deepecgkit.cli._common import checkpoint_dataset_kwargs
        from deepecgkit.cli.run_dir import resolve_checkpoint

        checkpoint = torch.load(resolve_checkpoint(trained_run["run"]), weights_only=False)
        checkpoint.pop("data_config")

        kwargs, _ = checkpoint_dataset_kwargs("af-classification", checkpoint, CLILogger())

        assert kwargs["sampling_rate"] == 500
        assert "does not record its preprocessing" in capsys.readouterr().err

    def test_recorded_preprocessing_rebuilds_the_dataset(self, trained_run):
        from deepecgkit.cli._common import checkpoint_dataset_kwargs
        from deepecgkit.cli.run_dir import resolve_checkpoint

        checkpoint = torch.load(resolve_checkpoint(trained_run["run"]), weights_only=False)
        kwargs, multi_label = checkpoint_dataset_kwargs(
            "af-classification", checkpoint, CLILogger()
        )

        assert multi_label is False
        assert kwargs["sampling_rate"] == 300
        assert kwargs["normalization"] == "zscore"
        assert kwargs["segment_duration_seconds"] == 10.0

    def test_probe_warns_when_the_encoder_was_trained_at_another_rate(self, trained_run, capsys):
        from deepecgkit.cli.commands.probe import _warn_on_rate_mismatch
        from deepecgkit.cli.run_dir import resolve_checkpoint

        source = {"af-classification": str(resolve_checkpoint(trained_run["run"]))}

        _warn_on_rate_mismatch(source, 300, CLILogger())
        assert capsys.readouterr().err == ""

        _warn_on_rate_mismatch(source, 500, CLILogger())
        assert "trained at 300 Hz but is probed at 500 Hz" in capsys.readouterr().err


class TestListDatasetsDetail:
    """The listing says what a download involves before it starts."""

    def test_listing_shows_signal_shape_size_and_location(self, capsys, monkeypatch, temp_dir):
        monkeypatch.setenv("DEEPECGKIT_HOME", str(temp_dir))
        list_datasets()
        output = capsys.readouterr().out

        assert "Signal: 1 lead(s), 300 Hz, 10 s segments" in output
        assert "Classes: 4 (Normal, AF, Other, Noisy)" in output
        assert "not downloaded (about 100 MB)" in output
        assert str(temp_dir / "datasets" / "afclassificationdataset") in output
        assert "Options: --binary-classification, --segment-duration" in output
