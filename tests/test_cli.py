"""
Tests for deepecgkit.cli module.
"""

import json
import sys
from unittest.mock import MagicMock, patch

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
    MODEL_WEIGHTS,
    CLILogger,
    _get_dataset_registry,
    _get_model_registry,
    evaluate,
    explain,
    list_datasets,
    list_models,
    load_config,
    main,
    predict,
    probe,
    resume,
    show_info,
    train,
)
from deepecgkit.cli.commands.evaluate import (
    _dataset_kwargs_for_checkpoint,
    _read_train_config,
)
from deepecgkit.cli.commands.probe import parse_source_checkpoints
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
        pytest.importorskip("yaml")
        config_file = temp_dir / "config.yaml"
        config_file.write_text("verbose: true\ntrain:\n  epochs: 100\n")

        loaded = load_config(str(config_file))
        assert loaded["verbose"] is True
        assert loaded["train"]["epochs"] == 100

    def test_load_yml_config(self, temp_dir):
        """Test loading .yml configuration."""
        pytest.importorskip("yaml")
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

    def test_list_models_includes_pretrained_weights(self, capsys):
        """Test list_models includes pretrained weights info."""
        list_models()
        captured = capsys.readouterr()

        for _, weights in MODEL_WEIGHTS.items():
            if weights:
                assert "Pretrained weights:" in captured.out


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


class TestResumeCommand:
    """Test the resume command."""

    def test_resume_missing_checkpoint(self):
        """Test resume command with missing checkpoint."""
        with pytest.raises(SystemExit):
            main(["resume"])

    def test_resume_checkpoint_not_found(self, capsys):
        """Test resume command with non-existent checkpoint."""
        result = main(["resume", "--checkpoint", "/nonexistent/model.ckpt", "-m", "kanres"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err


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

        mock_model = MagicMock()
        mock_model_class = MagicMock(return_value=mock_model)

        with patch.object(predict_module, "torch") as mock_torch, patch.object(
            predict_module, "_get_model_registry"
        ) as mock_registry:
            mock_torch.load.return_value = {"model_state_dict": {}}
            mock_registry.return_value = {"kanres": mock_model_class}

            result = predict(
                checkpoint=str(checkpoint_file),
                input_path=str(input_file),
                model_name="kanres",
            )

        assert result == 1
        captured = capsys.readouterr()
        assert "Unsupported input format" in captured.err


class TestResumeFunction:
    """Test the resume function directly."""

    def test_resume_checkpoint_not_found(self, capsys):
        """Test resume with non-existent checkpoint."""
        result = resume(checkpoint="/nonexistent/model.ckpt", model_name="kanres")

        assert result == 1
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err


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
            assert call_kwargs["num_workers"] == 4
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["devices"] == 1
            assert call_kwargs["early_stopping_patience"] == 10
            assert call_kwargs["seed"] == 42

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
            assert call_kwargs["num_workers"] == 4
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
            assert call_kwargs["num_workers"] == 4
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["devices"] == 1
            assert call_kwargs["split"] == "test"

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
            assert call_kwargs["batch_size"] == 1
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["output_path"] is None

    def test_resume_default_values(self, temp_dir):
        """Test resume command default argument values."""
        checkpoint_file = temp_dir / "model.ckpt"
        checkpoint_file.touch()

        with patch("deepecgkit.cli.resume") as mock_resume:
            mock_resume.return_value = 0
            main(["resume", "--checkpoint", str(checkpoint_file), "-m", "kanres"])

            call_kwargs = mock_resume.call_args[1]
            assert call_kwargs["epochs"] is None
            assert call_kwargs["output_dir"] is None
            assert call_kwargs["accelerator"] == "auto"
            assert call_kwargs["devices"] == 1
            assert call_kwargs["early_stopping_patience"] == 10


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
        assert call_kwargs["sampling_rate"] == 500.0
        assert call_kwargs["plots"] is True
        assert call_kwargs["target_class"] is None
