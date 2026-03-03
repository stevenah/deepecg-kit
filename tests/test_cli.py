"""
Tests for deepecgkit.cli module.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
    list_datasets,
    list_models,
    load_config,
    main,
    predict,
    resume,
    show_info,
    train,
)

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
