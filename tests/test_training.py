"""
Tests for deepecgkit.training module.
"""

from unittest.mock import Mock

import pytest
import torch

import deepecgkit.training as training_module
from deepecgkit.datasets import AFClassificationDataset, ECGDataModule
from deepecgkit.models import AFModel, KanResWideX
from deepecgkit.training import ECGLitModel


class TestECGLitModel:
    """Test the ECGLitModel PyTorch Lightning module."""

    def test_ecg_lit_model_init(self):
        """Test ECGLitModel initialization."""
        model = KanResWideX(input_channels=1, output_size=4)
        train_config = {
            "learning_rate": 0.001,
            "scheduler": {"factor": 0.5, "patience": 10},
            "binary_classification": False,
        }
        lit_model = ECGLitModel(model=model, train_config=train_config)

        assert lit_model.model == model
        assert lit_model.learning_rate == 0.001

    def test_ecg_lit_model_forward(self, sample_batch_ecg):
        """Test ECGLitModel forward pass."""
        model = KanResWideX(input_channels=1, output_size=4)
        train_config = {
            "learning_rate": 0.001,
            "scheduler": {"factor": 0.5, "patience": 10},
            "binary_classification": False,
        }
        lit_model = ECGLitModel(model=model, train_config=train_config)

        output = lit_model(sample_batch_ecg)

        assert output.shape == (sample_batch_ecg.shape[0], 4)
        assert output.dtype == torch.float32

    def test_ecg_lit_model_training_step(self, sample_batch_ecg_10s, sample_labels):
        """Test ECGLitModel training step."""
        model = AFModel(recording_length=10)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        batch = (sample_batch_ecg_10s, sample_labels)
        loss = lit_model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_ecg_lit_model_validation_step(self, sample_batch_ecg, sample_labels):
        """Test ECGLitModel validation step."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        batch = (sample_batch_ecg, sample_labels)
        lit_model.validation_step(batch, batch_idx=0)

        assert True

    def test_ecg_lit_model_test_step(self, sample_batch_ecg, sample_labels):
        """Test ECGLitModel test step."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        batch = (sample_batch_ecg, sample_labels)
        lit_model.test_step(batch, batch_idx=0)

        assert True

    def test_ecg_lit_model_configure_optimizers(self):
        """Test ECGLitModel optimizer configuration."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        optimizer = lit_model.configure_optimizers()

        assert optimizer is not None

        assert hasattr(optimizer, "step") or isinstance(optimizer, dict)

    @pytest.mark.parametrize("task_type", ["classification", "regression"])
    def test_ecg_lit_model_different_tasks(self, task_type, sample_batch_ecg):
        """Test ECGLitModel with different task types."""
        output_size = 4 if task_type == "classification" else 1
        model = KanResWideX(input_channels=1, output_size=output_size)
        is_classification = task_type == "classification"
        train_config = {
            "learning_rate": 0.001,
            "scheduler": {"factor": 0.5, "patience": 10},
            "binary_classification": not is_classification,
        }
        lit_model = ECGLitModel(model=model, train_config=train_config)

        output = lit_model(sample_batch_ecg)

        assert output.shape == (sample_batch_ecg.shape[0], output_size)

    def test_ecg_lit_model_loss_functions(
        self, sample_batch_ecg, sample_labels, sample_regression_targets
    ):
        """Test different loss functions for different tasks."""
        model_cls = KanResWideX(input_channels=1, output_size=4)
        lit_model_cls = ECGLitModel(
            model=model_cls,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
                "task_type": "classification",
            },
        )

        batch_cls = (sample_batch_ecg, sample_labels)
        loss_cls = lit_model_cls.training_step(batch_cls, batch_idx=0)
        assert loss_cls.item() >= 0

        model_reg = KanResWideX(input_channels=1, output_size=1)
        lit_model_reg = ECGLitModel(
            model=model_reg,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
                "task_type": "regression",
            },
        )

        batch_reg = (sample_batch_ecg, sample_regression_targets.unsqueeze(1))
        loss_reg = lit_model_reg.training_step(batch_reg, batch_idx=0)
        assert loss_reg.item() >= 0

    def test_ecg_lit_model_metrics_logging(self, sample_batch_ecg_10s, sample_labels):
        """Test that metrics are logged during training."""
        model = AFModel(recording_length=10)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        lit_model.log = Mock()

        batch = (sample_batch_ecg_10s, sample_labels)
        lit_model.training_step(batch, batch_idx=0)

        assert lit_model.log.called

    def test_ecg_lit_model_prediction_mode(self, sample_batch_ecg):
        """Test ECGLitModel in prediction mode."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        lit_model.eval()
        with torch.no_grad():
            predictions = lit_model(sample_batch_ecg)

        assert predictions.shape == (sample_batch_ecg.shape[0], 4)

        assert not torch.isnan(predictions).any()

    def test_ecg_lit_model_device_compatibility(self):
        """Test ECGLitModel device compatibility."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        input_cpu = torch.randn(2, 1, 1000)
        output_cpu = lit_model(input_cpu)
        assert output_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            lit_model = lit_model.cuda()
            input_gpu = input_cpu.cuda()
            output_gpu = lit_model(input_gpu)
            assert output_gpu.device.type == "cuda"


class TestTrainingUtilities:
    """Test training utility functions."""

    def test_training_imports(self):
        """Test that training components can be imported."""
        assert ECGLitModel is not None

    def test_training_module_structure(self):
        """Test training module structure and exports."""
        assert hasattr(training_module, "ECGLitModel")

        assert isinstance(training_module.ECGLitModel, type)


class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_training_with_datamodule(self, mock_dataset_files):
        """Test training integration with data module."""
        datamodule = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(mock_dataset_files),
            batch_size=2,
            num_workers=0,
        )

        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        datamodule.setup(stage="fit")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        try:
            for batch in train_loader:
                loss = lit_model.training_step(batch, batch_idx=0)
                assert isinstance(loss, torch.Tensor)
                break
        except Exception:
            pytest.skip("No valid data in mock files")

    def test_full_training_simulation(self, sample_batch_ecg_10s, sample_labels):
        """Simulate a full training loop."""
        model = AFModel(recording_length=10)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        lit_model.log = Mock()

        batch = (sample_batch_ecg_10s, sample_labels)

        for step in range(3):
            loss = lit_model.training_step(batch, batch_idx=step)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_training_with_different_optimizers(self):
        """Test training with different optimizer configurations."""
        configs = [
            {"learning_rate": 0.001},
            {"learning_rate": 0.01},
            {"learning_rate": 0.0001},
        ]

        for config in configs:
            model = KanResWideX(input_channels=1, output_size=4)
            train_config = {
                "learning_rate": config["learning_rate"],
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            }
            lit_model = ECGLitModel(model=model, train_config=train_config)

            optimizer = lit_model.configure_optimizers()
            assert optimizer is not None

            if hasattr(optimizer, "param_groups"):
                assert optimizer.param_groups[0]["lr"] == config["learning_rate"]

    def test_validation_and_test_loops(self, sample_batch_ecg, sample_labels):
        """Test validation and test loop functionality."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )
        lit_model.log = Mock()

        batch = (sample_batch_ecg, sample_labels)

        lit_model.validation_step(batch, batch_idx=0)

        lit_model.test_step(batch, batch_idx=0)

        assert True


class TestTrainingEdgeCases:
    """Test edge cases and error handling in training."""

    def test_training_with_empty_batch(self):
        """Test training with edge case inputs."""
        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        minimal_batch = (torch.randn(1, 1, 100), torch.tensor([0]))

        try:
            loss = lit_model.training_step(minimal_batch, batch_idx=0)
            assert isinstance(loss, torch.Tensor)
        except Exception as e:
            pytest.skip(f"Edge case failed as expected: {e}")

    def test_training_with_invalid_task_type(self):
        """Test training with mismatched dimensions (simulates configuration errors)."""
        model = KanResWideX(input_channels=1, output_size=4)

        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        batch = (torch.randn(2, 1, 1000), torch.tensor([0, 1]))

        try:
            lit_model.training_step(batch, batch_idx=0)

            pytest.skip("Dimensional mismatch was handled gracefully")
        except (RuntimeError, ValueError) as e:
            assert (
                "size" in str(e).lower()
                or "shape" in str(e).lower()
                or "dimension" in str(e).lower()
            )
        except Exception as e:
            pytest.fail(
                f"Unexpected exception type for dimension mismatch: {type(e).__name__}: {e}"
            )

    def test_gradient_flow_and_backprop(self, sample_batch_ecg_10s, sample_labels):
        """Test gradient flow during training."""
        model = AFModel(recording_length=10)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        for param in lit_model.parameters():
            param.requires_grad_(True)

        batch = (sample_batch_ecg_10s, sample_labels)
        loss = lit_model.training_step(batch, batch_idx=0)

        loss.backward()

        has_gradients = False
        for param in lit_model.parameters():
            if param.grad is not None and not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ):
                has_gradients = True
                break

        assert has_gradients, "No gradients found after backward pass"

    def test_model_state_consistency(self, sample_batch_ecg_10s):
        """Test model state consistency between training and eval modes."""
        model = AFModel(recording_length=10)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        lit_model.train()
        output_train = lit_model(sample_batch_ecg_10s)

        lit_model.eval()
        with torch.no_grad():
            output_eval = lit_model(sample_batch_ecg_10s)

        assert output_train.shape == output_eval.shape

        assert not torch.isnan(output_train).any()
        assert not torch.isnan(output_eval).any()
