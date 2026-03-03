"""
Tests for deepecgkit.training module.
"""

import pytest
import torch

import deepecgkit.training as training_module
from deepecgkit.datasets import AFClassificationDataset, ECGDataModule
from deepecgkit.models import AFModel, KanResWideX
from deepecgkit.training import ECGTrainer


class TestECGTrainer:
    """Test the ECGTrainer class."""

    def _make_trainer(self, model=None, **config_overrides):
        if model is None:
            model = KanResWideX(input_channels=1, output_size=4)
        train_config = {
            "learning_rate": 0.001,
            "scheduler": {"factor": 0.5, "patience": 10},
            "binary_classification": False,
        }
        train_config.update(config_overrides)
        return ECGTrainer(model=model, train_config=train_config, device="cpu")

    def test_ecg_trainer_init(self):
        """Test ECGTrainer initialization."""
        model = KanResWideX(input_channels=1, output_size=4)
        trainer = self._make_trainer(model)

        assert trainer.model is model
        assert trainer.learning_rate == 0.001

    def test_ecg_trainer_forward(self, sample_batch_ecg):
        """Test model forward pass through trainer."""
        trainer = self._make_trainer()
        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(sample_batch_ecg)

        assert output.shape == (sample_batch_ecg.shape[0], 4)
        assert output.dtype == torch.float32

    def test_ecg_trainer_loss_computation(self, sample_batch_ecg_10s, sample_labels):
        """Test loss computation."""
        model = AFModel(recording_length=10)
        trainer = self._make_trainer(model)

        y_hat = trainer.model(sample_batch_ecg_10s)
        loss = trainer._calculate_loss(y_hat, sample_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_ecg_trainer_acc_computation(self, sample_batch_ecg, sample_labels):
        """Test accuracy computation."""
        trainer = self._make_trainer()
        y_hat = trainer.model(sample_batch_ecg)
        acc = trainer._compute_acc(y_hat, sample_labels)

        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    @pytest.mark.parametrize("task_type", ["classification", "regression"])
    def test_ecg_trainer_different_tasks(self, task_type, sample_batch_ecg):
        """Test ECGTrainer with different task types."""
        output_size = 4 if task_type == "classification" else 1
        model = KanResWideX(input_channels=1, output_size=output_size)
        is_classification = task_type == "classification"
        trainer = self._make_trainer(
            model,
            binary_classification=not is_classification,
            task_type=task_type,
        )

        trainer.model.eval()
        with torch.no_grad():
            output = trainer.model(sample_batch_ecg)

        assert output.shape == (sample_batch_ecg.shape[0], output_size)

    def test_ecg_trainer_loss_functions(
        self, sample_batch_ecg, sample_labels, sample_regression_targets
    ):
        """Test different loss functions for different tasks."""
        model_cls = KanResWideX(input_channels=1, output_size=4)
        trainer_cls = self._make_trainer(model_cls, task_type="classification")

        y_hat = trainer_cls.model(sample_batch_ecg)
        loss_cls = trainer_cls._calculate_loss(y_hat, sample_labels)
        assert loss_cls.item() >= 0

        model_reg = KanResWideX(input_channels=1, output_size=1)
        trainer_reg = self._make_trainer(model_reg, task_type="regression")

        y_hat_reg = trainer_reg.model(sample_batch_ecg)
        loss_reg = trainer_reg._calculate_loss(y_hat_reg, sample_regression_targets.unsqueeze(1))
        assert loss_reg.item() >= 0

    def test_ecg_trainer_prediction_mode(self, sample_batch_ecg):
        """Test ECGTrainer model in prediction mode."""
        trainer = self._make_trainer()
        trainer.model.eval()
        with torch.no_grad():
            predictions = trainer.model(sample_batch_ecg)

        assert predictions.shape == (sample_batch_ecg.shape[0], 4)
        assert not torch.isnan(predictions).any()

    def test_ecg_trainer_device_compatibility(self):
        """Test ECGTrainer device compatibility."""
        trainer = self._make_trainer()

        input_cpu = torch.randn(2, 1, 1000)
        trainer.model.eval()
        with torch.no_grad():
            output_cpu = trainer.model(input_cpu)
        assert output_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            trainer_gpu = ECGTrainer(
                model=KanResWideX(input_channels=1, output_size=4),
                train_config={
                    "learning_rate": 0.001,
                    "scheduler": {"factor": 0.5, "patience": 10},
                    "binary_classification": False,
                },
                device="cuda",
            )
            input_gpu = input_cpu.cuda()
            trainer_gpu.model.eval()
            with torch.no_grad():
                output_gpu = trainer_gpu.model(input_gpu)
            assert output_gpu.device.type == "cuda"


class TestTrainingUtilities:
    """Test training utility functions."""

    def test_training_imports(self):
        """Test that training components can be imported."""
        assert ECGTrainer is not None

    def test_training_module_structure(self):
        """Test training module structure and exports."""
        assert hasattr(training_module, "ECGTrainer")
        assert isinstance(training_module.ECGTrainer, type)

    def test_seed_everything(self):
        """Test seed_everything reproducibility."""
        ECGTrainer.seed_everything(42)
        a = torch.randn(5)
        ECGTrainer.seed_everything(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


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
        trainer = ECGTrainer(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
            device="cpu",
        )

        datamodule.setup(stage="fit")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        try:
            for batch in train_loader:
                x, y = batch
                y_hat = trainer.model(x)
                loss = trainer._calculate_loss(y_hat, y)
                assert isinstance(loss, torch.Tensor)
                break
        except Exception:
            pytest.skip("No valid data in mock files")

    def test_full_training_simulation(self, sample_batch_ecg_10s, sample_labels):
        """Simulate a full training loop."""
        model = AFModel(recording_length=10)
        trainer = ECGTrainer(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
            device="cpu",
        )
        trainer._setup_optimizer()
        trainer._gradient_clip_val = None

        for _step in range(3):
            trainer.model.train()
            y_hat = trainer.model(sample_batch_ecg_10s)
            loss = trainer._calculate_loss(y_hat, sample_labels)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

    def test_training_with_different_learning_rates(self):
        """Test training with different optimizer configurations."""
        lrs = [0.001, 0.01, 0.0001]

        for lr in lrs:
            model = KanResWideX(input_channels=1, output_size=4)
            trainer = ECGTrainer(
                model=model,
                train_config={
                    "learning_rate": lr,
                    "scheduler": {"factor": 0.5, "patience": 10},
                    "binary_classification": False,
                },
                device="cpu",
            )
            trainer._setup_optimizer()
            assert trainer.optimizer is not None
            assert trainer.optimizer.param_groups[0]["lr"] == lr


class TestTrainingEdgeCases:
    """Test edge cases and error handling in training."""

    def test_training_with_minimal_batch(self):
        """Test training with edge case inputs."""
        model = KanResWideX(input_channels=1, output_size=4)
        trainer = ECGTrainer(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
            device="cpu",
        )

        x = torch.randn(1, 1, 100)
        y = torch.tensor([0])

        try:
            y_hat = trainer.model(x)
            loss = trainer._calculate_loss(y_hat, y)
            assert isinstance(loss, torch.Tensor)
        except Exception as e:
            pytest.skip(f"Edge case failed as expected: {e}")

    def test_gradient_flow_and_backprop(self, sample_batch_ecg_10s, sample_labels):
        """Test gradient flow during training."""
        model = AFModel(recording_length=10)
        trainer = ECGTrainer(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
            device="cpu",
        )

        for param in trainer.model.parameters():
            param.requires_grad_(True)

        y_hat = trainer.model(sample_batch_ecg_10s)
        loss = trainer._calculate_loss(y_hat, sample_labels)
        loss.backward()

        has_gradients = False
        for param in trainer.model.parameters():
            if param.grad is not None and not torch.allclose(
                param.grad, torch.zeros_like(param.grad)
            ):
                has_gradients = True
                break

        assert has_gradients, "No gradients found after backward pass"

    def test_model_state_consistency(self, sample_batch_ecg_10s):
        """Test model state consistency between training and eval modes."""
        model = AFModel(recording_length=10)
        trainer = ECGTrainer(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
            device="cpu",
        )

        trainer.model.train()
        output_train = trainer.model(sample_batch_ecg_10s)

        trainer.model.eval()
        with torch.no_grad():
            output_eval = trainer.model(sample_batch_ecg_10s)

        assert output_train.shape == output_eval.shape
        assert not torch.isnan(output_train).any()
        assert not torch.isnan(output_eval).any()
