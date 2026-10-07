"""
Tests for deepecgkit.training module.
"""

import pytest
import torch

import deepecgkit.training as training_module
from deepecgkit.datasets import AFClassificationDataset, ECGDataModule
from deepecgkit.models import AFModel, KanResWideX, SimpleCNN
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


class TestTrainerLogitCapture:
    """Test that raw logits survive an evaluation pass."""

    def _loader(self, signals, labels):
        dataset = torch.utils.data.TensorDataset(signals, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def test_no_logits_before_any_evaluation(self):
        """A trainer that has evaluated nothing has nothing to hand back."""
        trainer = TestECGTrainer()._make_trainer()

        assert trainer.get_test_logits() is None

    def test_logits_are_captured_alongside_probabilities(self, sample_batch_ecg_10s, sample_labels):
        """Temperature scaling needs the logits, not the probabilities they became."""
        trainer = TestECGTrainer()._make_trainer()
        loader = self._loader(sample_batch_ecg_10s, sample_labels)

        trainer._evaluate_loader(loader)
        logits = trainer.get_test_logits()
        _, _, probabilities = trainer.get_test_results()

        assert logits is not None
        assert logits.shape == (len(sample_labels), 4)
        assert logits.shape == probabilities.shape

    def test_captured_logits_reproduce_the_stored_probabilities(
        self, sample_batch_ecg_10s, sample_labels
    ):
        """The captured logits must be the ones the probabilities came from."""
        trainer = TestECGTrainer()._make_trainer()
        loader = self._loader(sample_batch_ecg_10s, sample_labels)

        trainer._evaluate_loader(loader)
        logits = torch.from_numpy(trainer.get_test_logits())
        _, _, probabilities = trainer.get_test_results()

        assert torch.allclose(
            torch.softmax(logits, dim=1), torch.from_numpy(probabilities), atol=1e-6
        )

    def test_a_second_pass_replaces_the_first(self, sample_batch_ecg_10s, sample_labels):
        """Buffers are reused, so results must be copied out between splits."""
        trainer = TestECGTrainer()._make_trainer()
        full = self._loader(sample_batch_ecg_10s, sample_labels)
        partial = self._loader(sample_batch_ecg_10s[:2], sample_labels[:2])

        trainer._evaluate_loader(full)
        trainer._evaluate_loader(partial)

        assert trainer.get_test_logits().shape[0] == 2


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

        saw_batch = False
        for batch in train_loader:
            x, y = batch
            y_hat = trainer.model(x)
            loss = trainer._calculate_loss(y_hat, y)
            assert isinstance(loss, torch.Tensor)
            saw_batch = True
            break
        assert saw_batch, "train_loader yielded no batches from the mock dataset"

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

        y_hat = trainer.model(x)
        loss = trainer._calculate_loss(y_hat, y)
        assert isinstance(loss, torch.Tensor)
        assert y_hat.shape == (1, 4)

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


class TestTrainerDefaults:
    """The trainer works without a hand-built configuration dictionary."""

    def test_no_config_is_needed(self):
        trainer = ECGTrainer(SimpleCNN(input_channels=1, output_size=4), device="cpu")

        assert trainer.learning_rate == 1e-3
        assert trainer.scheduler_factor == 0.5
        assert trainer.scheduler_patience == 5
        assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)

    def test_a_partial_config_is_completed(self):
        trainer = ECGTrainer(
            SimpleCNN(input_channels=1, output_size=4),
            {"learning_rate": 0.01, "scheduler": {"patience": 2}},
            device="cpu",
        )

        assert trainer.learning_rate == 0.01
        assert trainer.train_config["scheduler"] == {"factor": 0.5, "patience": 2}

    def test_the_callers_config_is_not_modified(self):
        config = {"learning_rate": 0.01}
        ECGTrainer(SimpleCNN(input_channels=1, output_size=4), config, device="cpu")
        assert config == {"learning_rate": 0.01}


class TestSelfDescribingCheckpoints:
    """A checkpoint says what it is and what its inputs went through."""

    @pytest.fixture
    def fitted(self, af_dataset_dir, temp_dir):
        data_module = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(af_dataset_dir),
            batch_size=16,
            verbose=False,
            dataset_kwargs={"verbose": False},
        )
        trainer = ECGTrainer(SimpleCNN(input_channels=1, output_size=4), device="cpu")
        trainer.fit(
            data_module, epochs=1, checkpoint_dir=temp_dir / "checkpoints", progress_bar=False
        )
        return trainer, data_module

    def test_validate_and_test_report_their_own_keys(self, fitted):
        trainer, data_module = fitted

        assert set(trainer.validate(data_module)) == {"val_loss", "val_acc"}
        assert set(trainer.test(data_module)) == {"test_loss", "test_acc"}

    def test_checkpoint_records_the_architecture(self, fitted):
        trainer, _ = fitted
        checkpoint = torch.load(trainer.best_checkpoint_path, weights_only=False)

        assert checkpoint["model_config"] == {
            "name": "simple-cnn",
            "input_channels": 1,
            "output_size": 4,
        }

    def test_checkpoint_records_the_preprocessing_and_classes(self, fitted):
        trainer, _ = fitted
        data_config = torch.load(trainer.best_checkpoint_path, weights_only=False)["data_config"]

        assert data_config["dataset"] == "af-classification"
        assert data_config["preprocessing"]["sampling_rate"] == 300
        assert data_config["preprocessing"]["segment_duration_seconds"] == 10.0
        assert data_config["preprocessing"]["normalization"] == "zscore"
        assert data_config["class_names"] == ["Normal", "AF", "Other", "Noisy"]
        assert data_config["input_length"] == 3000

    def test_explicit_configs_are_left_alone(self, af_dataset_dir):
        data_module = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(af_dataset_dir),
            verbose=False,
            dataset_kwargs={"verbose": False},
        )
        trainer = ECGTrainer(SimpleCNN(input_channels=1, output_size=4), device="cpu")
        trainer.model_config = {"name": "custom", "input_channels": 1, "output_size": 4}
        trainer.data_config = {"dataset": "mine"}

        trainer.fit(data_module, epochs=1, progress_bar=False)

        assert trainer.model_config["name"] == "custom"
        assert trainer.data_config == {"dataset": "mine"}

    def test_load_checkpoint_rebuilds_the_recorded_model(self, fitted):
        trainer, _ = fitted

        restored = ECGTrainer.load_checkpoint(trainer.best_checkpoint_path, device="cpu")

        assert isinstance(restored.model, SimpleCNN)
        assert restored.data_config["preprocessing"]["sampling_rate"] == 300
        for key, value in trainer.model.state_dict().items():
            assert torch.equal(value.cpu(), restored.model.state_dict()[key].cpu())

    def test_load_checkpoint_without_a_recorded_model_asks_for_one(self, temp_dir):
        path = temp_dir / "bare.pt"
        torch.save(
            {"model_state_dict": SimpleCNN(1, 4).state_dict(), "train_config": {}}, str(path)
        )
        with pytest.raises(ValueError, match="does not record which architecture"):
            ECGTrainer.load_checkpoint(path, device="cpu")


class TestCheckpointArchitectureErrors:
    """Loading weights into the wrong architecture fails in one readable line."""

    def test_unrecorded_checkpoint_with_the_wrong_name(self, temp_dir):
        from deepecgkit.models.factory import build_model_from_checkpoint

        path = temp_dir / "bare.pt"
        torch.save({"model_state_dict": SimpleCNN(1, 4).state_dict()}, str(path))

        with pytest.raises(ValueError, match="do not fit the 'kanres' architecture") as excinfo:
            build_model_from_checkpoint(path, name="kanres", input_channels=1)

        assert "\n" not in str(excinfo.value)


UNEVEN_SAMPLES = 10
UNEVEN_BATCH_SIZE = 8
ZERO_CLASS_LOGIT = 2.0


class _AlwaysClassZero(torch.nn.Module):
    """A two-class model whose answer does not depend on its input."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([ZERO_CLASS_LOGIT, 0.0]))

    def forward(self, x):
        return self.logits.expand(len(x), 2)


class _Loaders:
    """The least a trainer asks of a data module: one loader for every split."""

    def __init__(self, loader):
        self.loader = loader

    def train_dataloader(self):
        return self.loader

    def val_dataloader(self):
        return self.loader

    def test_dataloader(self):
        return self.loader


class TestMetricsAreAveragedOverSamples:
    """A short last batch must not count for as much as a full one.

    Ten samples in batches of eight leave a batch of two. The first batch is
    all class 0, which the model always predicts, and the last is all class 1:
    eight of ten are right, where the mean of the two batch accuracies is a half.
    """

    @pytest.fixture
    def uneven(self):
        labels = torch.tensor([0] * UNEVEN_BATCH_SIZE + [1] * (UNEVEN_SAMPLES - UNEVEN_BATCH_SIZE))
        signals = torch.zeros(UNEVEN_SAMPLES, 1, 16)
        dataset = torch.utils.data.TensorDataset(signals, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=UNEVEN_BATCH_SIZE)
        logits = torch.tensor([[ZERO_CLASS_LOGIT, 0.0]]).expand(UNEVEN_SAMPLES, 2)
        expected_loss = torch.nn.functional.cross_entropy(logits, labels).item()
        return _Loaders(loader), expected_loss

    def test_test_and_validate_weight_every_sample_equally(self, uneven):
        data_module, expected_loss = uneven
        trainer = ECGTrainer(_AlwaysClassZero(), device="cpu")

        tested = trainer.test(data_module)
        validated = trainer.validate(data_module)

        assert tested["test_acc"] == pytest.approx(0.8)
        assert tested["test_loss"] == pytest.approx(expected_loss)
        assert validated["val_acc"] == pytest.approx(0.8)
        assert validated["val_loss"] == pytest.approx(expected_loss)

    def test_the_loss_that_selects_checkpoints_is_a_sample_mean(self, uneven, temp_dir):
        """With a learning rate of zero the model stays put, so the epoch's loss is known."""
        data_module, expected_loss = uneven
        trainer = ECGTrainer(_AlwaysClassZero(), train_config={"learning_rate": 0.0}, device="cpu")

        trainer.fit(
            data_module, epochs=1, checkpoint_dir=temp_dir / "checkpoints", progress_bar=False
        )

        assert trainer.best_val_loss == pytest.approx(expected_loss)
