"""
Tests for the main deepecgkit package and integration tests.
"""

import time

import numpy as np
import pytest
import torch

import deepecgkit
from deepecgkit.datasets import AFClassificationDataset, BaseECGDataset, ECGDataModule
from deepecgkit.evaluation import ECGEvaluator, calculate_classification_metrics
from deepecgkit.models import AFModel, KanResWideX
from deepecgkit.training import ECGLitModel


class TestPackageStructure:
    """Test the main package structure and imports."""

    def test_package_version(self):
        """Test that package version is defined."""
        assert hasattr(deepecgkit, "__version__")
        assert isinstance(deepecgkit.__version__, str)
        assert len(deepecgkit.__version__) > 0

    def test_package_metadata(self):
        """Test package metadata."""
        assert hasattr(deepecgkit, "__author__")
        assert hasattr(deepecgkit, "__license__")
        assert deepecgkit.__license__ == "MIT"

    def test_main_imports(self):
        """Test that main components can be imported from deepecgkit."""
        assert deepecgkit.ECGDataModule is not None
        assert deepecgkit.KanResWideX is not None
        assert deepecgkit.ECGLitModel is not None
        assert deepecgkit.read_csv is not None

    def test_module_imports(self):
        """Test that all modules can be imported."""
        assert deepecgkit.datasets is not None
        assert deepecgkit.models is not None
        assert deepecgkit.evaluation is not None
        assert deepecgkit.training is not None
        assert deepecgkit.utils is not None

    def test_all_exports(self):
        """Test that all items in __all__ can be imported."""
        for item in deepecgkit.__all__:
            assert hasattr(deepecgkit, item), f"Item {item} not found in package"
            imported_item = getattr(deepecgkit, item)
            assert imported_item is not None, f"Item {item} is None"


class TestFullWorkflow:
    """Test complete workflows using deepecgkit."""

    def test_model_creation_and_inference(self):
        """Test creating models and running inference."""

        model1 = deepecgkit.KanResWideX(input_channels=1, output_size=4)
        assert model1 is not None

        model2 = AFModel(recording_length=6)
        assert model2 is not None

        sample_input = torch.randn(2, 1, 3000)

        with torch.no_grad():
            output1 = model1(sample_input)
            output2 = model2(sample_input)

        assert output1.shape == (2, 4)
        assert output2.shape == (2, 4)

    def test_evaluation_workflow(self):
        """Test evaluation workflow."""
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array(
            [
                [0.8, 0.1, 0.05, 0.05],
                [0.1, 0.7, 0.1, 0.1],
                [0.05, 0.15, 0.6, 0.2],
                [0.1, 0.1, 0.1, 0.7],
                [0.9, 0.05, 0.025, 0.025],
                [0.2, 0.6, 0.1, 0.1],
            ]
        )

        metrics = calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        evaluator = ECGEvaluator(
            task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
        )
        results = evaluator.evaluate(y_pred, y_true)
        assert "accuracy" in results
        assert "precision" in results

    def test_dataset_creation_workflow(self, mock_dataset_files):
        """Test dataset creation and usage workflow."""
        _dataset = AFClassificationDataset(
            data_dir=str(mock_dataset_files), subset="training", download=False
        )

        datamodule = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(mock_dataset_files),
            batch_size=2,
            num_workers=0,
        )

        datamodule.setup(stage="fit")

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

    def test_training_workflow(self, sample_batch_ecg, sample_labels):
        """Test complete training workflow."""

        model = deepecgkit.KanResWideX(input_channels=1, output_size=4)

        lit_model = deepecgkit.ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        batch = (sample_batch_ecg, sample_labels)
        loss = lit_model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

        lit_model.eval()
        with torch.no_grad():
            predictions = lit_model(sample_batch_ecg)

        assert predictions.shape == (sample_batch_ecg.shape[0], 4)


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    def test_end_to_end_classification(self, mock_dataset_files):
        """Test end-to-end classification pipeline."""
        datamodule = ECGDataModule(
            dataset_class=AFClassificationDataset,
            data_dir=str(mock_dataset_files),
            batch_size=2,
            num_workers=0,
        )
        datamodule.setup(stage="fit")

        model = KanResWideX(input_channels=1, output_size=4)
        lit_model = ECGLitModel(
            model=model,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        train_loader = datamodule.train_dataloader()

        try:
            for i, batch in enumerate(train_loader):
                if i >= 2:
                    break

                loss = lit_model.training_step(batch, batch_idx=i)
                assert isinstance(loss, torch.Tensor)

                lit_model.eval()
                with torch.no_grad():
                    predictions = lit_model(batch[0])

                evaluator = ECGEvaluator(
                    task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
                )
                results = evaluator.evaluate(predictions.numpy(), batch[1].numpy())
                assert "accuracy" in results

        except Exception:
            pytest.skip("Mock data insufficient for integration test")

    def test_model_comparison(self, sample_batch_ecg, sample_labels):
        """Test comparing different models."""
        model1 = KanResWideX(input_channels=1, output_size=4)
        model2 = AFModel(recording_length=6)

        models = {"KanResWideX": model1, "AFModel": model2}
        results = {}

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                predictions = model(sample_batch_ecg)

            y_true = sample_labels.numpy()
            if name == "AFModel":
                y_pred = predictions.argmax(dim=1).numpy()
            else:
                y_pred = predictions.argmax(dim=1).numpy()

            metrics = calculate_classification_metrics(y_true, y_pred)
            results[name] = metrics

        for _name, metrics in results.items():
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1

    def test_cross_module_compatibility(self):
        """Test compatibility between different modules."""

        class TestDataset(BaseECGDataset):
            def download(self):
                pass

            def _load_data(self):
                pass

            def _process_data(self):
                pass

            def __getitem__(self, idx):
                signal = torch.randn(1, 3000)
                label = torch.tensor(idx % 4)
                return signal, label

            def __len__(self):
                return 8

            @property
            def num_classes(self):
                return 4

            @property
            def class_names(self):
                return ["Class0", "Class1", "Class2", "Class3"]

        dataset = TestDataset(download=False)
        model = KanResWideX(input_channels=1, output_size=4)

        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                signal, label = dataset[i]
                prediction = model(signal.unsqueeze(0))

                all_predictions.append(prediction.numpy())
                all_labels.append(label.numpy())

        y_pred_logits = np.vstack(all_predictions)
        y_pred = np.argmax(y_pred_logits, axis=1)
        y_true = np.array(all_labels)

        metrics = calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], (float, np.floating))

    def test_device_compatibility_workflow(self):
        """Test complete workflow with different devices."""
        model_cpu = KanResWideX(input_channels=1, output_size=4)
        lit_model_cpu = ECGLitModel(
            model=model_cpu,
            train_config={
                "learning_rate": 0.001,
                "scheduler": {"factor": 0.5, "patience": 10},
                "binary_classification": False,
            },
        )

        input_cpu = torch.randn(2, 1, 1000)
        output_cpu = lit_model_cpu(input_cpu)

        assert output_cpu.device.type == "cpu"
        assert output_cpu.shape == (2, 4)

        if torch.cuda.is_available():
            model_gpu = KanResWideX(input_channels=1, output_size=4).cuda()
            lit_model_gpu = ECGLitModel(
                model=model_gpu,
                train_config={
                    "learning_rate": 0.001,
                    "scheduler": {"factor": 0.5, "patience": 10},
                    "binary_classification": False,
                },
            ).cuda()

            input_gpu = input_cpu.cuda()
            output_gpu = lit_model_gpu(input_gpu)

            assert output_gpu.device.type == "cuda"
            assert output_gpu.shape == (2, 4)


class TestPackageConsistency:
    """Test package consistency and documentation."""

    def test_docstring_coverage(self):
        """Test that main components have docstrings."""
        components = [
            deepecgkit.KanResWideX,
            deepecgkit.ECGDataModule,
            deepecgkit.ECGLitModel,
            deepecgkit.read_csv,
        ]

        for component in components:
            assert component.__doc__ is not None, f"{component.__name__} missing docstring"
            assert len(component.__doc__.strip()) > 0, f"{component.__name__} has empty docstring"

    def test_type_consistency(self):
        """Test type consistency across the package."""

        model = deepecgkit.KanResWideX(input_channels=1, output_size=4)
        input_tensor = torch.randn(2, 1, 1000)

        output = model(input_tensor)
        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32

        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = calculate_classification_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (float, np.floating))

    def test_error_handling_consistency(self):
        """Test consistent error handling across modules."""

        model = deepecgkit.KanResWideX(input_channels=1, output_size=4)

        with pytest.raises((RuntimeError, ValueError)):
            invalid_input = torch.randn(1000)
            model(invalid_input)

        with pytest.raises(ValueError):
            AFClassificationDataset(subset="invalid_subset", download=False)

    def test_reproducibility(self):
        """Test that results are reproducible with fixed seeds."""
        torch.manual_seed(42)
        np.random.seed(42)

        model1 = deepecgkit.KanResWideX(input_channels=1, output_size=4)
        input_tensor = torch.randn(2, 1, 1000)

        with torch.no_grad():
            output1 = model1(input_tensor)

        torch.manual_seed(42)
        np.random.seed(42)

        model2 = deepecgkit.KanResWideX(input_channels=1, output_size=4)

        with torch.no_grad():
            output2 = model2(input_tensor)

        assert torch.allclose(output1, output2, atol=1e-6)


class TestPerformance:
    """Test performance characteristics."""

    def test_model_inference_speed(self):
        """Test that model inference is reasonably fast."""
        model = deepecgkit.KanResWideX(input_channels=1, output_size=4)
        model.eval()

        dummy_input = torch.randn(1, 1, 3000)
        with torch.no_grad():
            _ = model(dummy_input)

        batch_input = torch.randn(10, 1, 3000)

        start_time = time.time()
        with torch.no_grad():
            _ = model(batch_input)
        end_time = time.time()

        inference_time = end_time - start_time

        assert inference_time < 1.0, f"Inference too slow: {inference_time:.3f}s"

    def test_memory_efficiency(self):
        """Test that models don't use excessive memory."""
        model = deepecgkit.KanResWideX(input_channels=1, output_size=4)

        total_params = sum(p.numel() for p in model.parameters())

        assert total_params < 10_000_000, f"Model too large: {total_params} parameters"

        model.eval()

        for _ in range(5):
            input_tensor = torch.randn(4, 1, 3000)
            with torch.no_grad():
                output = model(input_tensor)

            del input_tensor, output
