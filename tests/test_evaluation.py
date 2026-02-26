"""
Tests for deepecgkit.evaluation module.
"""

import numpy as np
import pytest
import torch

from deepecgkit.evaluation import (
    ECGEvaluator,
    calculate_classification_metrics,
    calculate_regression_metrics,
    confusion_matrix_analysis,
    plot_confusion_matrix,
    plot_ecg_signals,
    plot_predictions,
    plot_roc_curve,
)
from deepecgkit.models import KanResWideX


class TestClassificationMetrics:
    """Test classification metrics functions."""

    def test_calculate_classification_metrics_binary(self):
        """Test binary classification metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics

        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

    def test_calculate_classification_metrics_multiclass(self):
        """Test multiclass classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 2])
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.9, 0.05, 0.05],
                [0.3, 0.6, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.8, 0.1],
                [0.05, 0.15, 0.8],
            ]
        )

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "auc" in metrics

        assert 0 <= metrics["auc"] <= 1

    def test_calculate_classification_metrics_custom_metrics(self):
        """Test with custom metrics list."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.3, 0.7, 0.8])

        metrics = calculate_classification_metrics(y_true, y_pred, metrics=["accuracy", "f1"])

        assert len(metrics) == 2
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" not in metrics

    def test_calculate_classification_metrics_edge_cases(self):
        """Test edge cases for classification metrics."""

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        metrics = calculate_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0

        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.7, 0.6])

        metrics = calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics


class TestRegressionMetrics:
    """Test regression metrics functions."""

    def test_calculate_regression_metrics_basic(self):
        """Test basic regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r2"] <= 1

    def test_calculate_regression_metrics_perfect(self):
        """Test with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        metrics = calculate_regression_metrics(y_true, y_pred)

        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_calculate_regression_metrics_custom_metrics(self):
        """Test with custom metrics list."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])

        metrics = calculate_regression_metrics(y_true, y_pred, metrics=["mse", "rmse"])

        assert len(metrics) == 2
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" not in metrics

        assert np.isclose(metrics["rmse"], np.sqrt(metrics["mse"]))


class TestConfusionMatrixAnalysis:
    """Test confusion matrix analysis."""

    def test_confusion_matrix_analysis_binary(self):
        """Test confusion matrix analysis for binary classification."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        result = confusion_matrix_analysis(y_true, y_pred)

        assert "confusion_matrix" in result
        assert "per_class_precision" in result
        assert "per_class_recall" in result
        assert "per_class_f1" in result
        assert "macro_precision" in result
        assert "macro_recall" in result
        assert "macro_f1" in result

        assert result["confusion_matrix"].shape == (2, 2)
        assert len(result["per_class_precision"]) == 2
        assert len(result["per_class_recall"]) == 2
        assert len(result["per_class_f1"]) == 2

    def test_confusion_matrix_analysis_multiclass(self):
        """Test confusion matrix analysis for multiclass classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 1, 1, 2])

        result = confusion_matrix_analysis(y_true, y_pred)

        assert result["confusion_matrix"].shape == (3, 3)
        assert len(result["per_class_precision"]) == 3
        assert len(result["per_class_recall"]) == 3
        assert len(result["per_class_f1"]) == 3

        assert 0 <= result["macro_precision"] <= 1
        assert 0 <= result["macro_recall"] <= 1
        assert 0 <= result["macro_f1"] <= 1


class TestECGEvaluator:
    """Test the ECGEvaluator class."""

    def test_ecg_evaluator_init(self):
        """Test ECGEvaluator initialization."""
        evaluator = ECGEvaluator()
        assert evaluator is not None

    @pytest.mark.parametrize("task_type", ["classification", "regression"])
    def test_ecg_evaluator_with_task_type(self, task_type):
        """Test ECGEvaluator with different task types."""
        evaluator = ECGEvaluator(task_type=task_type)
        assert evaluator.task_type == task_type

    def test_ecg_evaluator_classification_evaluation(self, sample_labels, sample_probabilities):
        """Test classification evaluation."""
        evaluator = ECGEvaluator(
            task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
        )

        y_true = sample_labels.numpy()
        y_pred = sample_probabilities.numpy()

        results = evaluator.evaluate(model=y_pred, test_data=y_true, y_scores=y_pred)

        assert "accuracy" in results

    def test_ecg_evaluator_regression_evaluation(self, sample_regression_targets):
        """Test regression evaluation."""
        evaluator = ECGEvaluator(task_type="regression", metrics=["mse", "mae", "r2"])

        y_true = sample_regression_targets.numpy()
        y_pred = y_true + np.random.normal(0, 0.1, y_true.shape)

        results = evaluator.evaluate(model=y_pred, test_data=y_true)

        assert "mse" in results
        assert "mae" in results
        assert "r2" in results

    def test_ecg_evaluator_batch_evaluation(self):
        """Test batch evaluation capabilities."""
        evaluator = ECGEvaluator(
            task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
        )

        batches = [
            {"y_true": np.array([0, 1, 0, 1]), "y_pred": np.array([0.1, 0.9, 0.2, 0.8])},
            {"y_true": np.array([1, 0, 1, 0]), "y_pred": np.array([0.8, 0.3, 0.7, 0.4])},
        ]

        batch_results = []
        for batch in batches:
            result = evaluator.evaluate(model=batch["y_pred"], test_data=batch["y_true"])
            batch_results.append(result)

        assert len(batch_results) == 2
        for result in batch_results:
            assert "accuracy" in result

    def test_ecg_evaluator_with_model_outputs(self, sample_batch_ecg, sample_labels):
        """Test evaluator with actual model outputs."""
        model = KanResWideX(input_channels=1, output_size=4)
        model.eval()

        with torch.no_grad():
            outputs = model(sample_batch_ecg)

        evaluator = ECGEvaluator(
            task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
        )

        y_true = sample_labels.numpy()
        y_pred = outputs.numpy()

        results = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in results


class TestVisualizationFunctions:
    """Test visualization functions from the evaluation module."""

    def test_plot_confusion_matrix_import(self):
        """Test that plot_confusion_matrix can be imported."""
        assert plot_confusion_matrix is not None

    def test_plot_roc_curve_import(self):
        """Test that plot_roc_curve can be imported."""
        assert plot_roc_curve is not None

    def test_plot_ecg_signals_import(self):
        """Test that plot_ecg_signals can be imported."""
        assert plot_ecg_signals is not None

    def test_plot_predictions_import(self):
        """Test that plot_predictions can be imported."""
        assert plot_predictions is not None


class TestMetricEdgeCases:
    """Test edge cases and error handling in metrics."""

    def test_empty_arrays(self):
        """Test metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])

        try:
            metrics = calculate_classification_metrics(y_true, y_pred)

            for value in metrics.values():
                assert np.isnan(value)
        except ValueError:
            pass

    def test_mismatched_shapes(self):
        """Test metrics with mismatched array shapes."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.9])

        metrics = calculate_classification_metrics(y_true, y_pred)

        for value in metrics.values():
            assert np.isnan(value)

    def test_single_class_predictions(self):
        """Test metrics when all predictions are the same class."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], (float, np.floating))

    def test_invalid_probabilities(self):
        """Test with invalid probability values."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1.5, -0.2, 0.5, 0.8])

        metrics = calculate_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics


class TestIntegrationWithModels:
    """Integration tests with actual models."""

    def test_evaluation_pipeline(self, sample_batch_ecg, sample_labels):
        """Test complete evaluation pipeline with a model."""
        model = KanResWideX(input_channels=1, output_size=4)
        model.eval()

        with torch.no_grad():
            predictions = model(sample_batch_ecg)

        evaluator = ECGEvaluator(
            task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
        )
        results = evaluator.evaluate(
            model=predictions.numpy(), test_data=sample_labels.numpy(), y_scores=predictions.numpy()
        )

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

        assert all(0 <= v <= 1 for v in results.values() if not np.isnan(v))

    def test_cross_validation_simulation(self, sample_batch_ecg, sample_labels):
        """Simulate cross-validation evaluation."""
        model = KanResWideX(input_channels=1, output_size=4)
        evaluator = ECGEvaluator(
            task_type="classification", metrics=["accuracy", "precision", "recall", "f1"]
        )

        fold_results = []
        for _fold in range(3):
            model.eval()
            with torch.no_grad():
                predictions = model(sample_batch_ecg)

            results = evaluator.evaluate(
                model=predictions.numpy(),
                test_data=sample_labels.numpy(),
                y_scores=predictions.numpy(),
            )
            fold_results.append(results)

        assert len(fold_results) == 3

        for result in fold_results:
            assert "accuracy" in result
            assert "precision" in result
