"""
Tests for deepecgkit.evaluation module.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from deepecgkit.evaluation import (
    TASK_BINARY,
    TASK_MULTICLASS,
    TASK_MULTILABEL,
    ECGEvaluator,
    TemperatureScaler,
    adaptive_calibration_error,
    apply_temperature,
    brier_score,
    calculate_calibration_metrics,
    calculate_classification_metrics,
    calculate_regression_metrics,
    calibration_report,
    confusion_matrix_analysis,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_ecg_signals,
    plot_predictions,
    plot_roc_curve,
    plot_training_curves,
    probabilities_to_logits,
    resolve_task_shape,
    temperature_nll,
)
from deepecgkit.models import KanResWideX

CALIBRATED_ECE_CEILING = 0.03
OVERCONFIDENT_ECE_FLOOR = 0.10
INJECTED_TEMPERATURE = 3.0
TEMPERATURE_TOLERANCE = 0.15


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

    def test_float32_scores_are_thresholded_like_float64(self):
        """Scores converted from a torch tensor are float32 and must not be cast to labels."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.4])

        expected = calculate_classification_metrics(y_true, scores)
        metrics = calculate_classification_metrics(y_true, scores.astype(np.float32))

        assert metrics == pytest.approx(expected)
        assert metrics["accuracy"] == 1.0

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

        results = evaluator.evaluate(y_pred, y_true)

        assert results["accuracy"] == pytest.approx((y_pred.argmax(axis=1) == y_true).mean())


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
        assert isinstance(metrics["accuracy"], float | np.floating)

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


class TestCalibrationSanity:
    """The two cases that pin the calibration implementation."""

    def test_perfectly_calibrated_probabilities_score_near_zero_ece(
        self, calibrated_multiclass_predictions
    ):
        """Labels drawn from the predictions leave nothing but sampling noise."""
        y_true, y_prob, _ = calibrated_multiclass_predictions

        assert expected_calibration_error(y_true, y_prob) < CALIBRATED_ECE_CEILING
        assert adaptive_calibration_error(y_true, y_prob) < CALIBRATED_ECE_CEILING

    def test_overconfident_probabilities_score_high_ece(self, overconfident_multiclass_predictions):
        """Sharpening the same predictions must be visible as a large error."""
        y_true, y_prob, _ = overconfident_multiclass_predictions

        assert expected_calibration_error(y_true, y_prob) > OVERCONFIDENT_ECE_FLOOR
        assert adaptive_calibration_error(y_true, y_prob) > OVERCONFIDENT_ECE_FLOOR


class TestCalibrationMetrics:
    """Test the calibration metrics."""

    def test_calibration_metrics_returns_every_name(self, calibrated_multiclass_predictions):
        """The aggregate reports all four values plus the context to read them."""
        y_true, y_prob, _ = calibrated_multiclass_predictions

        metrics = calculate_calibration_metrics(y_true, y_prob)

        for name in ("ece", "adaptive_ece", "mce", "brier"):
            assert name in metrics
        assert metrics["n_bins"] == 15
        assert metrics["task"] == TASK_MULTICLASS

    def test_maximum_error_is_at_least_the_expected_error(
        self, overconfident_multiclass_predictions
    ):
        """A weighted mean of bin gaps cannot exceed the largest of them."""
        y_true, y_prob, _ = overconfident_multiclass_predictions

        assert maximum_calibration_error(y_true, y_prob) >= expected_calibration_error(
            y_true, y_prob
        )

    def test_bin_count_changes_the_reported_error(self, calibrated_multiclass_predictions):
        """ECE is bin-dependent, which is why the count travels with the value."""
        y_true, y_prob, _ = calibrated_multiclass_predictions

        coarse = expected_calibration_error(y_true, y_prob, n_bins=2)
        fine = expected_calibration_error(y_true, y_prob, n_bins=100)

        assert coarse != fine

    def test_bin_count_is_recorded_in_the_aggregate(self, calibrated_multiclass_predictions):
        """A value binned differently is not comparable, so the count is reported."""
        y_true, y_prob, _ = calibrated_multiclass_predictions

        assert calculate_calibration_metrics(y_true, y_prob, n_bins=7)["n_bins"] == 7

    def test_adaptive_bins_are_populated_where_uniform_bins_are_not(self):
        """Equal-mass binning is worth reporting because it never leaves a bin empty."""
        n_samples = 600
        rng = np.random.default_rng(5)
        confident = 0.95 + 0.05 * rng.random(n_samples)
        y_prob = np.column_stack([1.0 - confident, confident])
        y_true = (rng.random(n_samples) < 0.8).astype(int)

        uniform = expected_calibration_error(y_true, y_prob, n_bins=20)
        adaptive = adaptive_calibration_error(y_true, y_prob, n_bins=20)

        assert not np.isnan(uniform)
        assert not np.isnan(adaptive)
        assert adaptive > 0.0

    def test_certain_and_correct_predictions_score_zero(self):
        """A model that is always right and always certain has nothing to fix."""
        y_true = np.array([0, 1, 2, 1])
        y_prob = np.eye(3)[y_true].astype(float)

        assert expected_calibration_error(y_true, y_prob) == 0.0
        assert brier_score(y_true, y_prob) == 0.0

    def test_certain_and_wrong_predictions_score_the_worst_possible(self):
        """Full confidence in the wrong class is the extreme of both scales."""
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.tile(np.array([0.0, 1.0]), (4, 1))

        assert expected_calibration_error(y_true, y_prob) == 1.0
        assert brier_score(y_true, y_prob) == 2.0

    def test_empty_input_yields_nan_rather_than_raising(self):
        """An empty split reports nothing rather than crashing the run."""
        y_true = np.array([], dtype=int)
        y_prob = np.empty((0, 3))

        assert np.isnan(expected_calibration_error(y_true, y_prob))
        assert np.isnan(brier_score(y_true, y_prob))

    def test_unpopulated_bins_carry_no_weight(self):
        """Bins nobody predicted into must not dilute the gap of the ones that are used.

        Every sample here is 0.9 confident and half of them are wrong, so the
        error is 0.4 however many bins are requested. Were empty bins averaged
        in as zero gaps, asking for a hundred bins would report 0.004.
        """
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.tile(np.array([0.9, 0.1]), (4, 1))

        assert expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.4)
        assert expected_calibration_error(y_true, y_prob, n_bins=100) == pytest.approx(0.4)

    def test_confidence_matching_accuracy_scores_zero(self):
        """A model right as often as it claims has no calibration error."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.tile(np.array([0.5, 0.5]), (4, 1))

        assert expected_calibration_error(y_true, y_prob, n_bins=100) == 0.0


class TestCalibrationTaskShapes:
    """Test the three prediction shapes this library produces."""

    def test_task_shape_is_resolved_from_the_arrays(
        self,
        calibrated_multiclass_predictions,
        calibrated_binary_predictions,
        calibrated_multilabel_predictions,
    ):
        """Each shape is recognised without being told what it is."""
        multiclass_true, multiclass_prob, _ = calibrated_multiclass_predictions
        binary_true, binary_prob, _ = calibrated_binary_predictions
        multilabel_true, multilabel_prob, _ = calibrated_multilabel_predictions

        assert resolve_task_shape(multiclass_true, multiclass_prob) == TASK_MULTICLASS
        assert resolve_task_shape(binary_true, binary_prob) == TASK_BINARY
        assert resolve_task_shape(multilabel_true, multilabel_prob) == TASK_MULTILABEL

    def test_unknown_task_override_raises(self, calibrated_binary_predictions):
        """A misspelled shape must fail loudly rather than fall back silently."""
        y_true, y_prob, _ = calibrated_binary_predictions

        with pytest.raises(ValueError, match="Unknown task shape"):
            resolve_task_shape(y_true, y_prob, task="ordinal")

    def test_binary_accepts_one_column_and_two(self, calibrated_binary_predictions):
        """The stacked [1 - p, p] layout scores the same as bare probabilities."""
        y_true, y_prob, _ = calibrated_binary_predictions

        two_column = expected_calibration_error(y_true, y_prob)
        one_column = expected_calibration_error(y_true, y_prob[:, 1])

        assert np.isclose(two_column, one_column)

    def test_multi_label_macro_is_the_mean_of_its_classes(self, calibrated_multilabel_predictions):
        """Averaging happens over classes, so the macro value must be their mean."""
        y_true, y_prob, _ = calibrated_multilabel_predictions

        metrics = calculate_calibration_metrics(y_true, y_prob)

        assert len(metrics["per_class_ece"]) == y_prob.shape[1]
        assert np.isclose(metrics["ece"], metrics["per_class_ece"].mean())

    def test_multi_label_is_not_flattened_into_one_stream(self):
        """Pooling classes would let a good class hide a badly calibrated one."""
        n_samples = 2000
        rng = np.random.default_rng(11)
        good = rng.random(n_samples)
        y_prob = np.column_stack([good, np.full(n_samples, 0.99)])
        y_true = np.column_stack(
            [(rng.random(n_samples) < good).astype(int), np.zeros(n_samples, dtype=int)]
        )

        metrics = calculate_calibration_metrics(y_true, y_prob)

        assert metrics["per_class_ece"][1] > 0.9
        assert metrics["per_class_ece"][0] < 0.1
        assert np.isclose(metrics["ece"], metrics["per_class_ece"].mean())

    def test_multi_label_brier_stays_on_the_per_class_scale(
        self, calibrated_multilabel_predictions
    ):
        """The multi-label convention averages classes, so it cannot exceed one."""
        y_true, y_prob, _ = calibrated_multilabel_predictions

        assert 0.0 <= brier_score(y_true, y_prob) <= 1.0

    def test_explicit_task_overrides_inference(self, calibrated_multilabel_predictions):
        """A caller who knows the shape from the checkpoint can say so."""
        y_true, y_prob, _ = calibrated_multilabel_predictions

        metrics = calculate_calibration_metrics(y_true, y_prob, task=TASK_MULTILABEL)

        assert metrics["task"] == TASK_MULTILABEL


class TestTemperatureScaling:
    """Test fitting a single scalar temperature and applying it."""

    def test_calibrated_predictions_fit_a_neutral_temperature(
        self, calibrated_multiclass_predictions
    ):
        """A model that needs no correction should be left alone."""
        y_true, _, logits = calibrated_multiclass_predictions

        assert abs(fit_temperature(logits, y_true) - 1.0) < TEMPERATURE_TOLERANCE

    def test_overconfident_predictions_fit_a_temperature_above_one(
        self, overconfident_multiclass_predictions
    ):
        """Softening an overconfident model requires dividing by more than one."""
        y_true, _, logits = overconfident_multiclass_predictions

        assert fit_temperature(logits, y_true) > 1.0

    def test_fitted_temperature_recovers_the_injected_one(
        self, overconfident_multiclass_predictions
    ):
        """Logits scaled by a known factor should fit that factor back."""
        y_true, _, logits = overconfident_multiclass_predictions

        fitted = fit_temperature(logits, y_true)

        assert abs(fitted - INJECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE

    def test_binary_temperature_recovers_the_injected_one(self, overconfident_binary_predictions):
        """The single-logit head is fitted through its own likelihood."""
        y_true, _, logits = overconfident_binary_predictions

        assert abs(fit_temperature(logits, y_true) - INJECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE

    def test_multi_label_temperature_recovers_the_injected_one(
        self, overconfident_multilabel_predictions
    ):
        """One scalar is shared across classes, fitted on the per-class likelihood."""
        y_true, _, logits = overconfident_multilabel_predictions

        fitted = fit_temperature(logits, y_true, task=TASK_MULTILABEL)

        assert abs(fitted - INJECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE

    def test_scaling_lowers_negative_log_likelihood(self, overconfident_multiclass_predictions):
        """NLL is the objective, so the fitted temperature must reduce it."""
        y_true, _, logits = overconfident_multiclass_predictions

        scaler = TemperatureScaler().fit(logits, y_true)

        assert scaler.nll(logits, y_true) < temperature_nll(logits, y_true, temperature=1.0)

    def test_fitting_on_validation_lowers_test_calibration_error(
        self, overconfident_multiclass_predictions
    ):
        """The supported use is fit on one split, report on another."""
        y_true, y_prob, logits = overconfident_multiclass_predictions
        half = len(y_true) // 2

        scaler = TemperatureScaler().fit(logits[:half], y_true[:half])
        before = expected_calibration_error(y_true[half:], y_prob[half:])
        after = expected_calibration_error(y_true[half:], scaler.transform(logits[half:]))

        assert after < before

    def test_scaling_leaves_the_predicted_class_unchanged(
        self, overconfident_multiclass_predictions
    ):
        """Dividing every logit by one positive scalar cannot reorder them."""
        y_true, y_prob, logits = overconfident_multiclass_predictions

        scaler = TemperatureScaler().fit(logits, y_true)

        assert np.array_equal(
            np.argmax(y_prob, axis=1), np.argmax(scaler.transform(logits), axis=1)
        )

    def test_binary_transform_returns_the_two_column_layout(self, overconfident_binary_predictions):
        """Scaled probabilities must be a drop-in for what ECGTrainer stores."""
        y_true, y_prob, logits = overconfident_binary_predictions

        scaled = TemperatureScaler().fit(logits, y_true).transform(logits)

        assert scaled.shape == y_prob.shape
        assert np.allclose(scaled.sum(axis=1), 1.0)

    def test_transform_before_fit_raises(self, calibrated_multiclass_predictions):
        """An unfitted scaler has no temperature to apply."""
        _, _, logits = calibrated_multiclass_predictions

        with pytest.raises(RuntimeError, match="must be fitted"):
            TemperatureScaler().transform(logits)

    def test_non_positive_temperature_raises(self, calibrated_multiclass_predictions):
        """Zero or negative temperatures are not a meaningful rescaling."""
        _, _, logits = calibrated_multiclass_predictions

        with pytest.raises(ValueError, match="must be positive"):
            apply_temperature(logits, 0.0)

    def test_empty_fitting_split_falls_back_to_no_scaling(self):
        """A run without validation data must leave probabilities untouched."""
        with pytest.warns(RuntimeWarning, match="empty split"):
            assert fit_temperature(np.empty((0, 4)), np.array([], dtype=int)) == 1.0

    def test_probability_round_trip_matches_true_logits(self, calibrated_multiclass_predictions):
        """Recovering logits from probabilities is exact away from saturation."""
        _, y_prob, logits = calibrated_multiclass_predictions

        recovered = probabilities_to_logits(y_prob.astype(np.float32))

        assert np.allclose(
            apply_temperature(logits, 2.0), apply_temperature(recovered, 2.0), atol=1e-6
        )

    def test_probability_round_trip_degrades_as_logits_saturate(
        self, overconfident_multiclass_predictions
    ):
        """Confident logits lose precision in float32, which is why they are captured.

        This is the cost the fallback path pays and the reason ECGTrainer keeps
        the real logits rather than rebuilding them from stored probabilities.
        """
        _, y_prob, logits = overconfident_multiclass_predictions

        recovered = probabilities_to_logits(y_prob.astype(np.float32))
        error = np.abs(apply_temperature(logits, 2.0) - apply_temperature(recovered, 2.0)).max()

        assert error > 1e-6
        assert error < 1e-2

    def test_multi_label_round_trip_preserves_per_class_probabilities(
        self, calibrated_multilabel_predictions
    ):
        """Per-class sigmoids invert exactly, given the shape is declared."""
        _, y_prob, logits = calibrated_multilabel_predictions

        recovered = probabilities_to_logits(y_prob.astype(np.float32), task=TASK_MULTILABEL)

        assert np.allclose(
            apply_temperature(logits, 1.7, task=TASK_MULTILABEL),
            apply_temperature(recovered, 1.7, task=TASK_MULTILABEL),
            atol=1e-5,
        )


class TestCalibrationReport:
    """Test the flat report the CLI and the metrics CSV are built from."""

    def test_report_without_fitting_data_leaves_scaled_entries_missing(
        self, overconfident_multiclass_predictions
    ):
        """A run with no validation split still produces a well-formed row."""
        y_true, y_prob, _ = overconfident_multiclass_predictions

        report = calibration_report(y_true, y_prob)

        assert report["ece"] > OVERCONFIDENT_ECE_FLOOR
        assert np.isnan(report["temperature"])
        assert np.isnan(report["ece_temp"])

    def test_report_with_fitting_data_improves_calibration(
        self, overconfident_multiclass_predictions
    ):
        """The before and after columns are what the paper table compares."""
        y_true, y_prob, logits = overconfident_multiclass_predictions
        half = len(y_true) // 2

        report = calibration_report(
            y_true[half:],
            y_prob[half:],
            fit_targets=y_true[:half],
            fit_logits=logits[:half],
            eval_logits=logits[half:],
        )

        assert report["ece_temp"] < report["ece"]
        assert abs(report["temperature"] - INJECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE

    def test_report_records_the_bin_count(self, calibrated_multiclass_predictions):
        """Every reported error carries the binning that produced it."""
        y_true, y_prob, _ = calibrated_multiclass_predictions

        assert calibration_report(y_true, y_prob, n_bins=9)["calibration_bins"] == 9

    def test_report_falls_back_to_probabilities_when_logits_are_absent(
        self, overconfident_multiclass_predictions
    ):
        """Callers holding only stored probabilities still get scaled numbers."""
        y_true, y_prob, logits = overconfident_multiclass_predictions
        half = len(y_true) // 2

        report = calibration_report(
            y_true[half:],
            y_prob[half:],
            fit_targets=y_true[:half],
            fit_logits=logits[:half],
        )

        assert report["ece_temp"] < report["ece"]

    def test_report_covers_the_multi_label_shape(self, overconfident_multilabel_predictions):
        """Multi-label runs need the same row shape as single-label ones."""
        y_true, y_prob, logits = overconfident_multilabel_predictions
        half = len(y_true) // 2

        report = calibration_report(
            y_true[half:],
            y_prob[half:],
            fit_targets=y_true[:half],
            fit_logits=logits[:half],
            eval_logits=logits[half:],
        )

        assert report["ece_temp"] < report["ece"]
        assert abs(report["temperature"] - INJECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE


class TestEvaluatorCalibration:
    """Test that calibration reaches the evaluator report."""

    def test_generate_report_includes_calibration_rows(self, overconfident_multiclass_predictions):
        """Calibration is reported without being asked for."""
        y_true, y_prob, _ = overconfident_multiclass_predictions
        evaluator = ECGEvaluator(metrics=["accuracy"])

        report = evaluator.generate_report(y_prob, y_true, y_scores=y_prob)

        assert set(report["Metric"]).issuperset({"ece", "adaptive_ece", "mce", "brier"})

    def test_generate_report_omits_temperature_without_a_fitting_split(
        self, overconfident_multiclass_predictions
    ):
        """Empty temperature rows would imply a fit that never happened."""
        y_true, y_prob, _ = overconfident_multiclass_predictions
        evaluator = ECGEvaluator(metrics=["accuracy"])

        report = evaluator.generate_report(y_prob, y_true, y_scores=y_prob)

        assert "temperature" not in set(report["Metric"])

    def test_generate_report_adds_temperature_rows_when_fitted(
        self, overconfident_multiclass_predictions
    ):
        """Passing a validation split adds the after-scaling counterparts."""
        y_true, y_prob, logits = overconfident_multiclass_predictions
        half = len(y_true) // 2
        evaluator = ECGEvaluator(metrics=["accuracy"])

        report = evaluator.generate_report(
            y_prob[half:],
            y_true[half:],
            y_scores=y_prob[half:],
            calibration_fit=(y_true[:half], logits[:half]),
        )

        values = dict(zip(report["Metric"], report["Value"], strict=False))
        assert values["ece_temp"] < values["ece"]
        assert abs(values["temperature"] - INJECTED_TEMPERATURE) < TEMPERATURE_TOLERANCE

    def test_evaluator_honours_a_custom_bin_count(self, calibrated_multiclass_predictions):
        """The bin count is a property of the evaluator, not a hidden constant."""
        y_true, y_prob, _ = calibrated_multiclass_predictions
        evaluator = ECGEvaluator(metrics=["accuracy"], calibration_bins=6)

        report = evaluator.generate_report(y_prob, y_true, y_scores=y_prob)

        values = dict(zip(report["Metric"], report["Value"], strict=False))
        assert values["calibration_bins"] == 6


EVALUATOR_SAMPLES = 64
EVALUATOR_SIGNAL_LENGTH = 100
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_ACCURACY = 0.8


class _SmallClassifier(torch.nn.Module):
    """A classifier with the two layers that behave differently in training mode."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(1)
        self.drop = torch.nn.Dropout(0.5)
        self.head = torch.nn.Linear(EVALUATOR_SIGNAL_LENGTH, n_classes)

    def forward(self, x):
        return self.head(self.drop(self.norm(x)).flatten(1))


class _SignLogit(torch.nn.Module):
    """A model with one output per sample: five times the mean of its input."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        return (self.scale * x.mean(dim=(1, 2))).unsqueeze(-1)


class _SoftmaxHead(torch.nn.Module):
    """A classifier that already ends in a softmax."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(EVALUATOR_SIGNAL_LENGTH, 4)

    def forward(self, x):
        return torch.softmax(self.head(x.flatten(1)), dim=1)


@pytest.fixture
def multiclass_loader() -> DataLoader:
    """Four-class labels over signals that say nothing about them."""
    signals = torch.randn(EVALUATOR_SAMPLES, 1, EVALUATOR_SIGNAL_LENGTH)
    labels = torch.arange(EVALUATOR_SAMPLES) % 4
    return DataLoader(TensorDataset(signals, labels), batch_size=16)


@pytest.fixture
def binary_loader() -> DataLoader:
    """Signals whose mean is positive exactly when the label is one."""
    labels = torch.arange(EVALUATOR_SAMPLES) % 2
    offsets = (2.0 * labels - 1.0).reshape(-1, 1, 1)
    signals = 0.1 * torch.randn(EVALUATOR_SAMPLES, 1, EVALUATOR_SIGNAL_LENGTH) + offsets
    return DataLoader(TensorDataset(signals, labels), batch_size=16)


@pytest.fixture
def labelled_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Binary targets and hard labels that agree on BOOTSTRAP_ACCURACY of the samples."""
    y_true = np.arange(BOOTSTRAP_SAMPLES) % 2
    y_pred = y_true.copy()
    wrong = int(BOOTSTRAP_SAMPLES * (1 - BOOTSTRAP_ACCURACY))
    y_pred[:wrong] = 1 - y_pred[:wrong]
    return y_true, y_pred


@pytest.fixture
def headless_matplotlib():
    """Render without a display and close every figure a test leaves open."""
    matplotlib.use("Agg", force=True)
    yield
    plt.close("all")


class TestEvaluatorWithTorchModels:
    """Test evaluating a torch model directly, the way the user guide shows it."""

    def test_default_metrics_work_on_a_model_and_loader(self, multiclass_loader):
        """The defaults include AUC, which needs probabilities where the model gives logits."""
        model = _SmallClassifier().eval()
        signals, labels = multiclass_loader.dataset.tensors
        with torch.no_grad():
            probabilities = torch.softmax(model(signals), dim=1).numpy()

        results = ECGEvaluator().evaluate(model, multiclass_loader)

        assert set(results) == {"accuracy", "precision", "recall", "f1", "auc"}
        expected_auc = roc_auc_score(labels.numpy(), probabilities, multi_class="ovr")
        assert results["auc"] == pytest.approx(expected_auc, abs=1e-4)
        expected_accuracy = (probabilities.argmax(axis=1) == labels.numpy()).mean()
        assert results["accuracy"] == pytest.approx(expected_accuracy)

    def test_the_model_is_handed_back_as_it_came(self, multiclass_loader):
        """Evaluation must not move the model, change its mode or touch its statistics."""
        model = _SmallClassifier()
        model.train()
        running_mean = model.norm.running_mean.clone()

        ECGEvaluator(metrics=["accuracy"]).evaluate(model, multiclass_loader)

        assert model.training
        assert next(model.parameters()).device.type == "cpu"
        assert torch.equal(model.norm.running_mean, running_mean)
        assert model.norm.num_batches_tracked.item() == 0

    def test_a_single_logit_is_thresholded_rather_than_argmaxed(self, binary_loader):
        """The binary models have one output, and an argmax over one column is always zero."""
        evaluator = ECGEvaluator(metrics=["accuracy", "auc"])

        results = evaluator.evaluate(_SignLogit(), binary_loader)

        assert results["accuracy"] == 1.0
        assert results["auc"] == 1.0

    def test_probabilities_are_not_softmaxed_twice(self, multiclass_loader):
        """A model that already emits probabilities keeps them."""
        model = _SoftmaxHead().eval()
        signals, _ = multiclass_loader.dataset.tensors
        with torch.no_grad():
            expected = model(signals).numpy()

        _, predictions = ECGEvaluator(metrics=["accuracy"]).evaluate(
            model, multiclass_loader, return_predictions=True
        )

        np.testing.assert_allclose(predictions, expected, atol=1e-5)

    def test_regression_outputs_are_left_alone(self):
        """Only the outputs of a classifier are logits."""
        signals = torch.randn(EVALUATOR_SAMPLES, 1, EVALUATOR_SIGNAL_LENGTH)
        targets = 5.0 * signals.mean(dim=(1, 2))
        loader = DataLoader(TensorDataset(signals, targets), batch_size=16)
        evaluator = ECGEvaluator(task_type="regression", metrics=["mse"])

        results = evaluator.evaluate(_SignLogit(), loader)

        assert results["mse"] == pytest.approx(0.0, abs=1e-8)

    def test_arrays_reach_a_torch_model(self, binary_loader):
        """Signals given as numpy arrays are converted rather than passed to the module raw."""
        signals, labels = binary_loader.dataset.tensors
        evaluator = ECGEvaluator(metrics=["accuracy"])

        results = evaluator.evaluate(_SignLogit(), (signals.numpy(), labels.numpy()))

        assert results["accuracy"] == 1.0

    def test_report_from_a_model_calibrates_probabilities(self, multiclass_loader):
        """Calibration computed on raw logits gave a Brier score above its maximum."""
        report = ECGEvaluator(metrics=["accuracy"]).generate_report(
            _SmallClassifier().eval(), multiclass_loader
        )

        values = dict(zip(report["Metric"], report["Value"], strict=False))
        assert 0.0 <= values["brier"] <= 1.0

    def test_inputs_without_targets_are_refused(self):
        """They used to be scored against targets of all zeros."""
        with pytest.raises(ValueError, match="targets"):
            ECGEvaluator(metrics=["accuracy"]).evaluate(
                _SignLogit(), np.zeros((4, 1, EVALUATOR_SIGNAL_LENGTH), dtype=np.float32)
            )

    def test_a_large_array_goes_through_the_model_in_batches(self):
        """One forward pass over a whole split can need more memory than there is."""
        seen = []

        class Recording(_SignLogit):
            def forward(self, x):
                seen.append(len(x))
                return super().forward(x)

        signals = torch.randn(150, 1, EVALUATOR_SIGNAL_LENGTH)
        labels = (signals.mean(dim=(1, 2)) > 0).long()

        results = ECGEvaluator(metrics=["accuracy"]).evaluate(Recording(), (signals, labels))

        assert seen == [64, 64, 22]
        assert results["accuracy"] == 1.0

    def test_unknown_metric_names_are_rejected(self):
        """A misspelt metric used to be dropped from the results without a word."""
        with pytest.raises(ValueError, match="auroc"):
            ECGEvaluator(metrics=["accuracy", "auroc"])

    def test_something_that_is_not_a_model_is_rejected(self):
        """Neither a model nor predictions leaves nothing to evaluate."""
        with pytest.raises(TypeError, match="model"):
            ECGEvaluator(metrics=["accuracy"]).evaluate(None, np.zeros((4, 1, 10)))

    def test_multiclass_auc_from_precomputed_probabilities(self, calibrated_multiclass_predictions):
        """AUC used to be computed for two classes only and raise for more."""
        y_true, y_prob, _ = calibrated_multiclass_predictions
        evaluator = ECGEvaluator(metrics=["auc"])

        results = evaluator.evaluate(y_prob.argmax(axis=1), y_true, y_scores=y_prob)

        assert results["auc"] == pytest.approx(roc_auc_score(y_true, y_prob, multi_class="ovr"))


class TestEvaluatorBootstrap:
    """Test bootstrap confidence intervals."""

    def test_bootstrap_accepts_precomputed_predictions(self, labelled_predictions):
        """evaluate takes predictions in place of a model, and so must bootstrap_evaluate."""
        y_true, y_pred = labelled_predictions
        evaluator = ECGEvaluator(metrics=["accuracy", "f1"], task_type="classification")

        intervals = evaluator.bootstrap_evaluate(y_pred, (y_pred, y_true), n_bootstrap=200)

        assert set(intervals) == {"accuracy", "f1"}
        accuracy = intervals["accuracy"]
        assert accuracy["lower_ci"] < BOOTSTRAP_ACCURACY < accuracy["upper_ci"]

    def test_bootstrap_auc_is_computed_from_the_scores(self):
        """Hard labels have an AUC of their own, well below that of the scores behind them."""
        rng = np.random.default_rng(0)
        y_true = np.arange(400) % 2
        y_scores = np.clip(0.5 + 0.25 * (2 * y_true - 1) + rng.normal(0, 0.25, 400), 0, 1)
        y_pred = (y_scores > 0.5).astype(int)
        score_auc = roc_auc_score(y_true, y_scores)
        assert roc_auc_score(y_true, y_pred) < score_auc - 0.03

        intervals = ECGEvaluator(metrics=["auc"]).bootstrap_evaluate(
            y_pred, y_true, n_bootstrap=200, y_scores=y_scores
        )

        assert intervals["auc"]["mean"] == pytest.approx(score_auc, abs=0.02)

    def test_bootstrap_auc_needs_scores_when_given_labels(self, labelled_predictions):
        """An AUC over hard labels would be reported as though it were the real one."""
        y_true, y_pred = labelled_predictions

        with pytest.raises(ValueError, match="y_scores"):
            ECGEvaluator(metrics=["auc"]).bootstrap_evaluate(y_pred, y_true, n_bootstrap=10)

    def test_resamples_where_a_metric_is_undefined_are_left_out(self):
        """With one positive in ten, a third of the resamples have none and no AUC."""
        y_true = np.array([0] * 9 + [1])
        y_scores = np.linspace(0.05, 0.95, 10)
        y_pred = (y_scores > 0.5).astype(int)

        intervals = ECGEvaluator(metrics=["auc"]).bootstrap_evaluate(
            y_pred, y_true, n_bootstrap=100, y_scores=y_scores
        )

        assert intervals["auc"]["mean"] == 1.0

    def test_generate_report_carries_confidence_intervals(self, labelled_predictions):
        """A failed bootstrap used to be swallowed, leaving the report without intervals."""
        y_true, y_pred = labelled_predictions
        evaluator = ECGEvaluator(metrics=["accuracy"], task_type="classification")

        report = evaluator.generate_report(y_pred, y_true)

        row = report.set_index("Metric").loc["accuracy"]
        assert row["95% CI Lower"] < row["Value"] < row["95% CI Upper"]

    def test_report_from_hard_labels_has_no_calibration_rows(self, labelled_predictions):
        """Labels carry no confidence, so there is nothing to calibrate."""
        y_true, y_pred = labelled_predictions
        evaluator = ECGEvaluator(metrics=["accuracy"], task_type="classification")

        report = evaluator.generate_report(y_pred, y_true)

        assert list(report["Metric"]) == ["accuracy"]


class TestPlotEcgSignals:
    """Test that a recording is drawn one row per lead, whichever way round it arrives."""

    @pytest.fixture
    def subplot_rows(self, monkeypatch) -> list[int]:
        """Record how many rows each figure is created with."""
        rows: list[int] = []
        create = plt.subplots

        def recording_subplots(nrows, *args, **kwargs):
            rows.append(nrows)
            return create(nrows, *args, **kwargs)

        monkeypatch.setattr(plt, "subplots", recording_subplots)
        return rows

    @pytest.mark.parametrize(
        ("shape", "expected_rows"),
        [((2, 500), 2), ((500, 2), 2), ((1, 3000), 1), ((3000,), 1)],
    )
    def test_one_row_per_lead_in_either_layout(
        self, shape, expected_rows, subplot_rows, headless_matplotlib, temp_dir
    ):
        """A (leads, time) sample used to be drawn as thousands of one-point leads."""
        path = temp_dir / "ecg.png"

        plot_ecg_signals(np.random.randn(*shape), sampling_rate=300, save_path=str(path))

        assert subplot_rows == [expected_rows]
        assert path.exists()
        assert plt.get_fignums() == []

    def test_a_dataset_sample_can_be_passed_as_a_tensor(
        self, subplot_rows, headless_matplotlib, temp_dir
    ):
        """Datasets return tensors, and the plot should not need them converted first."""
        plot_ecg_signals(torch.randn(1, 3000), save_path=str(temp_dir / "ecg.png"))

        assert subplot_rows == [1]

    def test_a_batch_is_rejected(self, headless_matplotlib):
        """One figure shows one recording."""
        with pytest.raises(ValueError, match="one recording"):
            plot_ecg_signals(np.zeros((4, 1, 500)))

    def test_lead_names_must_match_the_leads(self, headless_matplotlib):
        """Too few names used to surface as an IndexError part-way through drawing."""
        with pytest.raises(ValueError, match="lead names"):
            plot_ecg_signals(np.zeros((2, 500)), leads=["I"])


class TestPlotOutput:
    """Test where the evaluation plots are written and what they accept."""

    def test_plots_create_the_folder_they_save_into(self, headless_matplotlib, temp_dir):
        """A save path in a folder that does not exist yet used to raise FileNotFoundError."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        path = temp_dir / "plots" / "nested" / "cm.png"

        plot_confusion_matrix(y_true, y_pred, save_path=str(path))

        assert path.exists()

    def test_calibration_and_training_curves_create_their_folder(
        self, headless_matplotlib, temp_dir, calibrated_binary_predictions
    ):
        """These two save several files into a directory rather than to one path."""
        y_true, y_prob, _ = calibrated_binary_predictions
        metrics_path = temp_dir / "metrics.csv"
        metrics_path.write_text(
            "epoch,train_loss,val_loss,train_acc,val_acc\n1,0.9,0.8,0.5,0.6\n2,0.7,0.75,0.6,0.65\n"
        )
        plots = temp_dir / "plots"

        plot_calibration_curve(y_true, y_prob, save_dir=str(plots / "calibration"))
        plot_training_curves(str(metrics_path), save_dir=str(plots / "training"))

        assert (plots / "calibration" / "calibration_curve.png").exists()
        assert (plots / "calibration" / "prediction_distribution.png").exists()
        assert (plots / "training" / "loss.png").exists()
        assert (plots / "training" / "accuracy.png").exists()

    def test_roc_curve_takes_the_two_column_matrix_of_a_binary_model(
        self, headless_matplotlib, temp_dir, calibrated_binary_predictions
    ):
        """get_test_results returns [1 - p, p] for a binary model, not the positive column."""
        y_true, y_prob, _ = calibrated_binary_predictions
        path = temp_dir / "roc.png"

        plot_roc_curve(y_true, y_prob, save_path=str(path))

        assert path.exists()

    def test_roc_curve_rejects_more_than_two_classes(
        self, headless_matplotlib, calibrated_multiclass_predictions
    ):
        """The message has to say what to pass instead; scikit-learn's does not."""
        y_true, y_prob, _ = calibrated_multiclass_predictions

        with pytest.raises(ValueError, match="two-class"):
            plot_roc_curve(y_true, y_prob)


class _LinearClassifier(torch.nn.Module):
    """The smallest model that can learn which way a signal is offset."""

    def __init__(self, n_outputs: int = 2):
        super().__init__()
        self.head = torch.nn.Linear(EVALUATOR_SIGNAL_LENGTH, n_outputs)

    def forward(self, x):
        return self.head(x.flatten(1))


@pytest.fixture
def separable_arrays(binary_loader) -> tuple[np.ndarray, np.ndarray]:
    """The binary signals and labels as arrays: positive mean exactly when the label is one."""
    signals, labels = binary_loader.dataset.tensors
    return signals.numpy(), labels.numpy()


class TestMultiLabelMetrics:
    """Test metrics over multi-hot targets, defined as the evaluate command reports them."""

    def test_metrics_match_their_definitions(self, calibrated_multilabel_predictions):
        from sklearn.metrics import precision_recall_fscore_support

        y_true, y_prob, _ = calibrated_multilabel_predictions
        predicted = (y_prob > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="macro", zero_division=0
        )

        metrics = calculate_classification_metrics(y_true, y_prob.astype(np.float32))

        assert metrics["accuracy"] == pytest.approx((predicted == y_true).mean())
        assert metrics["exact_match_accuracy"] == pytest.approx(
            (predicted == y_true).all(axis=1).mean()
        )
        assert (metrics["precision"], metrics["recall"], metrics["f1"]) == pytest.approx(
            (precision, recall, f1)
        )
        assert metrics["auc"] == pytest.approx(roc_auc_score(y_true, y_prob, average="macro"))

    def test_mcc_is_reported_as_undefined(self, calibrated_multilabel_predictions):
        y_true, y_prob, _ = calibrated_multilabel_predictions

        metrics = calculate_classification_metrics(y_true, y_prob, metrics=["mcc"])

        assert np.isnan(metrics["mcc"])

    def test_the_evaluator_scores_a_multi_label_model(self):
        """Multi-hot targets used to come back as NaN with a warning per metric."""
        signals = torch.randn(EVALUATOR_SAMPLES, 1, EVALUATOR_SIGNAL_LENGTH)
        targets = (torch.rand(EVALUATOR_SAMPLES, 3) > 0.5).float()
        loader = DataLoader(TensorDataset(signals, targets), batch_size=16)
        model = _LinearClassifier(n_outputs=3).eval()
        with torch.no_grad():
            predicted = (torch.sigmoid(model(signals)) > 0.5).float()
        evaluator = ECGEvaluator(metrics=["accuracy", "exact_match_accuracy", "f1", "auc"])

        results = evaluator.evaluate(model, loader)

        assert results["accuracy"] == pytest.approx((predicted == targets).float().mean().item())
        assert results["exact_match_accuracy"] == pytest.approx(
            (predicted == targets).all(dim=1).float().mean().item()
        )
        assert 0.0 <= results["auc"] <= 1.0

    def test_bootstrap_resamples_multi_label_rows(self, calibrated_multilabel_predictions):
        y_true, y_prob, _ = calibrated_multilabel_predictions
        evaluator = ECGEvaluator(metrics=["accuracy", "f1"])
        point = evaluator.evaluate(y_prob, y_true)

        intervals = evaluator.bootstrap_evaluate(y_prob, y_true, n_bootstrap=50)

        for name in ("accuracy", "f1"):
            assert intervals[name]["lower_ci"] <= point[name] <= intervals[name]["upper_ci"]


class TestCrossValidation:
    """Test that cross-validation fits a model per fold and scores it on the fold held out."""

    def test_a_torch_model_is_trained_on_each_fold(self, separable_arrays):
        """An untrained model used to be scored, which is chance on a learnable problem."""
        evaluator = ECGEvaluator(metrics=["accuracy"], device="cpu")

        scores = evaluator.cross_validate(
            _LinearClassifier,
            separable_arrays,
            k_folds=4,
            epochs=20,
            batch_size=8,
            train_config={"learning_rate": 0.01},
        )

        assert len(scores["accuracy"]) == 4
        assert min(scores["accuracy"]) > 0.9

    def test_a_scikit_learn_estimator_is_fitted_and_asked_for_probabilities(self, separable_arrays):
        """An estimator is not callable, so calling it as a model used to fail."""
        from sklearn.linear_model import LogisticRegression

        signals, labels = separable_arrays
        evaluator = ECGEvaluator(metrics=["accuracy", "auc"])

        scores = evaluator.cross_validate(
            LogisticRegression, (signals.reshape(len(signals), -1), labels), k_folds=4
        )

        assert scores["accuracy"] == [1.0] * 4
        assert scores["auc"] == [1.0] * 4

    def test_groups_are_never_split_between_training_and_scoring(self):
        """Segments of one recording on both sides would flatter the score."""
        groups = np.repeat(np.arange(8), 5)

        for train_idx, val_idx in ECGEvaluator._folds(len(groups), 4, groups, seed=0):
            assert not set(groups[train_idx]) & set(groups[val_idx])

    def test_grouped_cross_validation_trains_and_scores(self, separable_arrays):
        signals, labels = separable_arrays
        groups = np.arange(len(labels)) // 4
        evaluator = ECGEvaluator(metrics=["accuracy"], device="cpu")

        scores = evaluator.cross_validate(
            _LinearClassifier,
            (signals, labels),
            k_folds=4,
            groups=groups,
            epochs=20,
            batch_size=8,
            train_config={"learning_rate": 0.01},
        )

        assert min(scores["accuracy"]) > 0.9

    def test_data_must_be_a_pair(self):
        with pytest.raises(ValueError, match=r"\(X, y\)"):
            ECGEvaluator(metrics=["accuracy"]).cross_validate(_LinearClassifier, np.zeros((4, 1)))
