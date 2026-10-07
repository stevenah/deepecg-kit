"""
Main ECG model evaluator class.
"""

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from deepecgkit.evaluation.calibration import calibration_report
from deepecgkit.evaluation.metrics import (
    CALIBRATION_REPORT_KEYS,
    DEFAULT_CALIBRATION_BINS,
    calculate_classification_metrics,
    calculate_regression_metrics,
)
from deepecgkit.utils.device import resolve_device

logger = logging.getLogger(__name__)

MAX_CLASSIFICATION_CLASSES = 20

CALIBRATION_ROW_ORDER = (*CALIBRATION_REPORT_KEYS, "calibration_bins")

#: Metric names the evaluator accepts, by the function that computes them.
CLASSIFICATION_METRICS = frozenset({"accuracy", "precision", "recall", "f1", "mcc"})
REGRESSION_METRICS = frozenset(
    {"mse", "mean_squared_error", "mae", "mean_absolute_error", "r2", "r2_score", "rmse"}
)
SCORE_METRICS = frozenset({"auc"})
SUPPORTED_METRICS = CLASSIFICATION_METRICS | REGRESSION_METRICS | SCORE_METRICS

PROBABILITY_SUM_TOLERANCE = 1e-4


class ECGEvaluator:
    """
    Comprehensive evaluator for ECG models.

    This class provides a unified interface for evaluating ECG models with
    various metrics and analysis tools.

    It takes either a model together with its data, or predictions computed
    elsewhere together with their targets. A torch model is run on the
    evaluation device in eval mode and handed back on the device and in the
    mode it came in. Its outputs are taken to be logits, as every model in the
    package emits and as ECGTrainer assumes, and are turned into probabilities
    for a classification task: a single output becomes the two columns
    [1 - p, p], several outputs go through a softmax, or through a sigmoid
    when the targets are multi-hot. Rows that already sum to one are left
    alone. Predictions passed in directly are used as they are.

    Args:
        metrics: List of metrics to compute: accuracy, precision, recall, f1,
            mcc and auc for classification; mse, mae, r2 and rmse for
            regression
        task_type: Type of task ("classification", "regression", "auto")
        device: Device for model evaluation
        calibration_bins: Number of bins for the calibration metrics reported
            by generate_report. ECE depends on this, so the value is emitted
            alongside the metrics rather than left implicit.

    Raises:
        ValueError: If metrics names one the evaluator does not compute.

    Examples:
        >>> evaluator = ECGEvaluator(metrics=["accuracy", "auc", "f1"])
        >>> results = evaluator.evaluate(model, test_data)
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        task_type: str = "auto",
        device: str = "auto",
        calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    ):
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1", "auc"]
        unknown = [name for name in self.metrics if name not in SUPPORTED_METRICS]
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {', '.join(unknown)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}."
            )
        self.task_type = task_type
        self.calibration_bins = calibration_bins
        self.y_scores: np.ndarray | None = None
        self.y_true: np.ndarray | None = None

        self.device = resolve_device(device)

    def evaluate(
        self,
        model: torch.nn.Module | np.ndarray | None = None,
        test_data: torch.utils.data.DataLoader | np.ndarray | tuple | None = None,
        return_predictions: bool = False,
        y_scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model: PyTorch model or numpy array of predictions
            test_data: Test data loader or (predictions, targets) tuple
            return_predictions: Whether to return predictions along with metrics
            y_scores: Scores/probabilities for AUC calculation (optional)

        Returns:
            Dictionary of metric values
        """

        self.y_scores = y_scores

        predictions, targets = self._process_input_data(model, test_data)
        self.y_true = targets
        metrics = self._calculate_metrics(predictions, targets, self.y_scores)

        if return_predictions:
            return metrics, predictions
        return metrics

    def _process_input_data(
        self,
        model: torch.nn.Module | np.ndarray | None,
        test_data: torch.utils.data.DataLoader | np.ndarray | tuple | None,
    ) -> tuple:
        """Process input data to get predictions and targets."""
        if isinstance(model, np.ndarray):
            return self._process_numpy_input(model, test_data)
        return self._get_predictions(model, test_data)

    def _process_numpy_input(
        self,
        model: np.ndarray,
        test_data: torch.utils.data.DataLoader | np.ndarray | tuple | None,
    ) -> tuple:
        """Process input when model is a numpy array."""
        predictions = model
        if not isinstance(test_data, tuple):
            return predictions, test_data

        if len(test_data) == 3:
            targets = test_data[1]
            self.y_scores = test_data[2]
        elif len(test_data) == 2:
            targets = test_data[1]
        else:
            targets = test_data
        return predictions, targets

    def _calculate_metric(
        self,
        name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> float:
        """Calculate one metric. Evaluation and bootstrapping both come through here."""
        if name in CLASSIFICATION_METRICS:
            return calculate_classification_metrics(targets, predictions, metrics=[name])[name]
        if name in REGRESSION_METRICS:
            return calculate_regression_metrics(targets, predictions, metrics=[name])[name]

        # Per-class predictions are their own scores; hard labels are not scores at all.
        if scores is None and np.ndim(predictions) > 1:
            scores = predictions
        if scores is None:
            raise ValueError("y_scores required for AUC calculation")
        return calculate_classification_metrics(targets, scores, metrics=[name])[name]

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Calculate metrics based on predictions and targets."""
        return {
            name: self._calculate_metric(name, predictions, targets, scores)
            for name in self.metrics
        }

    def _get_predictions(
        self,
        model: torch.nn.Module | Any | None,
        test_data: torch.utils.data.DataLoader | np.ndarray | tuple | None,
    ) -> tuple:
        """Extract predictions and targets from model and data."""
        if model is None and isinstance(test_data, tuple):
            return self._handle_tuple_test_data(test_data)

        if not callable(model):
            raise TypeError(
                f"Expected a model to call or an array of predictions, got {type(model).__name__}."
            )

        with self._evaluating(model):
            if isinstance(test_data, torch.utils.data.DataLoader):
                outputs, targets = self._process_dataloader(model, test_data)
            elif isinstance(test_data, tuple) and len(test_data) == 2:
                outputs, targets = self._process_tuple_data(model, test_data)
            elif isinstance(test_data, np.ndarray):
                outputs, targets = self._process_numpy_data(model, test_data)
            else:
                raise ValueError(f"Unsupported test data type: {type(test_data)}")

        if isinstance(model, torch.nn.Module) and self._is_classification(targets):
            outputs = self._as_probabilities(outputs, targets)
        return outputs, targets

    @contextlib.contextmanager
    def _evaluating(self, model: Any) -> Iterator[None]:
        """Hold a torch module on the evaluation device and in eval mode.

        The inputs are moved to the evaluation device, so the model has to be
        there too; and a model left in training mode would be measured with
        dropout active and would have its batch-norm statistics overwritten by
        the test data. Both are put back afterwards.
        """
        if not isinstance(model, torch.nn.Module):
            yield
            return

        was_training = model.training
        parameter = next(model.parameters(), None)
        previous_device = None if parameter is None else parameter.device
        model.to(self.device).eval()
        try:
            yield
        finally:
            model.train(was_training)
            if previous_device is not None:
                model.to(previous_device)

    def _predict(self, model: Any, x: Any) -> Any:
        """Run one batch through the model, as numpy when the model returns a tensor."""
        if isinstance(model, torch.nn.Module) and not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        if torch.is_tensor(x):
            x = x.to(self.device)

        with torch.no_grad():
            pred = model(x)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        return pred

    def _is_classification(self, targets: np.ndarray | None) -> bool:
        """Whether the evaluated task is classification, resolving "auto" from the targets."""
        if self.task_type != "auto":
            return self.task_type == "classification"
        if set(self.metrics) <= REGRESSION_METRICS:
            return False
        return self._detect_task_type(targets) == "classification"

    def _as_probabilities(self, outputs: np.ndarray, targets: np.ndarray | None) -> np.ndarray:
        """Turn the logits of a classifier into probabilities, the way ECGTrainer does."""
        logits = torch.as_tensor(np.asarray(outputs))
        if not torch.is_floating_point(logits) or logits.ndim > 2:
            return outputs

        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.ndim == 1:
            positive = torch.sigmoid(logits)
            return torch.stack([1 - positive, positive], dim=1).numpy()

        if targets is not None and np.ndim(targets) > 1:
            return torch.sigmoid(logits).numpy()

        row_sums = logits.sum(dim=1)
        sums_to_one = torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=PROBABILITY_SUM_TOLERANCE
        )
        if sums_to_one and bool((logits >= 0).all()):
            return outputs
        return torch.softmax(logits, dim=1).numpy()

    def _handle_tuple_test_data(self, test_data: tuple) -> tuple:
        """Handle test data when it's a tuple."""
        if len(test_data) == 3:
            y_true, y_pred, y_scores = test_data
            self.y_scores = y_scores
            return y_pred, y_true
        elif len(test_data) == 2:
            return test_data[0], test_data[1]
        return test_data, None

    def _process_dataloader(
        self, model: torch.nn.Module, test_data: torch.utils.data.DataLoader
    ) -> tuple:
        """Process data from a DataLoader."""
        predictions = []
        targets = []
        for batch in test_data:
            if isinstance(batch, list | tuple) and len(batch) == 2:
                x, y = batch
            else:
                x, y = batch, None

            predictions.append(self._predict(model, x))

            if y is not None:
                if torch.is_tensor(y):
                    y = y.cpu().numpy()
                targets.append(y)

        return np.concatenate(predictions), np.concatenate(targets)

    def _process_tuple_data(self, model: torch.nn.Module, test_data: tuple) -> tuple:
        """Process data when it's a tuple."""
        x, y = test_data
        pred = self._predict(model, x)
        if torch.is_tensor(y):
            y = y.cpu().numpy()

        return pred, y

    def _process_numpy_data(self, model: torch.nn.Module, test_data: np.ndarray) -> tuple:
        """Process data when it's a numpy array."""
        pred = self._predict(model, torch.tensor(test_data, dtype=torch.float32))
        return pred, np.zeros(len(pred))

    def _calibration_rows(
        self, predictions: np.ndarray, calibration_fit: tuple | None
    ) -> list[dict[str, Any]]:
        """Build the calibration rows appended to the report.

        Rows whose value is unavailable are omitted rather than emitted as NaN,
        so a report produced without a fitting split simply carries no
        temperature rows instead of carrying empty ones.

        Args:
            predictions: Model output for the evaluated split, used as the
                probability matrix when no explicit y_scores were supplied
            calibration_fit: Optional (targets, logits) pair from the
                validation split, used to fit the temperature

        Returns:
            A list of {"Metric", "Value"} rows, possibly empty.
        """
        if self.y_true is None:
            return []

        scores = self.y_scores if self.y_scores is not None else predictions
        if scores is None or not np.issubdtype(np.asarray(scores).dtype, np.floating):
            # Hard labels carry no confidence to calibrate.
            return []

        fit_targets, fit_logits = calibration_fit or (None, None)
        try:
            report = calibration_report(
                self.y_true,
                scores,
                n_bins=self.calibration_bins,
                fit_targets=fit_targets,
                fit_logits=fit_logits,
            )
        except Exception as error:
            logger.warning(f"Could not compute calibration metrics: {error}")
            return []

        rows: list[dict[str, Any]] = []
        for name in CALIBRATION_ROW_ORDER:
            value = report.get(name)
            if value is None:
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            rows.append({"Metric": name, "Value": value})
        return rows

    def _detect_task_type(self, targets: np.ndarray | None) -> str:
        """Detect whether task is classification or regression."""
        if targets is None:
            return "regression"

        unique_values = np.unique(targets)

        if len(unique_values) <= MAX_CLASSIFICATION_CLASSES and np.all(
            np.equal(np.mod(targets, 1), 0)
        ):
            return "classification"
        return "regression"

    def cross_validate(
        self, model_class: type, data: Any, k_folds: int = 5, **model_kwargs: Any
    ) -> dict[str, list[float]]:
        """
        Perform k-fold cross-validation.

        Args:
            model_class: Class of model to evaluate
            data: Dataset for cross-validation
            k_folds: Number of folds
            **model_kwargs: Keyword arguments for model initialization

        Returns:
            Dictionary of metric scores across folds
        """
        if isinstance(data, tuple):
            x, y = data
        else:
            raise ValueError("Data must be (X, y) tuple for cross-validation")

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results: dict[str, list[float]] = {metric: [] for metric in self.metrics}

        for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
            logger.info(f"Evaluating fold {fold + 1}/{k_folds}")

            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(**model_kwargs)

            if hasattr(model, "fit"):
                model.fit(x_train, y_train)

            results = self.evaluate(model, (x_val, y_val))

            for metric in self.metrics:
                if metric in results:
                    fold_results[metric].append(results[metric])

        return fold_results

    def bootstrap_evaluate(
        self,
        model: Any,
        test_data: Any,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        y_scores: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Bootstrap evaluation for confidence intervals.

        Takes the same inputs as evaluate, and computes each metric the same
        way, so an interval describes the value evaluate reports. Resampling
        draws from numpy's global generator; seed it for repeatable intervals.

        Args:
            model: PyTorch model or numpy array of predictions
            test_data: Test data loader or (predictions, targets) tuple
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            y_scores: Scores/probabilities for AUC calculation (optional)

        Returns:
            Dictionary with mean, std, and confidence intervals for each metric
        """
        self.y_scores = y_scores
        predictions, targets = self._process_input_data(model, test_data)
        return self._bootstrap(predictions, targets, self.y_scores, n_bootstrap, confidence_level)

    def _bootstrap(
        self,
        predictions: np.ndarray,
        targets: np.ndarray | None,
        scores: np.ndarray | None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict[str, dict[str, float]]:
        """Resample predictions with their targets and summarise each metric."""
        if targets is None:
            raise ValueError("Bootstrap evaluation requires ground truth targets")

        predictions, targets = np.asarray(predictions), np.asarray(targets)
        scores = None if scores is None else np.asarray(scores)

        # A metric that is undefined on the whole set is undefined on every resample.
        point_estimates = self._calculate_metrics(predictions, targets, scores)
        names = [name for name, value in point_estimates.items() if not np.isnan(value)]

        n_samples = len(targets)
        bootstrap_results: dict[str, list[float]] = {name: [] for name in names}

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_predictions, boot_targets = predictions[indices], targets[indices]
            boot_scores = None if scores is None else scores[indices]
            for name in names:
                bootstrap_results[name].append(
                    self._calculate_metric(name, boot_predictions, boot_targets, boot_scores)
                )

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        final_results = {}
        for metric, values in bootstrap_results.items():
            # A resample can lack a class and leave a metric undefined; those are left out.
            defined = [value for value in values if not np.isnan(value)]
            if defined:
                final_results[metric] = {
                    "mean": np.mean(defined),
                    "std": np.std(defined),
                    "lower_ci": np.percentile(defined, lower_percentile),
                    "upper_ci": np.percentile(defined, upper_percentile),
                }

        return final_results

    def generate_report(
        self,
        model: Any,
        test_data: Any,
        save_path: str | None = None,
        y_scores: np.ndarray | None = None,
        calibration_fit: tuple | None = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.

        Calibration is always reported: expected calibration error, its
        equal-mass variant, maximum calibration error and the Brier score. Pass
        calibration_fit to additionally fit a temperature and report what
        rescaling recovers, which adds a "temperature" row and the "_temp"
        counterparts of each calibration metric.

        Args:
            model: Trained model
            test_data: Test dataset
            save_path: Path to save report (optional)
            y_scores: Scores/probabilities for AUC calculation (optional)
            calibration_fit: Optional (targets, logits) pair from a held-out
                validation split, used to fit the temperature. Never pass the
                split being reported on. Callers holding probabilities rather
                than logits can convert with
                deepecgkit.evaluation.probabilities_to_logits.

        Returns:
            DataFrame with evaluation results
        """

        results = self.evaluate(model, test_data, return_predictions=True, y_scores=y_scores)
        if isinstance(results, tuple):
            metrics_dict, predictions = results[0], results[1]
        else:
            metrics_dict, predictions = results, None

        try:
            bootstrap_results = self._bootstrap(predictions, self.y_true, self.y_scores)
        except Exception as error:
            logger.warning(f"Could not compute bootstrap confidence intervals: {error}")
            bootstrap_results = {}

        report_data = []

        for metric in self.metrics:
            if metric in metrics_dict and metric not in {"predictions", "targets"}:
                row = {
                    "Metric": metric,
                    "Value": metrics_dict[metric],
                }

                if metric in bootstrap_results:
                    row.update(
                        {
                            "Std": bootstrap_results[metric]["std"],
                            "95% CI Lower": bootstrap_results[metric]["lower_ci"],
                            "95% CI Upper": bootstrap_results[metric]["upper_ci"],
                        }
                    )

                report_data.append(row)

        report_data.extend(self._calibration_rows(predictions, calibration_fit))

        report_df = pd.DataFrame(report_data)

        if save_path:
            report_df.to_csv(save_path, index=False)
            logger.info(f"Report saved to {save_path}")

        return report_df
