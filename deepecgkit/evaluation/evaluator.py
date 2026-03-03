"""
Main ECG model evaluator class.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from deepecgkit.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)

MAX_CLASSIFICATION_CLASSES = 20


class ECGEvaluator:
    """
    Comprehensive evaluator for ECG models.

    This class provides a unified interface for evaluating ECG models with
    various metrics and analysis tools.

    Args:
        metrics: List of metrics to compute
        task_type: Type of task ("classification", "regression", "auto")
        device: Device for model evaluation

    Examples:
        >>> evaluator = ECGEvaluator(metrics=["accuracy", "auc", "f1"])
        >>> results = evaluator.evaluate(model, test_data)
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        task_type: str = "auto",
        device: str = "auto",
    ):
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1", "auc"]
        self.task_type = task_type

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def evaluate(
        self,
        model: Optional[Union[torch.nn.Module, np.ndarray]] = None,
        test_data: Optional[Union[torch.utils.data.DataLoader, np.ndarray, tuple]] = None,
        return_predictions: bool = False,
        y_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
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
        metrics = self._calculate_metrics(predictions, targets)

        if return_predictions:
            return metrics, predictions
        return metrics

    def _process_input_data(
        self,
        model: Optional[Union[torch.nn.Module, np.ndarray]],
        test_data: Optional[Union[torch.utils.data.DataLoader, np.ndarray, tuple]],
    ) -> tuple:
        """Process input data to get predictions and targets."""
        if isinstance(model, np.ndarray):
            return self._process_numpy_input(model, test_data)
        return self._get_predictions(model, test_data)

    def _process_numpy_input(
        self,
        model: np.ndarray,
        test_data: Optional[Union[torch.utils.data.DataLoader, np.ndarray, tuple]],
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

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate metrics based on predictions and targets."""
        metrics = {}
        for metric_name in self.metrics:
            if metric_name in ["accuracy", "precision", "recall", "f1"]:
                metrics[metric_name] = calculate_classification_metrics(
                    targets, predictions, metrics=[metric_name]
                )[metric_name]
            elif metric_name in ["mse", "mae", "r2"]:
                metrics[metric_name] = calculate_regression_metrics(
                    targets, predictions, metrics=[metric_name]
                )[metric_name]
            elif metric_name == "auc":
                if self.y_scores is None:
                    raise ValueError("y_scores required for AUC calculation")
                scores = self.y_scores
                n_classes = len(np.unique(targets))
                if n_classes <= 2 and scores.ndim > 1:
                    scores = scores[:, 1]
                metrics[metric_name] = roc_auc_score(targets, scores)
        return metrics

    def _get_predictions(
        self,
        model: Optional[Union[torch.nn.Module, Any]],
        test_data: Optional[Union[torch.utils.data.DataLoader, np.ndarray, tuple]],
    ) -> tuple:
        """Extract predictions and targets from model and data."""
        if model is None and isinstance(test_data, tuple):
            return self._handle_tuple_test_data(test_data)

        if isinstance(test_data, torch.utils.data.DataLoader):
            return self._process_dataloader(model, test_data)
        elif isinstance(test_data, tuple) and len(test_data) == 2:
            return self._process_tuple_data(model, test_data)
        elif isinstance(test_data, np.ndarray):
            return self._process_numpy_data(model, test_data)
        else:
            raise ValueError(f"Unsupported test data type: {type(test_data)}")

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
        with torch.no_grad():
            for batch in test_data:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                else:
                    x, y = batch, None

                if torch.is_tensor(x):
                    x = x.to(self.device)

                pred = model(x)

                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()
                predictions.append(pred)

                if y is not None:
                    if torch.is_tensor(y):
                        y = y.cpu().numpy()
                    targets.append(y)

        return np.concatenate(predictions), np.concatenate(targets)

    def _process_tuple_data(self, model: torch.nn.Module, test_data: tuple) -> tuple:
        """Process data when it's a tuple."""
        x, y = test_data
        if torch.is_tensor(x):
            x = x.to(self.device)

        with torch.no_grad():
            pred = model(x)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(y):
            y = y.cpu().numpy()

        return pred, y

    def _process_numpy_data(self, model: torch.nn.Module, test_data: np.ndarray) -> tuple:
        """Process data when it's a numpy array."""
        x = torch.tensor(test_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = model(x)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        return pred, np.zeros(len(pred))

    def _detect_task_type(self, targets: np.ndarray) -> str:
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
        self, model_class: type, data: Any, k_folds: int = 5, **model_kwargs
    ) -> Dict[str, List[float]]:
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
        fold_results = {metric: [] for metric in self.metrics}

        for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
            print(f"Evaluating fold {fold + 1}/{k_folds}")

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
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap evaluation for confidence intervals.

        Args:
            model: Trained model
            test_data: Test dataset
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary with mean, std, and confidence intervals for each metric
        """
        predictions, targets = self._get_predictions(model, test_data)

        if targets is None:
            raise ValueError("Bootstrap evaluation requires ground truth targets")

        n_samples = len(targets)
        bootstrap_results = {metric: [] for metric in self.metrics}

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_predictions = predictions[indices]
            boot_targets = targets[indices]

            if (
                self.task_type == "classification"
                or self._detect_task_type(boot_targets) == "classification"
            ):
                results = calculate_classification_metrics(
                    boot_targets, boot_predictions, metrics=self.metrics
                )
            else:
                results = calculate_regression_metrics(
                    boot_targets, boot_predictions, metrics=self.metrics
                )

            for metric in self.metrics:
                if metric in results:
                    bootstrap_results[metric].append(results[metric])

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        final_results = {}
        for metric, values in bootstrap_results.items():
            if values:
                final_results[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "lower_ci": np.percentile(values, lower_percentile),
                    "upper_ci": np.percentile(values, upper_percentile),
                }

        return final_results

    def generate_report(
        self,
        model: Any,
        test_data: Any,
        save_path: Optional[str] = None,
        y_scores: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.

        Args:
            model: Trained model
            test_data: Test dataset
            save_path: Path to save report (optional)
            y_scores: Scores/probabilities for AUC calculation (optional)

        Returns:
            DataFrame with evaluation results
        """

        results = self.evaluate(model, test_data, return_predictions=True, y_scores=y_scores)
        if isinstance(results, tuple):
            metrics_dict = results[0]
        else:
            metrics_dict = results

        try:
            bootstrap_results = self.bootstrap_evaluate(model, test_data)
        except Exception:
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

        report_df = pd.DataFrame(report_data)

        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"Report saved to {save_path}")

        return report_df
