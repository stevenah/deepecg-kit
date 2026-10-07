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
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, TensorDataset

from deepecgkit.evaluation.calibration import calibration_report
from deepecgkit.evaluation.metrics import (
    ACCURACY_METRICS,
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
CLASSIFICATION_METRICS = frozenset({*ACCURACY_METRICS, "precision", "recall", "f1", "mcc"})
REGRESSION_METRICS = frozenset(
    {"mse", "mean_squared_error", "mae", "mean_absolute_error", "r2", "r2_score", "rmse"}
)
SCORE_METRICS = frozenset({"auc"})
SUPPORTED_METRICS = CLASSIFICATION_METRICS | REGRESSION_METRICS | SCORE_METRICS

PROBABILITY_SUM_TOLERANCE = 1e-4

#: Rows a torch model is given at a time when its inputs arrive as one array.
PREDICTION_BATCH_SIZE = 64

#: Share of each training fold held back for the learning-rate schedule and
#: early stopping, as one fold in this many.
INNER_VALIDATION_FOLDS = 10


class _ArrayLoaders:
    """What ECGTrainer asks of a data module, over arrays held in memory."""

    def __init__(self, train: tuple, tune: tuple, batch_size: int, seed: int) -> None:
        self._train = train
        self._tune = tune
        self._batch_size = batch_size
        self._seed = seed

    @staticmethod
    def _dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y))

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self._seed)
        return DataLoader(
            self._dataset(*self._train),
            batch_size=self._batch_size,
            shuffle=True,
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset(*self._tune), batch_size=self._batch_size)


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
            regression. Multi-hot targets are scored as
            calculate_classification_metrics describes, where
            exact_match_accuracy is also available.
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
        if targets is None:
            raise ValueError(
                "Evaluation needs the targets as well as the inputs: pass (inputs, targets), "
                "a DataLoader that yields both, or predictions together with their targets."
            )
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
        if isinstance(model, torch.nn.Module) and len(x) > PREDICTION_BATCH_SIZE:
            # A whole split in one forward pass can need more memory than there is.
            chunks = torch.split(x, PREDICTION_BATCH_SIZE)
            return np.concatenate([self._predict(model, chunk) for chunk in chunks])
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
        return pred, None

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
        self,
        model_class: type,
        data: Any,
        k_folds: int = 5,
        groups: np.ndarray | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        train_config: dict[str, Any] | None = None,
        seed: int = 42,
        **model_kwargs: Any,
    ) -> dict[str, list[float]]:
        """
        Perform k-fold cross-validation.

        A fresh model is built and fitted for every fold and scored on the
        fold held out. A torch module is trained with ECGTrainer; a tenth of
        its training fold is set aside for the learning-rate schedule and
        early stopping, so the held-out fold plays no part in training.
        Anything else with a fit method is fitted with fit(X, y) and asked for
        predict_proba, or failing that predict, unless it can be called
        directly.

        Args:
            model_class: Class of model to evaluate
            data: (X, y) tuple of arrays or tensors
            k_folds: Number of folds
            groups: Group of each sample, such as the recording or patient it
                comes from. Folds then never split a group, which segments cut
                from longer recordings need for the estimate to be honest.
            epochs: Maximum training epochs per fold, for a torch module
            batch_size: Batch size for training and scoring a torch module
            train_config: ECGTrainer settings for a torch module. Multi-hot
                targets and a regression task_type are recognised without it;
                a single-output binary model needs binary_classification.
            seed: Seed for shuffling the folds and the training batches
            **model_kwargs: Keyword arguments for model initialization

        Returns:
            Dictionary of metric scores across folds
        """
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Data must be (X, y) tuple for cross-validation")
        x, y = (np.asarray(part.cpu() if torch.is_tensor(part) else part) for part in data)
        if groups is not None:
            groups = np.asarray(groups)

        fold_results: dict[str, list[float]] = {metric: [] for metric in self.metrics}

        folds = self._folds(len(x), k_folds, groups, seed)
        for fold, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"Evaluating fold {fold + 1}/{k_folds}")
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            model = model_class(**model_kwargs)
            if isinstance(model, torch.nn.Module):
                fold_groups = None if groups is None else groups[train_idx]
                self._train_fold(
                    model, x_train, y_train, fold_groups, epochs, batch_size, train_config, seed
                )
                held_out = _ArrayLoaders((x_val, y_val), (x_val, y_val), batch_size, seed)
                results = self.evaluate(model, held_out.val_dataloader())
            else:
                results = self._fit_and_score(model, x_train, y_train, x_val, y_val)

            for metric in self.metrics:
                if metric in results:
                    fold_results[metric].append(results[metric])

        return fold_results

    @staticmethod
    def _folds(
        n_samples: int, n_splits: int, groups: np.ndarray | None, seed: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Split sample indices into folds, keeping each group whole when groups are given."""
        indices = np.arange(n_samples)
        folds: Iterator[tuple[np.ndarray, np.ndarray]]
        if groups is None:
            folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(indices)
        else:
            folds = GroupKFold(n_splits=n_splits).split(indices, groups=groups)
        return folds

    def _train_fold(
        self,
        model: torch.nn.Module,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
        epochs: int,
        batch_size: int,
        train_config: dict[str, Any] | None,
        seed: int,
    ) -> None:
        """Train a torch module on one training fold with ECGTrainer."""
        # Imported here: training does not depend on evaluation, nor the reverse at import.
        from deepecgkit.training import ECGTrainer

        config = dict(train_config or {})
        if y.ndim > 1:
            config.setdefault("multi_label", True)
        if self.task_type == "regression":
            config.setdefault("task_type", "regression")

        available = len(x) if groups is None else len(np.unique(groups))
        inner_folds = min(INNER_VALIDATION_FOLDS, available)
        if inner_folds < 2:
            raise ValueError(
                "Each training fold needs at least two samples (or groups) to set a "
                "validation part aside; use fewer folds or more data."
            )
        fit_idx, tune_idx = next(self._folds(len(x), inner_folds, groups, seed))
        loaders = _ArrayLoaders(
            (x[fit_idx], y[fit_idx]), (x[tune_idx], y[tune_idx]), batch_size, seed
        )
        trainer = ECGTrainer(model, train_config=config, device=self.device)
        trainer.fit(loaders, epochs=epochs, progress_bar=False)

    def _fit_and_score(
        self,
        model: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Fit an estimator that is not a torch module and score it on the held-out fold."""
        if hasattr(model, "fit"):
            model.fit(x_train, y_train)

        if callable(model):
            return self.evaluate(model, (x_val, y_val))
        if hasattr(model, "predict_proba"):
            return self.evaluate(model.predict_proba(x_val), y_val)
        if hasattr(model, "predict"):
            return self.evaluate(model.predict(x_val), y_val)
        raise TypeError(
            f"{type(model).__name__} can be neither called nor asked to predict, so it "
            f"cannot be cross-validated."
        )

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
