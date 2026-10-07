# Evaluation

DeepECG-Kit provides metrics, visualization, and a high-level evaluator class for model assessment.

## Quick Evaluation via ECGTrainer

The simplest way to evaluate after training:

```python
results = trainer.test(data_module)
print(f"Test loss: {results['test_loss']:.4f}")
print(f"Test accuracy: {results['test_acc']:.4f}")

y_pred, y_true, y_prob = trainer.get_test_results()
```

`trainer.validate(data_module)` does the same on the validation split and reports `val_loss`
and `val_acc`.

## ECGEvaluator

For more comprehensive evaluation:

```python
from deepecgkit.evaluation import ECGEvaluator

evaluator = ECGEvaluator(metrics=["accuracy", "precision", "recall", "f1", "auc"])
results = evaluator.evaluate(model, test_loader)
```

`ECGEvaluator` takes a model with a DataLoader, or predictions computed elsewhere with their
targets. A model is run on the evaluation device in eval mode and handed back as it came, and its
logits are turned into probabilities the way `ECGTrainer` does. With predictions, pass the
probabilities as `y_scores`, which is what AUC and the calibration metrics are computed from:

```python
y_pred, y_true, y_prob = trainer.get_test_results()

results = evaluator.evaluate(y_pred, y_true, y_scores=y_prob)
intervals = evaluator.bootstrap_evaluate(y_pred, y_true, n_bootstrap=1000, y_scores=y_prob)
report = evaluator.generate_report(y_pred, y_true, y_scores=y_prob, save_path="report.csv")
```

`bootstrap_evaluate` resamples the test set and returns the mean, standard deviation and 95%
interval of each metric; `generate_report` returns a table with the same intervals and the
calibration metrics. Precision, recall and F1 are positive-class values for a two-class problem and
macro averages otherwise. Multi-label targets are not supported by this class; `deepecg evaluate`
handles them.

## Metrics

### Classification Metrics

```python
from deepecgkit.evaluation import calculate_classification_metrics

metrics = calculate_classification_metrics(y_true, y_prob)
```

Returns a dictionary with: `accuracy`, `precision`, `recall`, `f1`, `auc`, `mcc`.

Calibration is measured separately with `calculate_calibration_metrics`; see
[Calibration](calibration.md).

### Regression Metrics

```python
from deepecgkit.evaluation import calculate_regression_metrics

metrics = calculate_regression_metrics(y_true, y_pred)
```

Returns: `mse`, `mae`, `r2`.

### Confusion Matrix Analysis

```python
from deepecgkit.evaluation import confusion_matrix_analysis

cm_results = confusion_matrix_analysis(y_true, y_pred)
```

Returns per-class precision, recall, and F1 scores.

## Visualization

All plot functions accept a `save_path` or `save_dir` parameter to save figures to disk.

### Confusion Matrix

```python
from deepecgkit.evaluation import plot_confusion_matrix

plot_confusion_matrix(y_true, y_pred, title="Results", save_path="cm.png")
```

### ROC Curve

```python
from deepecgkit.evaluation import plot_roc_curve

plot_roc_curve(y_true, y_prob, save_path="roc.png")
```

This is the curve of a two-class problem: `y_prob` is the score of the positive class, or the
two-column matrix a binary model yields. With more classes, draw one class against the rest, for
example `plot_roc_curve(y_true == 1, y_prob[:, 1])`.

### Training Curves

Plot loss and accuracy from CSV logs:

```python
from deepecgkit.evaluation import plot_training_curves

plot_training_curves("logs/metrics.csv", save_dir="plots/")
```

### Calibration Curve

```python
from deepecgkit.evaluation import plot_calibration_curve

plot_calibration_curve(y_true, y_prob, save_dir="plots/")
```

For numeric calibration measurement and temperature scaling, see [Calibration](calibration.md).

### ECG Signal Visualization

```python
from deepecgkit.evaluation import plot_ecg_signals

plot_ecg_signals(signal, sampling_rate=300, leads=["ECG"], save_path="ecg.png")
```

`signal` is one recording, `(leads, time)` as the datasets return it; `(time, leads)` is accepted as
well, the longer axis being taken as time. Pass the rate the signal is actually sampled at
(`data_module.sampling_rate` after `setup()`), since the default of 500 Hz only suits PTB-XL.

### Prediction Visualization

```python
from deepecgkit.evaluation import plot_predictions

plot_predictions(y_true, y_pred, save_path="predictions.png")
```

## CLI Evaluation

```bash
deepecg evaluate --checkpoint runs/<run-directory> -d af-classification --split test
```

The dataset is rebuilt with the split and the preprocessing recorded for the checkpoint, and the
full metric set is printed; add `-o DIR` to save it with the predictions and plots. See the
[CLI reference](cli.md#deepecg-evaluate) for all options.
