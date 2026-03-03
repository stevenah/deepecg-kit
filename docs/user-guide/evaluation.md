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

## ECGEvaluator

For more comprehensive evaluation:

```python
from deepecgkit.evaluation import ECGEvaluator

evaluator = ECGEvaluator(metrics=["accuracy", "precision", "recall", "f1", "auc"])
results = evaluator.evaluate(model, test_loader)
```

`ECGEvaluator` supports passing a model + DataLoader, or pre-computed predictions + targets as numpy arrays.

## Metrics

### Classification Metrics

```python
from deepecgkit.evaluation import calculate_classification_metrics

metrics = calculate_classification_metrics(y_true, y_prob)
```

Returns a dictionary with: `accuracy`, `precision`, `recall`, `f1`, `auc`, `mcc`.

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

### ECG Signal Visualization

```python
from deepecgkit.evaluation import plot_ecg_signals

plot_ecg_signals(signals, labels=labels, save_path="ecg.png")
```

### Prediction Visualization

```python
from deepecgkit.evaluation import plot_predictions

plot_predictions(y_true, y_pred, save_path="predictions.png")
```

## CLI Evaluation

```bash
deepecg evaluate --checkpoint model.pt -m kanres -d af-classification --split test
```

See the [CLI reference](cli.md#deepecg-evaluate) for all options.
