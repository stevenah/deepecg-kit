# Calibration

Accuracy and AUROC say whether a model ranks recordings correctly. They say nothing about whether a
predicted probability of 0.9 means the model is right nine times out of ten. DeepECG-Kit measures
that gap and corrects it with temperature scaling and, for sigmoid outputs, with per-label Platt
scaling.

## Measuring calibration

```python
from deepecgkit.evaluation import calculate_calibration_metrics

metrics = calculate_calibration_metrics(y_true, y_prob, n_bins=15)
```

Returns: `ece`, `adaptive_ece`, `mce`, `brier`, plus `n_bins` and `task` recording how the values
were produced. Multi-label inputs also return `per_class_ece`, `per_class_mce` and
`per_class_brier`.

Each metric is available on its own as `expected_calibration_error`, `adaptive_calibration_error`,
`maximum_calibration_error` and `brier_score`.

### Choosing a bin count

Expected calibration error sorts predictions into bins and compares each bin's mean confidence with
how often it was right. The result depends on how many bins there are, and two values binned
differently are not comparable. `n_bins` is a parameter everywhere and defaults to 15, following
Guo et al. (2017), which keeps values comparable with the temperature-scaling literature.

`adaptive_calibration_error` uses equal-mass bins rather than equal-width ones, so every bin holds
the same number of samples and no bin's estimate rests on a handful of predictions. Report it
alongside the equal-width number. Where the two disagree, the binning is doing more work than the
model.

The reliability diagram drawn by `plot_calibration_curve` uses 10 bins, so the picture and the
table are binned differently by design. Every run records the count it used in the
`calibration_bins` column of `classification_metrics.csv`.

### What each metric tells you

`ece` is the average gap between confidence and accuracy, weighted by how many predictions fall in
each bin. `mce` is the largest single gap, so it describes the confidence level where the model is
least trustworthy; it is noisier than `ece` because one sparse bin can set it.

`brier` is a proper scoring rule and rewards sharpness as well as calibration, so a model can
improve its `ece` while its `brier` worsens. Multi-class and binary problems use the classic
sum-over-classes form, which ranges over `[0, 2]`; multi-label problems average the per-class binary
score, which ranges over `[0, 1]`. The two conventions are on different scales, which is safe
because tables are emitted one dataset at a time.

## Task shapes

The three shapes this library trains are each scored the way that shape requires, and the shape is
detected from the arrays:

```python
from deepecgkit.evaluation import resolve_task_shape

shape = resolve_task_shape(y_true, y_prob)
```

Multi-class and binary predictions are scored on the top label: the confidence is the largest
predicted probability and the prediction is right when the argmax matches the label. The binary
head emits one logit, which `ECGTrainer` stores as the two columns `[1 - p, p]`, so it needs no
special handling.

Multi-label predictions are scored one class at a time over the full probability range and then
averaged over classes. They are never pooled into a single stream, because pooling lets a
well-calibrated common class hide a badly calibrated rare one.

## Temperature scaling

Temperature scaling divides the logits by a single positive scalar chosen to minimise negative
log-likelihood. Because one positive scalar cannot reorder logits, accuracy, precision, recall, F1
and AUROC are all unchanged; only the probabilities move.

```python
from deepecgkit.evaluation import TemperatureScaler

scaler = TemperatureScaler().fit(val_logits, val_targets)
test_probabilities = scaler.transform(test_logits)
print(scaler.temperature)
```

Fit on validation and report on test. Fitting and reporting on the same split measures nothing. A
temperature above one means the model was overconfident and has been softened; below one means it
was underconfident.

Raw logits come from `ECGTrainer.get_test_logits()` after any evaluation pass:

```python
trainer.validate(data_module)
val_targets = trainer.get_test_results()[1]
val_logits = trainer.get_test_logits()
```

Note that `validate` and `test` reuse one set of buffers, so copy the results out before running
the next split.

## Per-label scaling

One temperature shared by every output corrects only a global over- or underconfidence. Sigmoid
outputs, one per label, can be miscalibrated in different directions, and class-weighted training
makes them so: a positive weight `w` on a label moves the logit that minimises the loss by
`log(w)`, so every label of a PTB-XL model trained with positive weights carries its own offset,
which no single temperature removes.

Per-label Platt scaling fits a positive scale `a` and a bias `b` for every sigmoid output,
mapping its logit `z` to `sigmoid(a * z + b)`, by minimising that label's logistic negative
log-likelihood on the fitting split. The scale is kept positive, so the ranking within each label,
and hence its AUROC, is unchanged. The bias moves the 0.5 threshold, so the decisions of the
recalibrated model can differ: per-label scaling is reported as a calibration result only, and the
discrimination metrics are always computed from the unscaled outputs.

```python
import numpy as np

from deepecgkit.evaluation import apply_platt, fit_platt
from deepecgkit.evaluation.metrics import TASK_MULTILABEL

rng = np.random.default_rng(0)
offsets = np.array([1.0, -0.5, 0.0])  # each label is off in its own direction
fit_label_logits = rng.normal(0, 2, size=(2000, 3))
fit_label_targets = (rng.random((2000, 3)) < 1 / (1 + np.exp(-(fit_label_logits - offsets)))).astype(float)

scales, biases = fit_platt(fit_label_logits, fit_label_targets, task=TASK_MULTILABEL)
recalibrated = apply_platt(rng.normal(0, 2, size=(500, 3)), scales, biases, task=TASK_MULTILABEL)
```

A single logit per sample (a binary head) is one output; pass no `task` and `apply_platt` returns
the two columns `[1 - p, p]` that `ECGTrainer` stores. A softmax over classes is not a set of
independent sigmoids, and `fit_platt` refuses it. An output whose fitting split holds a single
class has no finite optimum and keeps the identity map, with a warning.

If only probabilities survive, `probabilities_to_logits` reconstructs the logits. The
reconstruction is exact in principle but float32 probabilities saturate to 1.0 once a logit passes
about 16.6, so confidently wrong models lose precision. That is why logits are captured rather than
rebuilt.

## Reporting both at once

`calibration_report` produces the before-and-after mapping that the CLI prints and the metrics CSV
stores:

```python
from deepecgkit.evaluation import calibration_report

report = calibration_report(
    test_targets,
    test_probabilities,
    fit_targets=val_targets,
    fit_logits=val_logits,
    eval_logits=test_logits,
)
```

Returns `ece`, `adaptive_ece`, `mce` and `brier`, their `_temp` counterparts measured after
temperature scaling, the fitted `temperature`, their `_platt` counterparts measured after per-label
scaling, `platt_parameters` (the fitted scales and biases as JSON, for example
`{"scale": [0.91, ...], "bias": [-1.17, ...]}`), and `calibration_bins`. The `_platt` entries are
`NaN` and `platt_parameters` is empty for a softmax head. Without a fitting split every scaled
entry is `NaN` and the rest is still populated.

`ECGEvaluator` reports calibration in `generate_report` without being asked, and adds the
temperature rows when given a validation split:

```python
from deepecgkit.evaluation import ECGEvaluator

evaluator = ECGEvaluator(metrics=["accuracy", "f1"], calibration_bins=15)
report = evaluator.generate_report(
    predictions, targets, y_scores=probabilities, calibration_fit=(val_targets, val_logits)
)
```

## From the command line

`deepecg train` runs a validation pass at the end of training, fits a temperature and, for sigmoid
outputs (multi-label and binary heads), the per-label scales and biases, and writes the raw and
recalibrated figures into `classification_metrics.csv`. The console table then has an `After P`
column beside `After T`. The fitted scales and biases are stored in the `platt_parameters` column,
so reliability diagrams after per-label scaling can be rebuilt from the stored probabilities.

`deepecg evaluate` does the same for an existing checkpoint and prints the comparison:

```bash
deepecg evaluate --checkpoint runs/<run-directory> -d af-classification --split test
```

The split is redrawn at evaluation time, and getting it right matters more here than usual: a
temperature fitted on a validation split that overlaps the original training data is fitted on
data the model has already seen. When the checkpoint lives inside a run directory written by
`deepecg train`, the `--seed`, `--val-split` and `--test-split` recorded in `run_metadata.json`
are adopted automatically for flags left at their defaults, and the rebuilt split sizes are
verified against `split_info.json`; explicit flags that contradict the recording are kept with a
warning. For a bare checkpoint with no run directory, pass the values the model was trained with
yourself. Use `--calibration-bins` to change the binning and `--no-calibration` to skip the extra
validation pass.

## Building the results table

Calibration columns flow into the same aggregation the other tables use, so a table of ECE and
Brier before and after scaling across models and datasets is one command:

```bash
deepecg aggregate --runs-dir runs --output-dir analysis --emit-latex \
  --metrics ece ece_temp ece_platt brier brier_temp brier_platt
```

(`python scripts/aggregate_results.py` remains as a shim for the same command.)

Each cell carries the mean over seeds with a Student-t interval. Rows are ordered by the first
metric, ascending for calibration metrics, so the table leads with the best-calibrated model.
