# Calibration

Accuracy and AUROC say whether a model ranks recordings correctly. They say nothing about whether a
predicted probability of 0.9 means the model is right nine times out of ten. DeepECG-Kit measures
that gap and corrects it with temperature scaling.

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

Returns `ece`, `adaptive_ece`, `mce` and `brier`, their `_temp` counterparts measured after scaling,
the fitted `temperature`, and `calibration_bins`. Without a fitting split the scaled entries are
`NaN` and the rest is still populated.

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

`deepecg train` runs a validation pass at the end of training, fits a temperature, and writes both
the raw and rescaled figures into `classification_metrics.csv`.

`deepecg evaluate` does the same for an existing checkpoint and prints the comparison:

```bash
deepecg evaluate --checkpoint model.pt -m resnet -d af-classification --split test
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
  --metrics ece ece_temp brier brier_temp
```

(`python scripts/aggregate_results.py` remains as a shim for the same command.)

Each cell carries the mean over seeds with a Student-t interval. Rows are ordered by the first
metric, ascending for calibration metrics, so the table leads with the best-calibrated model.
