# CLI Usage

DeepECG-Kit provides the `deepecg` command for training, evaluation, and inference.

## Global Options

These work before or after the command name (`deepecg -v train ...` and `deepecg train ... -v`
are the same).

| Flag | Description |
|------|-------------|
| `--verbose`, `-v` | Enable verbose output, including tracebacks for errors |
| `--quiet`, `-q` | Suppress non-essential output |
| `--config`, `-c` | Path to configuration file (YAML or JSON) |
| `--import MODULE` | Import a module or `.py` file first, so the models and datasets it registers can be used by name (repeatable) |
| `--version` | Print the installed version |

## Checkpoints

Every command that loads a model takes `--checkpoint`, which may be a checkpoint file or a run
directory. A run directory stands for its best checkpoint, the one with the lowest validation
loss, so the path `deepecg train` prints at the end can be passed straight on.

Checkpoints record the architecture they hold, their class names, and the preprocessing of the
data they were trained on (sampling rate, segment length, normalisation). The commands use that
record:

- `--model` is optional. It is only needed for checkpoints written by earlier releases, and a
  value that contradicts the checkpoint is reported as such.
- Datasets are rebuilt with the recorded preprocessing, so the model is evaluated on inputs
  shaped like its training data.
- Recording files given with `--input` are resampled, segmented and normalised to match.

A checkpoint from before this was recorded is assumed to have been trained at 500 Hz, the rate
earlier releases resampled every dataset to; the command says so and `--sampling-rate`
overrides it.

## Commands

### `deepecg train`

Train a model on a dataset.

```bash
deepecg train -m kanres -d af-classification --epochs 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | *required* | Model architecture (see `deepecg list-models`) |
| `--dataset`, `-d` | *required* | Dataset to train on (see `deepecg list-datasets`) |
| `--data-dir` | auto-detect | Directory containing the dataset |
| `--output-dir`, `-o` | `runs/{timestamp}-{model}-{dataset}-seed{seed}` | Output directory |
| `--weights`, `-w` | None | Weights to start from: a weights file, a checkpoint, a run directory, or a pretrained weight name |
| `--epochs`, `-e` | 50 | Number of training epochs |
| `--batch-size`, `-b` | 32 | Batch size |
| `--learning-rate`, `--lr` | 0.001 | Learning rate |
| `--val-split` | 0.2 | Validation split ratio |
| `--test-split` | 0.1 | Test split ratio |
| `--num-workers` | 0 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--force-download` | false | Force re-download the dataset |
| `--early-stopping-patience` | 10 | Epochs without improvement before stopping |
| `--seed` | 42 | Random seed |
| `--sampling-rate` | the dataset's own | Sampling rate in Hz the signals are resampled to |
| `--segment-duration` | the dataset's own (10) | Length of each training segment in seconds |
| `--normalization` | the dataset's own (`zscore`) | Normalization method: `zscore`, `minmax`, `none` |
| `--multi-label` | false | Multi-label classification with BCE loss (`ptbxl`) |
| `--binary-classification` | false | Collapse labels to AF vs Non-AF (`mitbih-afdb`, `ltafdb`, `unified-af`) |

Each dataset is used at its own sampling rate unless `--sampling-rate` is given: 300 Hz for the
AF datasets, 500 Hz for PTB-XL. A flag the chosen dataset cannot honour, such as
`--binary-classification` with `af-classification` or `--segment-duration` with `ptbxl`, is an
error rather than being ignored; `deepecg list-datasets` shows which options each dataset
takes.

Before training starts the command checks that the model can take the dataset's signal length
and stops with the flag to change if it cannot. `afmodel`, for instance, needs 30-second
segments: `deepecg train -m afmodel -d af-classification --segment-duration 30`.

`--num-workers` defaults to 0 because the built-in datasets hold their signals in memory; more
workers add start-up time without speeding up an epoch. `--devices` is still accepted for
existing scripts, has no effect, and will be removed.

Every run directory receives, alongside its metrics: `checkpoints/` with the best checkpoints by
validation loss, named like `epoch=07-val_loss=0.4123.pt`; `run_metadata.json` recording what
produced the run (model, dataset, seed, split fractions, labelling mode, hyperparameters, the
final returncode and the best checkpoint); and `split_info.json` with the realised split sizes.
The preprocessing fields in `run_metadata.json` (`sampling_rate`, `normalization`,
`segment_duration_seconds`, `input_length`) hold the values that were in effect, whether or not
a flag set them. `deepecg evaluate` and the [analysis commands](#deepecg-aggregate) read these
files back. The directory is created once the dataset has loaded and the model has been checked
against it, so a run that cannot start leaves nothing behind.

### `deepecg evaluate`

Evaluate a trained model on a dataset split.

```bash
deepecg evaluate --checkpoint runs/<run-directory> -d af-classification
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Checkpoint file or run directory |
| `--model`, `-m` | recorded in the checkpoint | Model architecture |
| `--dataset`, `-d` | *required* | Dataset to evaluate on |
| `--data-dir` | None | Directory containing the dataset |
| `--batch-size`, `-b` | 32 | Batch size |
| `--num-workers` | 0 | Data loading workers |
| `--accelerator` | auto | Device selection |
| `--sampling-rate` | recorded in the checkpoint | Sampling rate in Hz to resample the dataset to |
| `--force-download` | false | Force re-download the dataset |
| `--split` | test | Dataset split: `test` or `val` |
| `--seed` | recorded, else 42 | Seed that produced the split during training |
| `--val-split` | recorded, else 0.2 | Validation fraction used during training |
| `--test-split` | recorded, else 0.1 | Test fraction used during training |
| `--calibration-bins` | 15 | Bins for the calibration metrics |
| `--no-calibration` | false | Skip the validation pass that fits the temperature |
| `--output`, `-o` | console only | Directory to write metrics CSV, predictions and plots into |

The evaluation split is rebuilt from `--seed`, `--val-split` and `--test-split`. When the
checkpoint lives inside a run directory written by `deepecg train`, the values recorded in that
run's `run_metadata.json` are adopted automatically for whichever of the three flags were left at
their defaults; an explicit flag that contradicts the recording is kept but produces a warning,
because the rebuilt split will then not match the one the model was trained on. The rebuilt split
sizes are also verified against the run's `split_info.json`, again with a warning on mismatch.

Accuracy, macro precision, recall and F1, AUROC, the per-class table and the calibration block
are always printed. With `--output`, the full artifact set is also written into the directory:
`classification_metrics.csv`, `predictions.txt`, `confusion_matrix.png`, `calibration_curve.png`
and `prediction_distribution.png`. The last column of `predictions.txt` names the recording each
sample was cut from, so predictions can be grouped by recording afterwards.

### `deepecg predict`

Run inference on an ECG file.

```bash
deepecg predict --checkpoint runs/<run-directory> --input ecg.npy --sampling-rate 250
deepecg predict --checkpoint model.pt --input record.hea -o prediction.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Checkpoint file or run directory |
| `--model`, `-m` | recorded in the checkpoint | Model architecture |
| `--input`, `-i` | *required* | Input file (`.npy`, `.csv`, or WFDB `.dat`/`.hea`) |
| `--output`, `-o` | None | Save predictions to JSON file |
| `--sampling-rate` | WFDB header, else the training rate | Sampling rate of the recording in Hz |
| `--no-preprocess` | false | Pass the input to the model unchanged |
| `--batch-size`, `-b` | 32 | Segments per forward pass |
| `--accelerator` | auto | Device selection |

The recording is put through the preprocessing the checkpoint was trained with: it is resampled
from `--sampling-rate` to the training rate, cut into training-length segments, and each segment
is normalised. A raw recording can therefore be passed as it is. Each recording gets one
prediction, averaged over its segments, reported with its class name; the JSON output also holds
the per-segment predictions with their start times.

A `.npy` or `.csv` file carries no sampling rate, so give `--sampling-rate` whenever the
recording was not sampled at the rate the model was trained at. A one-dimensional array is one
single-lead recording; in a two-dimensional array for a single-lead model, each row is a
recording. CSV files may have a header row or not. Use `--no-preprocess` for input that has
already been prepared.

### `deepecg extract-features`

Extract deep embeddings and write them to disk. See
[Feature Extraction](feature-extraction.md) for what the embeddings represent.

```bash
deepecg extract-features --checkpoint runs/<run-directory> -d af-classification -o features.npz
deepecg extract-features --checkpoint model.pt -i ecg.npy --sampling-rate 250 -o features.csv
```

Exactly one weight source (`--checkpoint` or `--weights`) and one data source (`--dataset` or
`--input`) are required.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *one required* | Checkpoint file or run directory |
| `--weights`, `-w` | *one required* | Pretrained weight name or path to a weights file |
| `--dataset`, `-d` | *one required* | Dataset to extract features from |
| `--input`, `-i` | *one required* | Input ECG file (`.npy`, `.csv`, WFDB `.dat`/`.hea`) |
| `--model`, `-m` | recorded in the checkpoint | Model architecture; required with a bare weights file |
| `--output`, `-o` | *required* | Output path (`.npz`, `.npy`, or `.csv`) |
| `--data-dir` | None | Directory containing the dataset |
| `--split` | `test` | Split to extract: `train`, `val`, `test`, `all` |
| `--seed` | 42 | Seed that produced the split during training |
| `--val-split` | 0.2 | Validation fraction used during training |
| `--test-split` | 0.1 | Test fraction used during training |
| `--batch-size`, `-b` | 32 | Batch size |
| `--num-workers` | 0 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--dtype` | `float32` | Saved precision: `float32` or `float16` |
| `--input-channels` | auto | Lead count, for models whose weights do not encode it |
| `--sampling-rate` | see below | With `--input`: rate of the recording. With `--dataset`: rate to resample to |
| `--no-preprocess` | false | With `--input`: pass the recording to the model unchanged |
| `--force-download` | false | Force re-download the dataset |

Output format follows the extension: `.npz` keeps features, labels and metadata; `.npy` stores
features only; `.csv` writes columns `f0..fN` plus a label column.

With `--input`, the recording goes through the checkpoint's preprocessing, as in
[`deepecg predict`](#deepecg-predict), and the output holds one embedding per recording: a
recording long enough to be cut into several segments has their embeddings averaged.

`--seed`, `--val-split` and `--test-split` rebuild the dataset partition, so passing the values
the checkpoint was trained with makes `--split test` cover exactly the recordings the model never
saw during training.

### `deepecg explain`

Attribute a model's predictions back to the signal and write the arrays and figures to a directory.
See [Explainability](explainability.md) for what the numbers mean.

```bash
deepecg explain --checkpoint runs/<run-directory> -d af-classification -o explanations/
deepecg explain --checkpoint model.pt -i ecg.npy --sampling-rate 250 -o explanations/ \
    --method integrated-gradients --ig-steps 128
deepecg explain -w weights.pt -m kanres -i ecg.npy -o explanations/ --no-plots
```

Exactly one weight source (`--checkpoint` or `--weights`) and one data source (`--dataset` or
`--input`) are required, as with `extract-features`. Unlike `extract-features`, `--output` is a
directory rather than a file.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *one required* | Checkpoint file or run directory |
| `--weights`, `-w` | *one required* | Pretrained weight name or path to a weights file |
| `--dataset`, `-d` | *one required* | Dataset to explain predictions on |
| `--input`, `-i` | *one required* | Input ECG file (`.npy`, `.csv`, WFDB `.dat`/`.hea`) |
| `--model`, `-m` | recorded in the checkpoint | Model architecture; required with a bare weights file |
| `--output`, `-o` | *required* | Output directory, created if absent |
| `--method` | `saliency` | `saliency`, `integrated-gradients`, or `occlusion` |
| `--target-class` | predicted class | Class index to explain, or `all` |
| `--sampling-rate` | see below | With `--input`: rate of the recording. With `--dataset`: rate to resample to |
| `--no-preprocess` | false | With `--input`: pass the recording to the model unchanged |
| `--max-samples` | 8 | Maximum number of recordings to explain |
| `--ig-steps` | 64 | Path steps, integrated gradients only |
| `--occlusion-window` | 100 | Occlusion window in samples, occlusion only |
| `--occlusion-stride` | 50 | Occlusion stride in samples, occlusion only |
| `--detect-lead` | second lead | Lead index R-peaks are detected on |
| `--no-plots` | false | Write the arrays and summary only |
| `--data-dir` | None | Directory containing the dataset |
| `--split` | `test` | Split to explain: `train`, `val`, `test`, `all` |
| `--seed` | 42 | Seed that produced the split during training |
| `--val-split` | 0.2 | Validation fraction used during training |
| `--test-split` | 0.1 | Test fraction used during training |
| `--batch-size`, `-b` | 8 | Batch size |
| `--num-workers` | 0 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--input-channels` | auto | Lead count, for models whose weights do not encode it |
| `--force-download` | false | Force re-download the dataset |

The beat windows are placed at the sampling rate of the signals the model actually sees. For a
dataset that is the rate it is resampled to, which comes from the checkpoint. For a file it is
the checkpoint's training rate once the recording has been resampled to it; `--sampling-rate`
says what rate the file itself is at. Only when neither the checkpoint nor the file says
anything is 500 Hz assumed, with a warning.

The output directory receives `attribution.npz` (the attribution, the signal the model saw, the lead
distribution and the beat profile), `summary.json` (the run's provenance and headline numbers), and,
unless `--no-plots` is given, `signal_attribution.png`, `lead_importance.png` and
`beat_profile.png`. The beat figure is skipped with a warning when no complete beats were detected.

`--max-samples` defaults to 8 because attribution is expensive: integrated gradients costs
`--ig-steps` forward and backward passes per recording, and occlusion costs one forward pass per
window per lead. `--target-class all` multiplies the whole run by the class count and stores the
result under an extra leading axis in `values_by_class`.

### `deepecg probe`

Freeze encoders, fit linear probes and write the cross-dataset transfer matrix. See
[Representation Quality](representation-quality.md) for what the numbers mean and why
standardisation is part of the protocol rather than a preprocessing detail.

```bash
deepecg probe -m kanres -o probe/ \
    --source-checkpoint af-classification=runs/af/best.pt \
    --source-checkpoint ptbxl=runs/ptbxl/best.pt
```

`--source-checkpoint` is repeatable, once per encoder, and each path may be a checkpoint file or
a run directory. One entry gives a single row plus the random-init floor; four give the full
matrix. With no sources at all only the floor is probed, which is a cheap way to see how much of
a target is linearly readable from an untrained network.

The protocol resamples every dataset to `--sampling-rate`. An encoder trained at a different
rate would be probed on inputs it never saw, so the command warns when a source checkpoint
records another rate, or records none.

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | *required* | Model architecture matching every checkpoint |
| `--output`, `-o` | *required* | Directory for the matrix, plots and table |
| `--source-checkpoint` | none | `DATASET=PATH`, repeatable once per encoder source |
| `--target` | all four | Dataset to probe, repeatable |
| `--data-dir` | None | Directory containing the datasets |
| `--metric` | `auroc_macro` | `auroc_macro`, `accuracy`, `balanced_accuracy`, `f1_macro` |
| `--input-channels` | 1 | Lead count the protocol runs at |
| `--sampling-rate` | 300 | Sampling rate for every dataset, in Hz |
| `--no-random-init` | false | Drop the untrained-encoder baseline row |
| `--no-plots` | false | Skip the heatmap and the embedding plots |
| `--no-latex` | false | Skip the LaTeX table |
| `--embedding-plots` | `diagonal` | `none`, `diagonal`, or `all` |
| `--perplexity` | 30 | t-SNE perplexity, clamped to what the split supports |
| `--probe-c` | 1.0 | Inverse regularisation strength for the probe |
| `--max-iter` | 1000 | Solver iteration ceiling for the probe |
| `--val-split` | 0.2 | Validation split fraction |
| `--test-split` | 0.1 | Held-out split the probe reports on |
| `--seed` | 42 | Seed for the splits, the random encoder and the projection |
| `--batch-size`, `-b` | 32 | Batch size |
| `--num-workers` | 0 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--force-download` | false | Force re-download the datasets |

The output directory receives `transfer_matrix.json` (every metric of every cell plus the resolved
protocol), `transfer_matrix.csv` (the headline table), `transfer_matrix.png`, `transfer_matrix.tex`
and one `embedding_<source>_<target>.png` per retained cell.

Pass `--input-channels` explicitly for `deep-res-cnn`, `gru`, `lstm` and `transformer`, whose lead
count cannot be read back from their weights.

### `deepecg aggregate`

Aggregate per-run test metrics across random seeds. Run directories written by `deepecg train`
(any directory holding `classification_metrics.csv`) are discovered, grouped by model, dataset
and seed, and each metric is summarised as a mean with a Student-t confidence interval.

```bash
deepecg aggregate --runs-dir runs --output-dir analysis --emit-latex \
    --metrics f1 auroc ece
```

| Flag | Default | Description |
|------|---------|-------------|
| `--runs-dir` | `runs` | One or more directories holding run directories |
| `--output-dir` | print only | Write CSV and LaTeX output here instead of printing only |
| `--recursive` | false | Search for run directories at any depth |
| `--include-legacy` | false | Include runs whose seed cannot be determined, under a sentinel seed |
| `--on-duplicate` | `newest` | `newest`, `error` or `all` when one model, dataset and seed appears twice |
| `--models` | all | Restrict to these models |
| `--datasets` | all | Restrict to these datasets |
| `--confidence-level` | 0.95 | Two-sided coverage of the intervals |
| `--emit-latex` | false | Write booktabs tables |
| `--metrics` | default set | Metrics for the report and LaTeX tables, in column order |

### `deepecg rank`

Report how stable architecture rankings are across seeds: per-seed ranks within each dataset,
rank distributions, Kendall's tau between seeds, and paired flip tests between models.

```bash
deepecg rank --runs-dir runs --output-dir analysis --metric f1
deepecg rank --claim kanres:af-classification:ptbxl
```

`deepecg rank` shares the run-discovery flags of `deepecg aggregate` (`--runs-dir`,
`--output-dir`, `--recursive`, `--include-legacy`, `--on-duplicate`, `--models`, `--datasets`,
`--confidence-level`, `--emit-latex`) and adds:

| Flag | Default | Description |
|------|---------|-------------|
| `--metric` | `f1` | Metric to rank on |
| `--correction` | `nadeau-bengio` | Standard error correction for the paired differences: `nadeau-bengio` or `none` |
| `--min-test-groups` | 10 | Fewest independent test units that can support a test |
| `--claim` | none | Evaluate a "best here, worst there" claim, as `MODEL:BEST_DATASET:WORST_DATASET` |

Both commands are also reachable as `python scripts/aggregate_results.py` and
`python scripts/ranking_stability.py`, thin shims kept for existing invocations. See the
[analysis API reference](../api/analysis.md) for the underlying functions.

### `deepecg info`

Show model information, including parameter counts.

```bash
deepecg info -m kanres
deepecg info -m kanres --input-channels 12 --num-classes 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model`, `-m` | *required* | Model to show information for |
| `--input-channels` | 1 | Lead count used for the parameter counts |
| `--num-classes` | 4 | Class count used for the parameter counts |

Parameter counts depend on the lead and class configuration, so the output labels them with the
configuration they were computed for.

### `deepecg list-models`

List all available model architectures, with any limits on the signal length they accept.

```bash
deepecg list-models
```

### `deepecg list-datasets`

List all available datasets: their classes, lead count, sampling rate and segment length, the
options they accept, how much a download fetches, and whether a local copy exists and where.

```bash
deepecg list-datasets
```

## Using Your Own Models and Datasets

Models and datasets registered with `@register_model` and `@register_dataset` become available
to every command once the module that defines them has been imported. `--import` does that:

```bash
deepecg --import my_models.py train -m my-model -d af-classification
deepecg --import mypackage.models list-models
```

It takes a `.py` file or a dotted module name, and can be repeated. Setting the
`DEEPECGKIT_PLUGINS` environment variable to a comma-separated list does the same without the
flag. See [Model Zoo](model-zoo.md#adding-your-own-model) and
[Datasets](datasets.md#adding-your-own-dataset) for what a model and a dataset need to provide.

## Configuration Files

Every command accepts `--config` with a YAML or JSON file, so an experiment's settings can live in
one file instead of a long command line. The file holds one section per command, named after the
command — `train:`, `evaluate:`, `extract-features:` (the underscore form `extract_features:` also
works) — with keys matching the command's flags, written with underscores:

```yaml
train:
  learning_rate: 0.001
  epochs: 100
  batch_size: 64
  val_split: 0.2
  early_stopping_patience: 15

evaluate:
  batch_size: 64
```

```bash
deepecg train -m kanres -d af-classification --config experiment.yaml
deepecg evaluate --checkpoint runs/<run-directory> -d af-classification --config experiment.yaml
```

Precedence is: explicit flags > config values > built-in defaults. Only the section for the
command being run is read, so one file can configure a whole workflow. A key the chosen command
does not define produces a warning and is ignored, so a typo cannot silently change a run. The
top-level keys `verbose` and `quiet` are also honoured.

A config may also supply a command's required arguments, such as `model` and `dataset` for
`train`, in which case they can be left off the command line:

```yaml
train:
  model: kanres
  dataset: af-classification
  epochs: 100
```

```bash
deepecg train --config experiment.yaml
```
