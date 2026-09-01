# CLI Usage

DeepECG-Kit provides the `deepecg` command for training, evaluation, and inference.

## Global Options

| Flag | Description |
|------|-------------|
| `--verbose`, `-v` | Enable verbose output |
| `--quiet`, `-q` | Suppress non-essential output |
| `--config`, `-c` | Path to configuration file (YAML or JSON) |

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
| `--output-dir`, `-o` | `runs/{timestamp}-{model}-{dataset}` | Output directory |
| `--weights`, `-w` | None | Pretrained weight name or file path |
| `--epochs`, `-e` | 50 | Number of training epochs |
| `--batch-size`, `-b` | 32 | Batch size |
| `--learning-rate`, `--lr` | 0.001 | Learning rate |
| `--val-split` | 0.2 | Validation split ratio |
| `--test-split` | 0.1 | Test split ratio |
| `--num-workers` | 4 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--devices` | 1 | Number of devices |
| `--force-download` | false | Force re-download the dataset |
| `--early-stopping-patience` | 10 | Epochs without improvement before stopping |
| `--seed` | 42 | Random seed |
| `--multi-label` | false | Use multi-label classification with BCE loss |
| `--sampling-rate` | None | Target sampling rate in Hz |
| `--binary-classification` | false | Binary AF vs Non-AF classification |
| `--normalization` | None | Normalization method: `zscore`, `minmax`, `none` |

### `deepecg evaluate`

Evaluate a trained model on a dataset split.

```bash
deepecg evaluate --checkpoint model.pt -m kanres -d af-classification
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Path to model checkpoint |
| `--model`, `-m` | *required* | Model architecture |
| `--dataset`, `-d` | *required* | Dataset to evaluate on |
| `--data-dir` | None | Directory containing the dataset |
| `--batch-size`, `-b` | 32 | Batch size |
| `--num-workers` | 4 | Data loading workers |
| `--accelerator` | auto | Device selection |
| `--devices` | 1 | Number of devices |
| `--force-download` | false | Force re-download the dataset |
| `--split` | test | Dataset split: `test` or `val` |

### `deepecg predict`

Run inference on an ECG file.

```bash
deepecg predict --checkpoint model.pt -m kanres --input ecg.npy
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Path to model checkpoint |
| `--model`, `-m` | *required* | Model architecture |
| `--input`, `-i` | *required* | Input file (`.npy`, `.csv`, or WFDB `.dat`/`.hea`) |
| `--output`, `-o` | None | Save predictions to JSON file |
| `--batch-size`, `-b` | 1 | Batch size |
| `--accelerator` | auto | Device selection |

### `deepecg extract-features`

Extract deep embeddings and write them to disk. See
[Feature Extraction](feature-extraction.md) for what the embeddings represent.

```bash
deepecg extract-features --checkpoint model.ckpt -m kanres -d af-classification -o features.npz
deepecg extract-features --checkpoint model.ckpt -m kanres -i ecg.npy -o features.csv
```

Exactly one weight source (`--checkpoint` or `--weights`) and one data source (`--dataset` or
`--input`) are required.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *one required* | Path to a model checkpoint |
| `--weights`, `-w` | *one required* | Pretrained weight name or path to a weights file |
| `--dataset`, `-d` | *one required* | Dataset to extract features from |
| `--input`, `-i` | *one required* | Input ECG file (`.npy`, `.csv`, WFDB `.dat`/`.hea`) |
| `--model`, `-m` | *required* | Model architecture |
| `--output`, `-o` | *required* | Output path (`.npz`, `.npy`, or `.csv`) |
| `--data-dir` | None | Directory containing the dataset |
| `--split` | `test` | Split to extract: `train`, `val`, `test`, `all` |
| `--batch-size`, `-b` | 32 | Batch size |
| `--num-workers` | 4 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--dtype` | `float32` | Saved precision: `float32` or `float16` |
| `--input-channels` | auto | Lead count, for models whose weights do not encode it |
| `--force-download` | false | Force re-download the dataset |

Output format follows the extension: `.npz` keeps features, labels and metadata; `.npy` stores
features only; `.csv` writes columns `f0..fN` plus a label column.

### `deepecg explain`

Attribute a model's predictions back to the signal and write the arrays and figures to a directory.
See [Explainability](explainability.md) for what the numbers mean.

```bash
deepecg explain --checkpoint model.ckpt -m kanres -d af-classification -o explanations/
deepecg explain --checkpoint model.ckpt -m kanres -i ecg.npy -o explanations/ \
    --method integrated-gradients --ig-steps 128
deepecg explain -w kanres-af-30s -m kanres -i ecg.npy -o explanations/ --no-plots
```

Exactly one weight source (`--checkpoint` or `--weights`) and one data source (`--dataset` or
`--input`) are required, as with `extract-features`. Unlike `extract-features`, `--output` is a
directory rather than a file.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *one required* | Path to a model checkpoint |
| `--weights`, `-w` | *one required* | Pretrained weight name or path to a weights file |
| `--dataset`, `-d` | *one required* | Dataset to explain predictions on |
| `--input`, `-i` | *one required* | Input ECG file (`.npy`, `.csv`, WFDB `.dat`/`.hea`) |
| `--model`, `-m` | *required* | Model architecture |
| `--output`, `-o` | *required* | Output directory, created if absent |
| `--method` | `saliency` | `saliency`, `integrated-gradients`, or `occlusion` |
| `--target-class` | predicted class | Class index to explain, or `all` |
| `--sampling-rate` | 500 | Sampling rate in Hz, used to place the beat windows |
| `--max-samples` | 8 | Maximum number of recordings to explain |
| `--ig-steps` | 64 | Path steps, integrated gradients only |
| `--occlusion-window` | 100 | Occlusion window in samples, occlusion only |
| `--occlusion-stride` | 50 | Occlusion stride in samples, occlusion only |
| `--detect-lead` | second lead | Lead index R-peaks are detected on |
| `--no-plots` | false | Write the arrays and summary only |
| `--data-dir` | None | Directory containing the dataset |
| `--split` | `test` | Split to explain: `train`, `val`, `test`, `all` |
| `--batch-size`, `-b` | 8 | Batch size |
| `--num-workers` | 4 | Data loading workers |
| `--accelerator` | auto | Device: `auto`, `cpu`, `gpu`, `mps` |
| `--input-channels` | auto | Lead count, for models whose weights do not encode it |
| `--force-download` | false | Force re-download the dataset |

The output directory receives `attribution.npz` (the attribution, the signal the model saw, the lead
distribution and the beat profile), `summary.json` (the run's provenance and headline numbers), and,
unless `--no-plots` is given, `signal_attribution.png`, `lead_importance.png` and
`beat_profile.png`. The beat figure is skipped with a warning when no complete beats were detected.

`--max-samples` defaults to 8 because attribution is expensive: integrated gradients costs
`--ig-steps` forward and backward passes per recording, and occlusion costs one forward pass per
window per lead. `--target-class all` multiplies the whole run by the class count and stores the
result under an extra leading axis in `values_by_class`.

### `deepecg resume`

Resume training from a checkpoint.

```bash
deepecg resume --checkpoint model.pt -m kanres --epochs 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *required* | Path to checkpoint to resume from |
| `--model`, `-m` | *required* | Model architecture |
| `--epochs`, `-e` | None | Additional epochs (default: continue original) |
| `--output-dir`, `-o` | None | Directory to save new checkpoints |
| `--accelerator` | auto | Device selection |
| `--devices` | 1 | Number of devices |
| `--early-stopping-patience` | 10 | Early stopping patience |

### `deepecg info`

Show model information.

```bash
deepecg info -m kanres
```

### `deepecg list-models`

List all available model architectures.

```bash
deepecg list-models
```

### `deepecg list-datasets`

List all available datasets.

```bash
deepecg list-datasets
```

## Configuration Files

Instead of passing all flags on the command line, you can use a YAML or JSON config file:

```yaml
train:
  learning_rate: 0.001
  epochs: 100
  batch_size: 64
  val_split: 0.2
  early_stopping_patience: 15
```

```bash
deepecg train -m kanres -d af-classification --config train_config.yaml
```

CLI flags override values from the config file.
