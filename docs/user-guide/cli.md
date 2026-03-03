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
