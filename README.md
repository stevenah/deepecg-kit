# DeepECG-Kit

PyTorch Lightning-based deep learning library for ECG analysis and arrhythmia classification.

## Installation

```bash
pip install deepecg-kit
```

### Local Development

```bash
git clone https://github.com/your-username/deepecg-kit.git
cd deepecg-kit
pip install -e ".[dev]"
```

Or using uv:

```bash
uv pip install -e ".[dev]"
```

## CLI Usage

```bash
# Train a model (dataset downloads automatically if not present)
deepecg train -m kanres -d af-classification

# Evaluate a checkpoint
deepecg evaluate --checkpoint model.ckpt -d af-classification

# Run inference
deepecg predict --checkpoint model.ckpt --input ecg.npy

# Resume training
deepecg resume --checkpoint model.ckpt --epochs 100

# List available models/datasets
deepecg list-models
deepecg list-datasets

# Show model info
deepecg info -m kanres
```

## Models

`afmodel`, `convnext-v2`, `crnn`, `deep-res-cnn`, `dualnet`, `fcn-wang`, `gru`, `inception-time`, `kanres`, `kanres-deep`, `lstm`, `mamba`, `medformer`, `resnet`, `resnet-wang`, `se-resnet`, `simple-cnn`, `tcn`, `transformer`, `xresnet`, `xresnet1d-benchmark`

## Datasets

| Name | Leads | Classes | Task | Source |
|------|-------|---------|------|--------|
| `af-classification` | 1 | 4 (Normal, AF, Other, Noisy) | Single-label | PhysioNet 2017 |
| `ptbxl` | 12 | 5 diagnostic superclasses | Multi-label | PTB-XL |
| `mitbih-afdb` | 2 | 4 (Normal, AF, AFL, J) or binary | Single-label | MIT-BIH AFDB |
| `ltafdb` | 2 | 4 or binary | Single-label | Long-Term AFDB |
| `unified-af` | 1 | 4 | Single-label | Combined (PhysioNet 2017 + MIT-BIH + LTAFDB) |

## Experiments

Experiments are defined in `experiments.yaml` and run via the experiment runner script. Each experiment trains a model on a dataset using the CLI under the hood.

### Default Training Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping patience | 10 |
| Validation split | 0.2 |
| Test split | 0.1 |
| Seed | 42 |

### Experiment Matrix

**PTB-XL** — 12-lead, multi-label classification (5 diagnostic superclasses):

`resnet`, `se-resnet`, `inception-time`, `convnext-v2`, `crnn`, `medformer` (lr=0.0005), `mamba` (lr=0.0005)

**PhysioNet 2017 (af-classification)** — 1-lead, 4-class single-label:

`resnet`, `se-resnet`, `inception-time`, `convnext-v2`, `crnn`, `medformer` (lr=0.0005), `mamba` (lr=0.0005)

**MIT-BIH AFDB** — 2-lead, binary AF detection (AF vs Non-AF):

`resnet`, `se-resnet`, `inception-time`, `convnext-v2`, `crnn`, `medformer` (lr=0.0005), `mamba` (lr=0.0005)

**Unified AF** — 1-lead, 4-class cross-dataset:

`resnet`, `convnext-v2`, `crnn`, `medformer` (lr=0.0005)

### Running Experiments

```bash
# Run all experiments (stops on first failure)
python scripts/run_experiments.py

# Dry run — print commands without executing
python scripts/run_experiments.py --dry-run

# Resume after fixing a failure
python scripts/run_experiments.py --resume

# Use a custom config
python scripts/run_experiments.py --config my_experiments.yaml

# Clear saved progress
python scripts/run_experiments.py --reset
```

### Training Outputs

Each run saves to `runs/{timestamp}-{model}-{dataset}/` with:
- Model checkpoints (top 3 by val_loss)
- Training loss/accuracy plots
- Test set evaluation (confusion matrix, classification report)
