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

`afmodel`, `kanres`, `kanres-deep`, `lstm`, `resnet`, `simple-cnn`, `transformer`

## Datasets

`af-classification`, `ltafdb`, `mitbih-afdb`, `ptbxl`, `unified-af`
