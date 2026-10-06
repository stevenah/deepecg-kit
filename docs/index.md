# DeepECG-Kit

A deep learning library for ECG analysis and arrhythmia classification.

DeepECG-Kit provides **21 model architectures**, **5 ECG datasets**, and a complete training/evaluation pipeline — all without framework dependencies beyond PyTorch.

## Quick Install

```bash
pip install deepecgkit
```

## Quick Start

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGTrainer

dataset_info = get_dataset_info("af-classification")
dataset_class = get_dataset("af-classification")
data_module = ECGDataModule(dataset_class=dataset_class, batch_size=32, download=True)
data_module.setup(stage="fit")

model_class = get_model("kanres")
model = model_class(
    input_channels=dataset_info["input_channels"],
    output_size=dataset_info["num_classes"],
)

trainer = ECGTrainer(model)
trainer.fit(data_module, epochs=50, checkpoint_dir="runs/kanres-af/checkpoints")
trainer.test(data_module)
```

Or use the CLI:

```bash
deepecg train -m kanres -d af-classification --epochs 50
```

The first run downloads the dataset (about 100 MB) into `~/.deepecgkit/datasets`.

## Features

- **21 model architectures** — CNNs, ResNets, RNNs, Transformers, Mamba, and hybrids
- **5 ECG datasets** — PhysioNet 2017, PTB-XL, MIT-BIH AFDB, LTAFDB, and unified AF
- **Pure PyTorch** — No framework dependencies beyond PyTorch itself
- **CLI & Python API** — Train, evaluate, and predict from the command line or code
- **Registry system** — Decorator-based model/dataset registration; your own are usable from
  the CLI with `--import`
- **Self-describing checkpoints** — Architecture, class names and preprocessing are recorded,
  so evaluation and prediction need only the checkpoint

## Next Steps

- [Installation](getting-started/installation.md) — Detailed installation instructions
- [Quick Start Guide](getting-started/quickstart.md) — Step-by-step walkthrough
- [Model Zoo](user-guide/model-zoo.md) — Browse all 21 model architectures
- [CLI Usage](user-guide/cli.md) — Command-line reference
- [API Reference](api/index.md) — Full API documentation
