# DeepECG-Kit

A deep learning library for ECG analysis and arrhythmia classification.

DeepECG-Kit provides **30 model architectures**, **5 ECG datasets**, and a complete training/evaluation pipeline — all without framework dependencies beyond PyTorch.

## Quick Install

```bash
pip install deepecgkit
```

## Quick Start

```python
from deepecgkit import ECGDataModule, ECGTrainer, build_model, predict

data = ECGDataModule("af-classification", download=True)
model = build_model("kanres", data.input_channels, data.num_classes)

trainer = ECGTrainer(model)
trainer.fit(data, epochs=50, checkpoint_dir="runs/kanres-af")
trainer.test(data)

result = predict("ecg.npy", checkpoint="runs/kanres-af", sampling_rate=250)
print(result.label, result.confidence)
```

Or use the CLI:

```bash
deepecg train -m kanres -d af-classification --epochs 50
```

The first run downloads the dataset (about 100 MB) into `~/.deepecgkit/datasets`.

## Features

- **30 model architectures** — CNNs, ResNets, RNNs, Transformers, state space models (S4, Mamba), and hybrids
- **5 ECG datasets** — PhysioNet 2017, PTB-XL, MIT-BIH AFDB, LTAFDB, and unified AF
- **Pure PyTorch** — No framework dependencies beyond PyTorch itself
- **CLI & Python API** — Train, evaluate, and predict from the command line or code
- **Registry system** — Decorator-based model/dataset registration; your own are usable from
  the CLI with `--import`
- **Self-describing checkpoints** — Architecture, class names, preprocessing, labelling and split
  are recorded, so evaluation and prediction need only the checkpoint
- **Standardised splits** — PTB-XL follows its recommended folds and the labels of the PTB-XL
  benchmark; the other datasets are split by recording or patient
- **Evaluation beyond one score** — bootstrap intervals over test recordings, calibration before
  and after temperature and per-label scaling, linear probes against untrained encoders, and
  attribution summaries with a model-randomisation check

## Next Steps

- [Installation](getting-started/installation.md) — Detailed installation instructions
- [Quick Start Guide](getting-started/quickstart.md) — Step-by-step walkthrough
- [Model Zoo](user-guide/model-zoo.md) — Browse all 30 model architectures
- [CLI Usage](user-guide/cli.md) — Command-line reference
- [API Reference](api/index.md) — Full API documentation
