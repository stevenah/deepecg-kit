<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
    <img src="https://raw.githubusercontent.com/stevenah/deepecg-kit/main/docs/assets/logo-light.png" alt="DeepECG-Kit logo" width="160">
  </picture>
</p>

# DeepECG-Kit

A deep learning library for ECG analysis and arrhythmia classification. Provides a CLI for
training, evaluation, inference, feature extraction, and explainability across 26 model
architectures and 5 ECG datasets.

**Documentation:** https://stevenah.github.io/deepecg-kit/

## Installation

Install DeepECG-Kit from PyPI, or from source to work on it. Requires Python ≥ 3.10.

### Quick install

```bash
# Using pip
pip install deepecgkit

# Using uv (recommended)
uv pip install deepecgkit
```

### From source (development)

```bash
git clone https://github.com/stevenah/deepecg-kit.git
cd deepecg-kit
uv sync --group dev
# or, without uv: pip install -e ".[dev]"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Quick Start

Load a dataset, train any architecture on it, predict on a raw recording, and explain the
predictions, in a few lines of Python. Datasets download from PhysioNet on first use into
`~/.deepecgkit/datasets`; the same steps from the shell are under [Usage](#usage).

### Load a dataset and iterate batches

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info
from deepecgkit.training import ECGTrainer

ECGTrainer.seed_everything(42)

# PhysioNet 2017 (about 100 MB) as 10 s single-lead segments, resampled
# from 300 Hz to the 500 Hz the benchmark uses for every dataset. The
# 70/20/10 split is by recording and stratified by class, so no recording
# has segments in both train and test.
info = get_dataset_info("af-classification")
data = ECGDataModule(
    dataset_class=get_dataset("af-classification"),
    sampling_rate=500,
    batch_size=32,
    download=True,
)
data.setup(stage="fit")

signals, labels = next(iter(data.train_dataloader()))
# signals: (32, 1, 5000) float32, each segment normalised on its own
# labels:  (32,) class indices
```

### Train and test any architecture

```python
from deepecgkit.models import build_model

# Every architecture takes the same two shape arguments, so any name from
# `deepecg list-models` can replace "kanres" with nothing else changed.
model = build_model("kanres", info["input_channels"], info["num_classes"])

# Training ends by restoring the epoch with the lowest validation loss.
trainer = ECGTrainer(model)
trainer.fit(data, epochs=50, checkpoint_dir="runs/kanres-af")
print(trainer.test(data))  # {"test_loss": 0.806, "test_acc": 0.655}
```

The figures in this Quick Start come from the benchmark's KanRes run with this setup (seed 42,
best epoch 50 of 50), which also scores macro F1 0.453, MCC 0.317 and AUROC 0.795 on the
same test split. A new run differs slightly with hardware and library version.

### Predict on a raw recording

```python
import numpy as np
import torch

from deepecgkit.datasets import prepare_recording
from deepecgkit.models import build_model_from_checkpoint

# The checkpoint records its architecture, class names and preprocessing,
# so a recording at another rate and length is resampled, cut and
# normalised to match.
model, checkpoint = build_model_from_checkpoint(trainer.best_checkpoint_path)
config = checkpoint["data_config"]
prepared = prepare_recording(
    np.load("ecg.npy"), config["preprocessing"], source_sampling_rate=250
)

# Softmax suits a single-label head like this one; `deepecg predict` also
# handles binary and multi-label heads.
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(prepared.segments))
probabilities = torch.softmax(logits, dim=1).mean(dim=0)  # over segments
print(config["class_names"][int(probabilities.argmax())])
```

### Explain the predictions

```python
from deepecgkit.explainability import explain

# Saliency for every test segment, averaged over R-peak-aligned beats. The
# checkpoint does not record the rate the beat windows assume, so state it.
result = explain(
    data, checkpoint=trainer.best_checkpoint_path, sampling_rate=500
)

# Share of the average beat's attribution in each region
result.pqrst_fractions   # P 0.315, QRS 0.082, T 0.343, baseline 0.260
# Each share over the region's share of the beat, so 1 is chance
result.pqrst_enrichment  # P 1.26, QRS 0.66, T 0.92, baseline 1.04
```

On multi-lead data such as PTB-XL, `result.lead_importance` also gives each lead's share of the
attribution.

## Usage

```bash
deepecg train -m kanres -d af-classification        # train (downloads the dataset, ~100 MB)
deepecg evaluate --checkpoint runs/<run> -d af-classification
deepecg predict --checkpoint runs/<run> --input ecg.npy --sampling-rate 300
deepecg extract-features --checkpoint runs/<run> -d af-classification -o features.npz
deepecg explain --checkpoint runs/<run> --input ecg.npy -o explanations/
deepecg list-models                                  # all 26 architectures
deepecg list-datasets                                # sizes, signal shape, local status
deepecg info -m kanres
```

Training writes a run directory `runs/{timestamp}-{model}-{dataset}-seed{seed}/` containing
checkpoints (top 3 by validation loss), training plots, test metrics
(`classification_metrics.csv`, confusion matrix, calibration curve), and run metadata, and ends
by printing the best checkpoint. Anywhere a checkpoint is expected, the run directory can be
passed instead and stands for that best checkpoint.

Checkpoints describe themselves: they record the architecture, the class names, and the
preprocessing the training data went through (sampling rate, segment length, normalisation).
That is why `--model` is optional above, why `deepecg evaluate` rebuilds the dataset the way
the model was trained, and why `deepecg predict` can take a raw recording: it is resampled,
segmented and normalised to match before it reaches the model.

Run `deepecg list-models` and `deepecg list-datasets` for the full catalogs — datasets cover
PhysioNet 2017 (`af-classification`), PTB-XL (`ptbxl`), MIT-BIH AFDB (`mitbih-afdb`),
Long-Term AFDB (`ltafdb`), and the combined `unified-af`. Each dataset is used at its own
sampling rate (300 Hz for the AF datasets, 500 Hz for PTB-XL) unless `--sampling-rate` says
otherwise.

Datasets are downloaded on first use into `~/.deepecgkit/datasets`; set `DEEPECGKIT_HOME` to
keep them somewhere else. Downloads come from PhysioNet's open-data mirror, fall back to
physionet.org, and resume if interrupted.

## Experiments

Experiment sweeps are defined in YAML files and run via
`python scripts/run_experiments.py --config <file>`, one run per architecture, dataset and seed,
saved to `runs/{model}-{dataset}-seed{seed}/`. `experiments.yaml` holds the benchmark of the
journal article: 24 architectures on PTB-XL and PhysioNet 2017 with seed 42 (48 runs);
`experiments-longterm.yaml` holds the long-term AF tasks for the benchmark website. A `seeds:`
list in either file, or `--seeds` at launch, repeats every pairing.

Aggregate and analyse finished runs with:

```bash
deepecg aggregate --runs-dir runs --output-dir analysis --emit-latex
deepecg bootstrap --runs-dir runs --output-dir analysis/bootstrap   # test-set intervals
deepecg rank --runs-dir runs --metric f1                             # needs several seeds
```

### Default Training Parameters

Defined once in `deepecgkit/cli/defaults.py`:

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
| Sampling rate | the dataset's own (override with `--sampling-rate`) |
| Data loader workers | 0 |

## References

### Datasets

1. **af-classification** — Clifford GD, Liu C, Moody B, Li-wei HL, Silva I, Li Q, Johnson AE, Mark RG. "AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017." *Computing in Cardiology (CinC)*, 2017. [[PhysioNet]](https://physionet.org/content/challenge-2017/1.0.0/)

2. **ptbxl** — Wagner P, Strodthoff N, Bousseljot RD, Kreiseler D, Lunze FI, Samek W, Schaeffter T. "PTB-XL, a Large Publicly Available Electrocardiography Dataset." *Scientific Data*, 7(1):154, 2020. [[PhysioNet]](https://physionet.org/content/ptb-xl/1.0.3/)

3. **mitbih-afdb** — Moody GB, Mark RG. "A New Method for Detecting Atrial Fibrillation Using R-R Intervals." *Computers in Cardiology*, 10:227–230, 1983. [[PhysioNet]](https://physionet.org/content/afdb/1.0.0/)

4. **ltafdb** — Petrutiu S, Sahakian AV, Swiryn S. "Abrupt Changes in Fibrillatory Wave Characteristics at the Termination of Paroxysmal Atrial Fibrillation in Humans." *Europace*, 9(7):466–470, 2007. [[PhysioNet]](https://physionet.org/content/ltafdb/1.0.0/)

5. **unified-af** — Combined cross-dataset using af-classification, mitbih-afdb, and ltafdb (see references above).

### Models

6. **convnext-v2** — Woo S, Debnath S, Hu R, Chen X, Liu Z, Kweon IS, Xie S. "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." *CVPR*, 2023. [[GitHub]](https://github.com/facebookresearch/ConvNeXt-V2)

7. **deep-res-cnn** — Elyamani H. "ECG Classification Using Deep Residual CNN." 2022. [[GitHub]](https://github.com/HaneenElyamani/ECG-classification)

8. **inception-time** — Fawaz HI, Lucas B, Forestier G, Pelletier C, Schmidt DF, Weber J, Webb GI, Idoumghar L, Muller PA, Petitjean F. "InceptionTime: Finding AlexNet for Time Series Classification." *Data Mining and Knowledge Discovery*, 34:1936–1962, 2020. [[GitHub]](https://github.com/hfawaz/InceptionTime)

9. **xresnet / xresnet1d-benchmark** — Strodthoff N, Wagner P, Schaeffter T, Samek W. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics*, 25(5):1519–1528, 2021. [[GitHub]](https://github.com/helme/ecg_ptbxl_benchmarking)
