# DeepECG-Kit

A deep learning library for ECG analysis and arrhythmia classification. Provides a CLI for
training, evaluation, inference, feature extraction, and explainability across 21 model
architectures and 5 ECG datasets.

**Documentation:** https://stevenah.github.io/deepecg-kit/

## Installation

```bash
pip install deepecgkit
```

For local development (Python ≥ 3.10):

```bash
uv sync --group dev        # blessed path
pip install -e ".[dev]"    # pip alternative
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Usage

```bash
deepecg train -m kanres -d af-classification        # train (downloads the dataset, ~100 MB)
deepecg evaluate --checkpoint runs/<run> -d af-classification
deepecg predict --checkpoint runs/<run> --input ecg.npy --sampling-rate 300
deepecg extract-features --checkpoint runs/<run> -d af-classification -o features.npz
deepecg explain --checkpoint runs/<run> --input ecg.npy -o explanations/
deepecg list-models                                  # all 21 architectures
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

Experiment sweeps are defined in `experiments.yaml` and run via
`python scripts/run_experiments.py`. Each experiment trains every configured model on every
dataset across multiple seeds (default `seeds: [42, 43, 44, 45, 46]` — 5 runs per
model/dataset pair), saving to `runs/{model}-{dataset}-seed{seed}/`.

Aggregate and analyse finished runs with:

```bash
deepecg aggregate --runs-dir runs --output-dir analysis --emit-latex
deepecg rank --runs-dir runs --metric f1
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
