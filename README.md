# DeepECG-Kit

A PyTorch Lightning-based deep learning library for ECG analysis and arrhythmia classification. Provides a CLI for training, evaluation, and inference across multiple model architectures and ECG datasets.

Install with `pip install deepecgkit`, or for local development clone the repo and run `pip install -e ".[dev]"`.

## Usage

```bash
deepecg train -m kanres -d af-classification    # train (auto-downloads dataset)
deepecg evaluate --checkpoint model.ckpt -d af-classification
deepecg predict --checkpoint model.ckpt --input ecg.npy
deepecg resume --checkpoint model.ckpt --epochs 100
deepecg list-models                              # list available models/datasets
deepecg list-datasets
deepecg info -m kanres
```

**Models:** `afmodel`, `convnext-v2`, `crnn`, `deep-res-cnn`, `dualnet`, `fcn-wang`, `gru`, `inception-time`, `kanres`, `kanres-deep`, `lstm`, `mamba`, `medformer`, `resnet`, `resnet-wang`, `se-resnet`, `simple-cnn`, `tcn`, `transformer`, `xresnet`, `xresnet1d-benchmark`

**Datasets:** `af-classification` (1-lead, 4-class, PhysioNet 2017), `ptbxl` (12-lead, 5 superclasses, multi-label), `mitbih-afdb` (2-lead, binary/4-class, MIT-BIH AFDB), `ltafdb` (2-lead, binary/4-class, Long-Term AFDB), `unified-af` (1-lead, 4-class, combined cross-dataset).

## Experiments

Experiments are defined in `experiments.yaml` and run via `python scripts/run_experiments.py`. Each experiment trains a model on a dataset using the CLI under the hood. Results save to `runs/{timestamp}-{model}-{dataset}/` with checkpoints (top 3 by val_loss), training plots, and test evaluation (confusion matrix, classification report).

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

9.  **mamba** — Gu A, Dao T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*, 2023. [[GitHub]](https://github.com/state-spaces/mamba)

10. **medformer** — Wang N, Liang X, Wang Z, Zhao J, Liu Y, Peng L, Miao C. "MedFormer: A Multi-Granularity Patching Transformer for Medical Time-Series Classification." *NeurIPS*, 2024. [[GitHub]](https://github.com/DL4mHealth/Medformer)

11. **xresnet / xresnet1d-benchmark** — Strodthoff N, Wagner P, Schaeffter T, Samek W. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics*, 25(5):1519–1528, 2021. [[GitHub]](https://github.com/helme/ecg_ptbxl_benchmarking)