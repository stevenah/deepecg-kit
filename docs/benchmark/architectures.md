# Benchmarked architectures

The table gives the design of each architecture in the benchmark, read from the implementation:
the main temporal operation, the depth and kernel sizes, the width of the feature vector, the
normalisation and dropout used, how the feature vector is read out before the classification
head, and the publications the implementation follows.

The 24 architectures listed here are a starting point. Architectures are added to the package
continuously, and this page is extended with each addition; the
[Model Zoo](../user-guide/model-zoo.md) lists everything that is currently registered, and
`deepecg list-models` prints the same list from the installed version. Parameter counts and
benchmark results for each architecture are on the
[benchmark website](https://stevenah.github.io/deepecg-kit-bench/).

Several implementations adapt their sources and do not reproduce them exactly. The Baseline
family holds the two non-deep references, in which only a linear or fully connected classifier is
trained.

## Design

Architectures are grouped by family and ordered within a family by 12-lead parameter count.

### Baseline

| Architecture | Registry name | Operation | Depth (kernel) | Width | Norm | Dropout | Readout | Source |
|---|---|---|---|---|---|---|---|---|
| ROCKET | `rocket` | Frozen random conv | 1 conv (9; dilation 1–512) | 2000 | BN | – | PPV, max per kernel | [1] |
| MLP | `mlp-wang` | Fully connected | 3 dense | 500 | – | 0.1, 0.2, 0.3 | Last hidden layer | [2] |

### Convolutional

| Architecture | Registry name | Operation | Depth (kernel) | Width | Norm | Dropout | Readout | Source |
|---|---|---|---|---|---|---|---|---|
| KanRes | `kanres` | Residual conv | 3 residual (3) | 64 | BN | – | GAP | [3] |
| SimpleCNN | `simple-cnn` | Conv | 4 conv (7, 5, 3, 3) | 256 | BN | 0.3 | GAP | – |
| FCN | `fcn-wang` | Conv | 3 conv (8, 5, 3) | 128 | BN | 0.3 | GAP | [2], [4] |
| XResNet1D-Bench | `xresnet1d-benchmark` | Residual conv | 8 residual (5) | 128 | BN | 0.25 | Avg and max, concat | [5], [4] |
| ResNet-Wang | `resnet-wang` | Residual conv | 3 residual (5) | 128 | BN | 0.3 | GAP | [2], [4] |
| InceptionTime | `inception-time` | Multi-scale conv | 6 Inception (5, 15, 41) | 128 | BN | 0.3 | GAP | [6], [7] |
| DeepResCNN | `deep-res-cnn` | Residual 2-D conv | 4 residual (1×5) | 128 | BN | 0.1, 0.15 | GAP, 2 dense layers | [8] |
| SpectrogramCNN | `spectrogram-cnn` | 2-D conv on spectrograms | 4 conv (3×3) | 256 | BN | 0.3 | GAP | – |
| KanRes-Deep | `kanres-deep` | Residual conv | 8 residual (50) | 32 | BN | – | GAP | [3] |
| TCN | `tcn` | Dilated causal conv | 6 residual (7) | 256 | WN, BN | 0.2 | GAP | [9] |
| ConvNeXt-V2 | `convnext-v2` | Depthwise conv, GRN | 12 ConvNeXt (7) | 512 | LN | 0.3 | GAP | [10] |
| ResNet1D | `resnet` | Residual conv | 8 residual (7) | 512 | BN | 0.3 | GAP | [11] |
| XResNet1D | `xresnet` | Residual conv, Mish, blur-pool | 8 residual (7) | 512 | BN | 0.3 | GAP | [5], [12], [13] |
| SE-ResNet1D | `se-resnet` | Residual conv, SE | 8 residual (7) | 512 | BN | 0.3 | GAP | [11], [14] |
| ResNet-Hannun | `resnet-hannun` | Residual conv (pre-activation) | 16 residual (16) | 256 | BN | 0.2, 0.3 | GAP | [15] |
| XResNet1D-101 | `xresnet1d101` | Residual conv (bottleneck) | 33 residual (5) | 4096 | BN | 0.25 | Avg and max, concat | [5], [4] |

### Recurrent

| Architecture | Registry name | Operation | Depth (kernel) | Width | Norm | Dropout | Readout | Source |
|---|---|---|---|---|---|---|---|---|
| LSTM | `lstm` | BiLSTM | 2 layers | 256 | – | 0.3 | Final states | [16] |
| GRU | `gru` | Unidirectional GRU | 2 layers | 768 | BN | 0.15 | Mean, max, last; concat | [17], [4] |

### Hybrid

| Architecture | Registry name | Operation | Depth (kernel) | Width | Norm | Dropout | Readout | Source |
|---|---|---|---|---|---|---|---|---|
| DualNet | `dualnet` | Conv–BiLSTM in parallel with Transformer | 3 conv (7, 5, 3), 1 LSTM, 2 enc. | 384 | BN, LN | 0.3 | Branches concat | – |
| CRNN | `crnn` | Conv, then BiLSTM | 4 conv (7, 5, 3, 3), 2 LSTM | 256 | BN | 0.3 | Final states | [18] |

### Transformer

| Architecture | Registry name | Operation | Depth (kernel) | Width | Norm | Dropout | Readout | Source |
|---|---|---|---|---|---|---|---|---|
| Transformer | `transformer` | Attention over samples | 4 enc. | 128 | LN | 0.1 | GAP | [19] |
| Medformer | `medformer` | Attention over patches | 2 enc. per scale (patch 10, 25, 50) | 384 | LN | 0.1 | Per-scale mean, concat | [20] |

### Reading the table

- **Depth (kernel)** gives blocks or layers, with kernel sizes in samples in parentheses (patch
  sizes for Medformer).
- **Width** is the size of the feature vector entering the classification head, the vector that
  [feature extraction](../user-guide/feature-extraction.md) and
  [probing](../user-guide/representation-quality.md) use as the embedding.
- **Norm**: BN, LN and WN are batch, layer and weight normalisation.
- **Readout**: GAP is global average pooling over time; final states are those of the last
  recurrent layer; concat is concatenation; PPV is the proportion of positive values.
- enc., Transformer encoder layers; conv, convolution; BiLSTM, bidirectional LSTM; GRN, global
  response normalisation; SE, squeeze-and-excitation; –, none or not applicable.
- **Source** cites the publications followed, with the PTB-XL benchmark [4] where the variant
  comes from its code. A dash marks an in-house design without a published reference.

## Sources

1. Dempster, Petitjean and Webb. ROCKET: exceptionally fast and accurate time series
   classification using random convolutional kernels. *Data Mining and Knowledge Discovery* 34,
   1454–1495 (2020). [doi:10.1007/s10618-020-00701-z](https://doi.org/10.1007/s10618-020-00701-z)
2. Wang, Yan and Oates. Time series classification from scratch with deep neural networks: a
   strong baseline. *IJCNN* (2017). [doi:10.1109/IJCNN.2017.7966039](https://doi.org/10.1109/IJCNN.2017.7966039)
3. Hicks et al. Explaining deep neural networks for knowledge discovery in electrocardiogram
   analysis. *Scientific Reports* 11, 10949 (2021). [doi:10.1038/s41598-021-90285-5](https://doi.org/10.1038/s41598-021-90285-5)
4. Strodthoff, Wagner, Schaeffter and Samek. Deep learning for ECG analysis: benchmarks and
   insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics* 25, 1519–1528 (2021).
   [doi:10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)
5. He et al. Bag of tricks for image classification with convolutional neural networks. *CVPR*
   (2019). [doi:10.1109/CVPR.2019.00065](https://doi.org/10.1109/CVPR.2019.00065)
6. Ismail Fawaz et al. InceptionTime: finding AlexNet for time series classification. *Data
   Mining and Knowledge Discovery* 34, 1936–1962 (2020). [doi:10.1007/s10618-020-00710-y](https://doi.org/10.1007/s10618-020-00710-y)
7. Szegedy et al. Rethinking the Inception architecture for computer vision. *CVPR* (2016).
   [doi:10.1109/CVPR.2016.308](https://doi.org/10.1109/CVPR.2016.308)
8. Elyamani, Salem, Melgani and Yhiea. Deep residual 2D convolutional neural network for
   cardiovascular disease classification. *Scientific Reports* 14, 22040 (2024).
   [doi:10.1038/s41598-024-72382-3](https://doi.org/10.1038/s41598-024-72382-3)
9. Bai, Kolter and Koltun. An empirical evaluation of generic convolutional and recurrent
   networks for sequence modeling. arXiv:1803.01271 (2018).
   [doi:10.48550/arXiv.1803.01271](https://doi.org/10.48550/arXiv.1803.01271)
10. Woo et al. ConvNeXt V2: co-designing and scaling ConvNets with masked autoencoders. *CVPR*
    (2023). [doi:10.1109/CVPR52729.2023.01548](https://doi.org/10.1109/CVPR52729.2023.01548)
11. He, Zhang, Ren and Sun. Deep residual learning for image recognition. *CVPR* (2016).
    [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
12. Misra. Mish: a self regularized non-monotonic activation function. arXiv:1908.08681 (2019).
    [doi:10.48550/arXiv.1908.08681](https://doi.org/10.48550/arXiv.1908.08681)
13. Zhang. Making convolutional networks shift-invariant again. *ICML*, PMLR 97, 7324–7334
    (2019). [proceedings.mlr.press/v97/zhang19a.html](https://proceedings.mlr.press/v97/zhang19a.html)
14. Hu, Shen and Sun. Squeeze-and-excitation networks. *CVPR* (2018).
    [doi:10.1109/CVPR.2018.00745](https://doi.org/10.1109/CVPR.2018.00745)
15. Hannun et al. Cardiologist-level arrhythmia detection and classification in ambulatory
    electrocardiograms using a deep neural network. *Nature Medicine* 25, 65–69 (2019).
    [doi:10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3)
16. Hochreiter and Schmidhuber. Long short-term memory. *Neural Computation* 9, 1735–1780
    (1997). [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
17. Cho et al. Learning phrase representations using RNN encoder–decoder for statistical machine
    translation. *EMNLP* (2014). [doi:10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
18. Zihlmann, Perekrestenko and Tschannen. Convolutional recurrent neural networks for
    electrocardiogram classification. *Computing in Cardiology* 44 (2017).
    [doi:10.22489/CinC.2017.070-060](https://doi.org/10.22489/CinC.2017.070-060)
19. Vaswani et al. Attention is all you need. *Advances in Neural Information Processing
    Systems* 30 (2017).
20. Wang, Huang, Li, Yan and Zhang. Medformer: a multi-granularity patching Transformer for
    medical time-series classification. *Advances in Neural Information Processing Systems* 37
    (2024). [doi:10.52202/079017-1145](https://doi.org/10.52202/079017-1145)
