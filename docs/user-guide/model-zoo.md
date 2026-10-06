# Model Zoo

DeepECG-Kit includes 21 model architectures for ECG signal classification. All models follow a consistent interface:

```python
model = ModelClass(input_channels=1, output_size=4)
```

## All Models

| Registry Name | Class | Type | Description |
|---------------|-------|------|-------------|
| `afmodel` | `AFModel` | CNN | Atrial Fibrillation model optimized for 30s ECG segments |
| `kanres` | `KanResWideX` | CNN+Residual | KAN-ResNet architecture with wide layers |
| `kanres-deep` | `KanResDeepX` | CNN+Residual | Deep KAN-ResNet architecture |
| `resnet` | `ResNet1D` | Residual CNN | 1D ResNet adapted for ECG |
| `resnet-wang` | `ResNetWang` | Residual CNN | ResNet (Wang et al.) for time series |
| `se-resnet` | `SEResNet1D` | Residual+Attention | ResNet with Squeeze-and-Excitation |
| `xresnet` | `XResNet1D` | Residual CNN | XResNet with Mish activation and blur-pool |
| `xresnet1d-benchmark` | `XResNet1dBenchmark` | Residual CNN | XResNet from PTB-XL benchmark |
| `inception-time` | `InceptionTime1D` | Multi-scale CNN | Multi-scale temporal InceptionTime |
| `convnext-v2` | `ConvNeXtV21D` | Modern CNN | ConvNeXt V2 with depthwise conv and GRN |
| `deep-res-cnn` | `DeepResCNN` | 2D Residual CNN | Deep Residual 2D CNN for multi-lead ECG |
| `fcn-wang` | `FCNWang` | FCN | Fully Convolutional Network (Wang et al.) |
| `simple-cnn` | `SimpleCNN` | Lightweight CNN | Lightweight CNN for fast inference |
| `crnn` | `CRNN` | CNN+LSTM | CNN-LSTM hybrid for temporal aggregation |
| `gru` | `GRUECG` | Recurrent | GRU-based sequential ECG model |
| `lstm` | `LSTMECG` | Recurrent | LSTM-based sequential ECG model |
| `tcn` | `TCN` | Temporal CNN | Temporal Convolutional Network with dilated causal convolutions |
| `transformer` | `TransformerECG` | Transformer | Transformer-based ECG classifier |
| `medformer` | `Medformer` | Transformer | Multi-granularity patching Transformer (NeurIPS 2024) |
| `dualnet` | `ECGDualNet` | Hybrid | Dual-path CNN-LSTM + Transformer |
| `mamba` | `Mamba1D` | State Space | Bidirectional Mamba with linear complexity |

## Choosing a Model

**Fast inference / small datasets:**

- `simple-cnn` — Minimal parameters, fast training
- `kanres` — Good accuracy/speed trade-off

**Long sequences:**

- `mamba` — Linear complexity, handles very long sequences efficiently
- `tcn` — Dilated convolutions capture long-range dependencies

**Multi-lead ECG (12-lead):**

- `deep-res-cnn` — 2D convolutions across leads
- `xresnet1d-benchmark` — Designed for the PTB-XL benchmark

**State-of-the-art research:**

- `medformer` — Multi-granularity Transformer (NeurIPS 2024)
- `dualnet` — Dual-path architecture combining CNN-LSTM and Transformer

## Input Length

Most architectures pool over time and accept any signal length. A few do not, and
`deepecg list-models` shows the limits:

- `afmodel` is built for 30-second recordings and needs at least 7971 samples per lead. The
  datasets cut 10-second segments by default, so train it with `--segment-duration 30`
  (`segment_duration_seconds=30` in Python).
- `transformer` and `dualnet` attend over every sample, with positional encodings up to 5000
  samples, and `medformer` accepts up to 5009. Their memory use grows with the square of the
  signal length; if a run stops with an out-of-memory error, lower `--batch-size` first, then
  the sampling rate or segment length.

`deepecg train` checks the model against the dataset's signal length before training starts.

## Pretrained Weights

No pretrained weights are published with the package yet. Train a model, or start from a
checkpoint of your own:

```bash
deepecg train -m kanres -d af-classification --weights runs/<run-directory>/checkpoints/<file>.pt
```

To distribute weights, register them with a URL you host; the name then works wherever a
weight name is accepted:

```python
from deepecgkit.utils import register_weights

register_weights(
    "kanres-my-task",
    url="https://example.org/kanres-my-task.pt",
    model_class="KanResWideX",
    model_kwargs={"input_channels": 1, "output_size": 4},
    description="KanResWideX trained on my task",
)
```

## Using the Registry

```python
from deepecgkit.registry import get_model, get_model_names, get_model_info

print(get_model_names())

info = get_model_info("kanres")
print(info["description"])

model_class = get_model("kanres")
model = model_class(input_channels=1, output_size=4)
```

## Adding Your Own Model

Subclass `ECGModelBase`, which supplies the forward pass, and register the class:

```python
import torch
from torch import nn

from deepecgkit.models import ECGModelBase
from deepecgkit.registry import register_model


@register_model("my-model", description="A small custom CNN", min_input_length=7)
class MyModel(ECGModelBase):
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        self.backbone = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self._feature_dim = 32
        self.classifier = nn.Linear(32, output_size)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).mean(dim=-1)   # (batch, feature_dim)
```

The constructor takes `input_channels` and `output_size`; `extract_features` returns a flat
`(batch, feature_dim)` tensor and the base class applies dropout and the head. Name the head
attribute `classifier`, or set `HEAD_ATTR` to the name you use.

Any `nn.Module` with that constructor can be trained, but feature extraction, probing and
explanation need `extract_features` and `feature_dim`, which is what the base class is for.
Registered models work with `build_model("my-model", ...)` in Python and, through
[`--import`](cli.md#using-your-own-models-and-datasets), with every CLI command.

## Model Comparison

Use the included benchmark script to compare architectures:

```bash
python examples/model_comparison.py --input-channels 1 --output-size 4 --signal-length 3000
```

This instantiates every model and reports parameter counts, feature dimensions, and inference speed. See the [Model Comparison example](../examples/model-comparison.md) for details.
