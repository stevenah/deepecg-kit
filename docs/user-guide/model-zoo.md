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
- `kanres` — Good accuracy/speed trade-off, pretrained weights available

**Long sequences:**

- `mamba` — Linear complexity, handles very long sequences efficiently
- `tcn` — Dilated convolutions capture long-range dependencies

**Multi-lead ECG (12-lead):**

- `deep-res-cnn` — 2D convolutions across leads
- `xresnet1d-benchmark` — Designed for the PTB-XL benchmark

**State-of-the-art research:**

- `medformer` — Multi-granularity Transformer (NeurIPS 2024)
- `dualnet` — Dual-path architecture combining CNN-LSTM and Transformer

## Pretrained Weights

| Weight Name | Model | Task |
|-------------|-------|------|
| `kanres-af-30s` | KanResWideX | AF classification, 30s segments |
| `afmodel-30s` | AFModel | AF classification, 30s segments |

```python
from deepecgkit.utils.weights import load_pretrained_weights

state_dict = load_pretrained_weights("kanres-af-30s")
model.load_state_dict(state_dict, strict=False)
```

Or via the CLI:

```bash
deepecg train -m kanres -d af-classification --weights kanres-af-30s
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

## Model Comparison

Use the included benchmark script to compare architectures:

```bash
python examples/model_comparison.py --input-channels 1 --output-size 4 --signal-length 3000
```

This instantiates every model and reports parameter counts, feature dimensions, and inference speed. See the [Model Comparison example](../examples/model-comparison.md) for details.
