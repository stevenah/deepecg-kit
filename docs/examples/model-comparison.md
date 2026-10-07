# Model Comparison

This example instantiates every registered model and benchmarks parameter count, feature dimensions, and inference speed on synthetic input.

**Source:** [`examples/model_comparison.py`](https://github.com/stevenah/deepecg-kit/blob/main/examples/model_comparison.py)

## Usage

```bash
python examples/model_comparison.py
python examples/model_comparison.py --input-channels 12 --output-size 5 --signal-length 5000
python examples/model_comparison.py --device mps
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--input-channels` | 1 | Number of ECG leads |
| `--output-size` | 4 | Number of output classes |
| `--signal-length` | 3000 | Length of input signal in samples |
| `--device` | cpu | Device to benchmark on |
| `--num-runs` | 50 | Number of inference runs for timing |

## How It Works

The script iterates over all registered models:

```python
import torch

from deepecgkit.registry import get_model, get_model_names

for model_name in get_model_names():
    model_class = get_model(model_name)
    model = model_class(input_channels=1, output_size=4)
    model.eval()

    x = torch.randn(1, 1, 3000)
    try:
        with torch.no_grad():
            output = model(x)
    except Exception as error:
        print(f"{model_name}: FAILED - {error}")
        continue

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}: {total_params:,} params, output {tuple(output.shape)}")
```

A model that cannot take the requested input is reported and skipped: `afmodel` is built for
30-second recordings, so it needs `--signal-length 9000` at 300 Hz.

For each model it reports:

- **Total parameters** and **trainable parameters**
- **Feature dimension** (`model.feature_dim`)
- **Output shape**
- **Inference time** (averaged over multiple runs with warmup)

This helps you choose the right model for your latency and memory constraints. See the [Model Zoo](../user-guide/model-zoo.md) for guidance on model selection.
