# Training

The `ECGTrainer` class wraps any `nn.Module` and provides a complete training loop with early stopping, checkpointing, learning rate scheduling, and CSV metric logging.

## Basic Usage

```python
from deepecgkit.training import ECGTrainer

model = ...  # any nn.Module

trainer = ECGTrainer(model)
trainer.fit(data_module, epochs=50, early_stopping_patience=10)
results = trainer.test(data_module)
```

## Train Config

The optional `train_config` dictionary controls the training behavior. Every key has a default,
so pass only what you want to change:

```python
trainer = ECGTrainer(model, train_config={"learning_rate": 3e-4, "scheduler": {"patience": 3}})
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `learning_rate` | float | 1e-3 | Learning rate for Adam optimizer |
| `scheduler` | dict | `{"factor": 0.5, "patience": 5}` | Settings for ReduceLROnPlateau |
| `binary_classification` | bool | False | Use BCE loss for binary tasks (single-logit head) |
| `multi_label` | bool | False | Use BCE loss for multi-label tasks |
| `task_type` | str | `"classification"` | `"classification"` (CrossEntropyLoss) or `"regression"` (MSELoss) |
| `pos_weight` | list | None | Optional positive class weights for BCE loss |

## Loss Function Selection

The loss function is determined automatically from the config:

- `multi_label=True` or `binary_classification=True` → `BCEWithLogitsLoss`
- `task_type="classification"` → `CrossEntropyLoss`
- `task_type="regression"` → `MSELoss`

## Device Selection

```python
trainer = ECGTrainer(model=model, train_config=config, device="auto")
```

The `device` parameter accepts `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`. With `"auto"`, the trainer checks for CUDA first, then MPS, then falls back to CPU.

## Checkpointing

```python
trainer.fit(
    data_module,
    epochs=50,
    checkpoint_dir="checkpoints/",
    save_top_k=3,
)
```

Checkpoints are saved whenever validation loss improves, as
`epoch=NN-val_loss=X.XXXX.pt`. Only the top `save_top_k` checkpoints are kept (older ones are
deleted), and `trainer.best_checkpoint_path` names the best one. The checkpoint format:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": int,
    "best_val_loss": float,
    "train_config": dict,
    "model_config": {"name": "kanres", "input_channels": 1, "output_size": 4},
    "data_config": {
        "dataset": "af-classification",
        "preprocessing": {"sampling_rate": 300, "segment_duration_seconds": 10.0,
                          "normalization": "zscore", "leads": ["ECG"]},
        "class_names": ["Normal", "AF", "Other", "Noisy"],
        "input_channels": 1,
        "input_length": 3000,
        "multi_label": False,
        "binary_classification": False,
        "split": {"seed": 42, "val_split": 0.2, "test_split": 0.1},
    },
    "deepecgkit_version": str,
}
```

`model_config` and `data_config` are what make a checkpoint usable on its own: the first says
which architecture to rebuild, the second which preprocessing new inputs need and what the
outputs mean. `fit()` fills both in from the model and the data module; set
`trainer.model_config` or `trainer.data_config` beforehand to record something else.

### Loading Checkpoints

```python
trainer = ECGTrainer.load_checkpoint(trainer.best_checkpoint_path)
```

The architecture is rebuilt from the checkpoint's `model_config`. Pass `model=` to load the
weights into an instance you built yourself, which is also what a checkpoint without a recorded
architecture needs. To get just the model, use
`deepecgkit.models.build_model_from_checkpoint(path)`, which returns the model and the
checkpoint dictionary.

## Progress

`fit()` shows a progress bar over the epochs and, on an interactive terminal, a second one over
the batches of the current epoch. Pass `progress_bar=False` to turn both off.

## CSV Metric Logging

```python
trainer.fit(data_module, epochs=50, log_dir="logs/")
```

This writes a `metrics.csv` file with columns: `epoch`, `train_loss`, `val_loss`, and (for classification) `train_acc`, `val_acc`.

## Gradient Clipping

```python
trainer.fit(data_module, epochs=50, gradient_clip_val=1.0)
```

## Getting Test Results

After calling `trainer.test()`, retrieve predictions for further analysis:

```python
trainer.test(data_module)
y_pred, y_true, y_prob = trainer.get_test_results()
```

Returns numpy arrays of predictions, targets, and class probabilities (or `None, None, None` if no test has been run).

## Starting From Existing Weights

To fine-tune, load weights from an earlier run before training:

```python
from deepecgkit.models import build_model_from_checkpoint

model, _ = build_model_from_checkpoint("runs/my-run/checkpoints/epoch=07-val_loss=0.4123.pt")
```

or, for a weights file whose classifier head has a different size, load what fits:

```python
import torch

state_dict = torch.load("weights.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict.get("model_state_dict", state_dict), strict=False)
```

No pretrained weights are published with the package yet. To distribute your own, call
`deepecgkit.utils.register_weights()` with a URL you host; the name then works with
`--weights` and `Model.from_pretrained()`, and `deepecgkit.utils.list_pretrained_weights()`
lists it.

## Reproducibility

```python
ECGTrainer.seed_everything(42)
```

Sets random seeds for Python, NumPy, and PyTorch (including CUDA).
