# Training

The `ECGTrainer` class wraps any `nn.Module` and provides a complete training loop with early stopping, checkpointing, learning rate scheduling, and CSV metric logging.

## Basic Usage

```python
from deepecgkit.training import ECGTrainer

model = ...  # any nn.Module
train_config = {
    "learning_rate": 1e-3,
    "scheduler": {"factor": 0.5, "patience": 5},
    "task_type": "classification",
}

trainer = ECGTrainer(model=model, train_config=train_config)
trainer.fit(data_module, epochs=50, early_stopping_patience=10)
results = trainer.test(data_module)
```

## Train Config

The `train_config` dictionary controls the training behavior:

| Key | Type | Description |
|-----|------|-------------|
| `learning_rate` | float | Learning rate for Adam optimizer |
| `scheduler` | dict | `{"factor": float, "patience": int}` for ReduceLROnPlateau |
| `binary_classification` | bool | Use BCE loss for binary tasks |
| `multi_label` | bool | Use BCE loss for multi-label tasks |
| `task_type` | str | `"classification"` (CrossEntropyLoss) or `"regression"` (MSELoss) |
| `pos_weight` | list | Optional positive class weights for BCE loss |

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

Checkpoints are saved whenever validation loss improves. Only the top `save_top_k` checkpoints are kept (older ones are deleted). The checkpoint format:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": int,
    "best_val_loss": float,
    "train_config": dict,
}
```

### Loading Checkpoints

```python
trainer = ECGTrainer.load_checkpoint("checkpoints/best.pt", model=model)
```

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

## Pretrained Weights

Load pretrained weights before training for transfer learning:

```python
from deepecgkit.utils.weights import load_pretrained_weights

state_dict = load_pretrained_weights("kanres-af-30s", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
```

Available pretrained weights:

| Name | Model | Description |
|------|-------|-------------|
| `kanres-af-30s` | KanResWideX | AF classification on 30s segments |
| `afmodel-30s` | AFModel | AF classification on 30s segments |

## Reproducibility

```python
ECGTrainer.seed_everything(42)
```

Sets random seeds for Python, NumPy, and PyTorch (including CUDA).
