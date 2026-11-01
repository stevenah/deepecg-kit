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
| `learning_rate` | float | 1e-3 | Learning rate; the peak rate under a one-cycle schedule |
| `optimizer` | dict | `{"name": "adam"}` | `"adam"` or `"adamw"`, with optional `betas` and `weight_decay` (decoupled for AdamW) |
| `scheduler` | dict | `{"factor": 0.5, "patience": 5}` | ReduceLROnPlateau settings, stepped once per epoch; `{"name": "one-cycle", ...}` uses OneCycleLR (`pct_start`, `div_factor`, `final_div_factor`, `base_momentum`, `max_momentum`), stepped once per batch |
| `inference_window` | dict | None | `{"length": ..., "stride": ..., "aggregate": "max"}` in samples: evaluation scores overlapping windows and combines their outputs by their maximum or mean |
| `binary_classification` | bool | from the data | Use BCE loss for binary tasks (single-logit head) |
| `multi_label` | bool | from the data | Use BCE loss for multi-label tasks |
| `task_type` | str | `"classification"` | `"classification"` (CrossEntropyLoss) or `"regression"` (MSELoss) |
| `pos_weight` | list | None | Optional positive class weights for BCE loss |

## Recipes

A recipe fixes a whole training and evaluation protocol under one name, for reproducing a published
protocol without restating its settings. The package has one, `ptbxl-benchmark`, the protocol of
the PTB-XL benchmark (Strodthoff et al., 2021); without a recipe, every architecture and dataset is
trained with the package's own settings. On the command line a recipe is one flag
(`deepecg train --recipe ptbxl-benchmark`, see [CLI Usage](cli.md#deepecg-train)); in Python its
parts are passed to the data module, the trainer and `fit()`:

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.models import build_model
from deepecgkit.training import ECGTrainer
from deepecgkit.training.recipes import get_recipe, kaiming_normal_init

recipe = get_recipe("ptbxl-benchmark")
data = ECGDataModule(
    "ptbxl",
    sampling_rate=recipe.sampling_rate,
    batch_size=recipe.batch_size,
    normalization_stats="fit",  # the recipe's "global" normalization
    train_crop_length=round(recipe.crop_seconds * recipe.sampling_rate),
    dataset_kwargs={"normalization": "none", "use_high_resolution": False},
)
model = build_model("fcn-wang", data.input_channels, data.num_classes)
kaiming_normal_init(model)
trainer = ECGTrainer(model, {
    "multi_label": True,
    "learning_rate": recipe.learning_rate,
    "optimizer": recipe.optimizer_config(),
    "scheduler": dict(recipe.scheduler),
    "inference_window": recipe.inference_window(recipe.sampling_rate),
})
trainer.fit(data, epochs=recipe.epochs, keep=recipe.keep, checkpoint_dir="checkpoints/")
```

`keep="last"` trains every epoch, logs the validation loss without acting on it, and keeps and
saves the final epoch's weights; the default, `keep="best"`, restores the lowest validation loss
and stops early. An `inference_window` changes only evaluation: `validate()`, `test()` and the
validation loss in `fit()` score each recording on overlapping windows, while training sees the
crops the data module draws.

## Loss Function Selection

The loss follows the task. When `train_config` sets none of `multi_label`,
`binary_classification` and `task_type`, the task is read from the data the trainer is first
given, which is what makes `ECGTrainer(model)` right for every built-in dataset:

- multi-hot labels (PTB-XL, multi-label by default) → `BCEWithLogitsLoss`, one sigmoid per class
- a binary dataset feeding a single-logit head → `BCEWithLogitsLoss`
- otherwise → `CrossEntropyLoss`

The task found is written into `train_config`, so checkpoints record how their outputs are read.
Stated explicitly, the keys win:

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
    checkpoint_dir="runs/my-run",
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
        "dataset_options": {},
        "split_protocol": "seeded",
        "split": {"seed": 42, "val_split": 0.2, "test_split": 0.1},
    },
    "deepecgkit_version": str,
}
```

`model_config` and `data_config` are what make a checkpoint usable on its own: the first says
which architecture to rebuild, the second which preprocessing new inputs need and what the
outputs mean. `dataset_options` and `split_protocol` say how the labels and the partitions were
formed; for PTB-XL they read `{"label_rule": "benchmark", "use_recommended_folds": True}` and
`"predefined"` by default (see [Datasets](datasets.md#ptb-xl-labels-and-split)). `fit()` fills both in from the model and the data module; set
`trainer.model_config` or `trainer.data_config` beforehand to record something else.

### Loading Checkpoints

```python
trainer = ECGTrainer.load_checkpoint(trainer.best_checkpoint_path)
```

Checkpoints are read in PyTorch's restricted mode, which loads tensors and plain values and does
not run code a file may carry; everything `ECGTrainer` writes loads that way. A checkpoint from
elsewhere that needs a full unpickle is refused unless you set `DEEPECGKIT_TRUST_CHECKPOINTS=1`.

The architecture is rebuilt from the checkpoint's `model_config`. Pass `model=` to load the
weights into an instance you built yourself, which is also what a checkpoint without a recorded
architecture needs. To get just the model, use
`deepecgkit.models.build_model_from_checkpoint(path)`, which returns the model and the
checkpoint dictionary; to predict on new recordings, `deepecgkit.predict(recording, path)`.
Wherever a checkpoint path is taken, the run directory it was written to can be passed instead
and stands for its checkpoint with the lowest validation loss.

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
