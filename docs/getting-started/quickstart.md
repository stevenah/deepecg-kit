# Quick Start

This guide walks through training an ECG classifier from scratch using the Python API, then shows the equivalent CLI workflow.

## Python API

### 1. Set up the dataset

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info
from deepecgkit.training import ECGTrainer

ECGTrainer.seed_everything(42)

dataset_info = get_dataset_info("af-classification")
dataset_class = get_dataset("af-classification")

data_module = ECGDataModule(
    dataset_class=dataset_class,
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    seed=42,
    stratify=True,
    download=True,
)
data_module.setup(stage="fit")
```

The dataset (about 100 MB) is downloaded on first use into `~/.deepecgkit/datasets`; an
interrupted download resumes on the next run. `ECGDataModule` handles train/val/test splitting
and DataLoader construction, and leaves the signals at the dataset's own sampling rate
(300 Hz here) unless you pass `sampling_rate`.

!!! note "Loader workers"
    The built-in datasets hold their signals in memory, so the default `num_workers=0` is the
    fastest setting. If you raise it in a script on macOS or Windows, put the code under
    `if __name__ == "__main__":` — worker processes re-import the script.

### 2. Create a model

```python
from deepecgkit.registry import get_model

model_class = get_model("kanres")
model = model_class(
    input_channels=dataset_info["input_channels"],
    output_size=dataset_info["num_classes"],
)
```

All models follow the same interface: `ModelClass(input_channels, output_size, **kwargs)`. Use `get_model_names()` to list all available architectures.

### 3. Train

```python
trainer = ECGTrainer(model, train_config={"learning_rate": 1e-3})
trainer.fit(
    data_module,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="runs/kanres-af/checkpoints",
)
print(trainer.best_checkpoint_path)
```

`train_config` is optional; every key has a default (see [Training](../user-guide/training.md)).
Training includes early stopping, automatic checkpointing (keeps top 3 by validation loss), and
tqdm progress bars.

### 4. Evaluate

```python
results = trainer.test(data_module)
print(f"Test loss: {results['test_loss']:.4f}")
print(f"Test accuracy: {results['test_acc']:.4f}")
```

### 5. Predict on a new recording

The checkpoint records the preprocessing its inputs went through, so a raw recording can be
prepared the same way:

```python
import numpy as np
import torch

from deepecgkit.datasets import prepare_recording
from deepecgkit.models import build_model_from_checkpoint

model, checkpoint = build_model_from_checkpoint(trainer.best_checkpoint_path)
data_config = checkpoint["data_config"]

recording = np.load("ecg.npy")  # one lead, any length, sampled at 250 Hz
prepared = prepare_recording(
    recording, data_config["preprocessing"], source_sampling_rate=250
)

model.eval()
with torch.no_grad():
    probabilities = torch.softmax(model(torch.from_numpy(prepared.segments)), dim=1).mean(dim=0)
print(data_config["class_names"][int(probabilities.argmax())])
```

## CLI Workflow

The same workflow via the `deepecg` command:

```bash
deepecg train -m kanres -d af-classification --epochs 50 --batch-size 32
```

Training ends by printing the run directory and its best checkpoint. Pass that run directory
wherever a checkpoint is expected:

```bash
deepecg evaluate --checkpoint runs/<run-directory> -d af-classification

deepecg predict --checkpoint runs/<run-directory> --input ecg.npy --sampling-rate 250

deepecg list-models

deepecg info -m kanres
```

See the [CLI reference](../user-guide/cli.md) for all available options.
