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
    num_workers=4,
    val_split=0.2,
    test_split=0.1,
    seed=42,
    stratify=True,
    download=True,
)
data_module.setup(stage="fit")
```

The dataset is automatically downloaded on first use. `ECGDataModule` handles train/val/test splitting and DataLoader construction.

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
train_config = {
    "learning_rate": 1e-3,
    "scheduler": {"factor": 0.5, "patience": 5},
    "binary_classification": False,
    "task_type": "classification",
}

trainer = ECGTrainer(model=model, train_config=train_config)
trainer.fit(
    data_module,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="runs/kanres-af/checkpoints",
)
```

Training includes early stopping, automatic checkpointing (keeps top 3 by validation loss), and an optional tqdm progress bar.

### 4. Evaluate

```python
results = trainer.test(data_module)
print(f"Test loss: {results['test_loss']:.4f}")
print(f"Test accuracy: {results['test_acc']:.4f}")
```

## CLI Workflow

The same workflow via the `deepecg` command:

```bash
deepecg train -m kanres -d af-classification --epochs 50 --batch-size 32

deepecg evaluate --checkpoint runs/*/checkpoints/*.pt -m kanres -d af-classification

deepecg list-models

deepecg info -m kanres
```

See the [CLI reference](../user-guide/cli.md) for all available options.
