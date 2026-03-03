# Basic Training

This example trains any registered model on any registered dataset with sensible defaults.

**Source:** [`examples/train_basic.py`](https://github.com/stevenah/deepecg-kit/blob/main/examples/train_basic.py)

## Usage

```bash
python examples/train_basic.py
python examples/train_basic.py --model resnet --dataset af-classification --epochs 30
python examples/train_basic.py --model tcn --dataset af-classification --batch-size 64 --download
```

## Walkthrough

### 1. Set up the data

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGTrainer

ECGTrainer.seed_everything(42)

dataset_info = get_dataset_info("af-classification")
input_channels = dataset_info["input_channels"]
num_classes = dataset_info["num_classes"]

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
data_module.print_metadata()
```

The registry provides input channel count and number of classes, so the model can be configured automatically.

### 2. Create the model

```python
model_class = get_model("kanres")
model = model_class(input_channels=input_channels, output_size=num_classes)
```

Swap `"kanres"` for any registered model name — the rest of the code stays the same.

### 3. Configure training

```python
train_config = {
    "learning_rate": 1e-3,
    "scheduler": {"factor": 0.5, "patience": 5},
    "binary_classification": num_classes == 2,
    "task_type": "classification",
}
```

### 4. Train and evaluate

```python
trainer = ECGTrainer(model=model, train_config=train_config)
trainer.fit(
    data_module,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="runs/kanres-af/checkpoints",
)
trainer.test(data_module)

if trainer.best_checkpoint_path:
    print(f"Best checkpoint: {trainer.best_checkpoint_path}")
    print(f"Best val_loss: {trainer.best_val_loss:.4f}")
```

The trainer automatically saves the top 3 checkpoints ranked by validation loss and stops early if no improvement occurs for 10 consecutive epochs.
