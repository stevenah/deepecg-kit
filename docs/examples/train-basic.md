# Basic Training

This example trains any registered model on any registered dataset with sensible defaults.

**Source:** [`examples/train_basic.py`](https://github.com/stevenah/deepecg-kit/blob/main/examples/train_basic.py)

## Usage

```bash
python examples/train_basic.py --download        # first run: fetches the dataset
python examples/train_basic.py --model resnet --dataset af-classification --epochs 30
python examples/train_basic.py --model tcn --dataset af-classification --batch-size 64
```

## Walkthrough

### 1. Set up the data

```python
from deepecgkit import ECGDataModule, ECGTrainer, build_model

ECGTrainer.seed_everything(42)

data_module = ECGDataModule(
    "af-classification",
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    seed=42,
    download=True,
)
data_module.setup(stage="fit")
data_module.print_metadata()
```

### 2. Create the model

```python
model = build_model("kanres", data_module.input_channels, data_module.num_classes)
```

The data module reports the lead count and the number of classes, so the model is shaped to
fit. Swap `"kanres"` for any registered model name — the rest of the code stays the same.

### 3. Configure training

```python
train_config = {
    "learning_rate": 1e-3,
    "scheduler": {"factor": 0.5, "patience": 5},
}
```

Every key is optional; `ECGTrainer(model)` uses these same values. The loss follows the data:
cross-entropy here, BCE for a multi-label dataset such as PTB-XL.

### 4. Train and evaluate

```python
trainer = ECGTrainer(model=model, train_config=train_config)
trainer.fit(
    data_module,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="runs/kanres-af",
)
results = trainer.test(data_module)
print(f"Test loss: {results['test_loss']:.4f}  Test accuracy: {results['test_acc']:.4f}")

if trainer.best_checkpoint_path:
    print(f"Best checkpoint: {trainer.best_checkpoint_path}")
    print(f"Best val_loss: {trainer.best_val_loss:.4f}")
```

The trainer automatically saves the top 3 checkpoints ranked by validation loss and stops early if no improvement occurs for 10 consecutive epochs.
