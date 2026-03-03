# Advanced Training

This example demonstrates pretrained weights, backbone freezing for transfer learning, CSV logging, and post-training evaluation with plots.

**Source:** [`examples/train_advanced.py`](https://github.com/stevenah/deepecg-kit/blob/main/examples/train_advanced.py)

## Usage

```bash
python examples/train_advanced.py --weights kanres-af-30s --freeze-backbone --epochs 20
python examples/train_advanced.py --model kanres --dataset af-classification --download
```

## Walkthrough

### Loading Pretrained Weights

```python
from deepecgkit.utils.weights import load_pretrained_weights

model_class = get_model("kanres")
model = model_class(input_channels=input_channels, output_size=num_classes)

state_dict = load_pretrained_weights("kanres-af-30s", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
```

`strict=False` allows loading weights even when the classifier head dimensions differ (e.g., different number of output classes).

### Freezing the Backbone

For transfer learning, freeze all parameters except the classifier head:

```python
for name, param in model.named_parameters():
    if "classifier" not in name and "fc" not in name and "head" not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,}")
```

### Training with CSV Logging

```python
trainer = ECGTrainer(model=model, train_config=train_config)
trainer.fit(
    data_module,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="runs/output/checkpoints",
    log_dir="runs/output/training_logs",
    gradient_clip_val=1.0,
)
```

The `log_dir` parameter enables CSV logging of per-epoch metrics (`epoch`, `train_loss`, `val_loss`, `train_acc`, `val_acc`).

### Post-Training Evaluation

```python
from deepecgkit.evaluation.metrics import (
    calculate_classification_metrics,
    confusion_matrix_analysis,
)
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_training_curves,
)

trainer.test(data_module)
y_pred, y_true, y_prob = trainer.get_test_results()

plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path="plots/cm.png")
plot_calibration_curve(y_true, y_prob, save_dir="plots/")
plot_training_curves("runs/output/training_logs/metrics.csv", save_dir="plots/")

cm_results = confusion_matrix_analysis(y_true, y_pred)
cls_metrics = calculate_classification_metrics(y_true, y_prob)
```
