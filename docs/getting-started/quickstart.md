# Quick Start

This guide walks through training an ECG classifier from scratch using the Python API, then shows the equivalent CLI workflow.

## Python API

### 1. Set up the dataset

```python
from deepecgkit import ECGDataModule, ECGTrainer, build_model, predict

ECGTrainer.seed_everything(42)

data_module = ECGDataModule("af-classification", batch_size=32, download=True)
```

The dataset (about 100 MB) is downloaded on first use into `~/.deepecgkit/datasets`; an
interrupted download resumes on the next run. `ECGDataModule` takes any name from
`deepecg list-datasets`, splits the data 70/20/10 into train, validation and test (by recording,
and stratified by class), and builds the DataLoaders. It leaves the signals at the dataset's own
sampling rate (300 Hz here) unless you pass `sampling_rate`.

!!! note "Loader workers"
    The built-in datasets hold their signals in memory, so the default `num_workers=0` is the
    fastest setting. If you raise it in a script on macOS or Windows, put the code under
    `if __name__ == "__main__":` — worker processes re-import the script.

### 2. Create a model

```python
model = build_model("kanres", data_module.input_channels, data_module.num_classes)
```

Every architecture takes the same two shape arguments, the lead count and the number of classes,
which the data module reports (reading them loads the dataset). Any name from
`deepecg list-models` can replace `"kanres"`.

### 3. Train

```python
trainer = ECGTrainer(model, train_config={"learning_rate": 1e-3})
trainer.fit(
    data_module,
    epochs=50,
    early_stopping_patience=10,
    checkpoint_dir="runs/kanres-af",
)
print(trainer.best_checkpoint_path)
```

`train_config` is optional; every key has a default (see [Training](../user-guide/training.md)).
The task is read from the data: a multi-label dataset such as PTB-XL is trained with a sigmoid
per class without any setting.
Training includes early stopping, automatic checkpointing (keeps top 3 by validation loss), and
tqdm progress bars.

### 4. Evaluate

```python
results = trainer.test(data_module)
print(f"Test loss: {results['test_loss']:.4f}")
print(f"Test accuracy: {results['test_acc']:.4f}")
```

### 5. Predict on a new recording

The checkpoint records the preprocessing its inputs went through, so a raw recording is
prepared the same way before it reaches the model:

```python
result = predict("ecg.npy", checkpoint="runs/kanres-af", sampling_rate=250)

print(result.label)          # the predicted class
print(result.confidence)     # its probability
print(result.probabilities)  # every class, averaged over the recording's segments
```

The recording here is one lead of any length, sampled at 250 Hz; it is resampled to the 300 Hz
the model was trained at and cut into 10 s segments. A run directory stands for its best
checkpoint, and single-label, binary and multi-label models are all read correctly. To predict
many recordings, load the checkpoint once:

```python
import numpy as np

from deepecgkit import ECGPredictor

predictor = ECGPredictor.from_checkpoint("runs/kanres-af")
from_file = predictor.predict("ecg.npy", sampling_rate=250)
from_array = predictor.predict(np.load("ecg.npy"), sampling_rate=250)  # arrays work as well
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
