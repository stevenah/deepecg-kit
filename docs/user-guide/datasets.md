# Datasets

DeepECG-Kit provides 5 ECG datasets with automatic downloading, standardized interfaces, and preprocessing utilities.

## Available Datasets

| Registry Name | Class | Leads | Classes | Sampling rate | Download |
|---------------|-------|-------|---------|---------------|----------|
| `af-classification` | `AFClassificationDataset` | 1 | 4 (Normal, AF, Other, Noisy) | 300 Hz, 10 s segments | about 100 MB |
| `mitbih-afdb` | `MITBIHAFDBDataset` | 1 (2 with `use_both_leads`) | 4, or 2 with `binary_classification` | 300 Hz, 10 s segments | about 640 MB |
| `ltafdb` | `LTAFDBDataset` | 1 (2 with `use_both_leads`) | 4, or 2 with `binary_classification` | 300 Hz, 10 s segments | about 3.7 GB |
| `unified-af` | `UnifiedAFDataset` | 1 | 4, or 2 with `binary_classification` | 300 Hz, 10 s segments | the three above |
| `ptbxl` | `PTBXLDataset` | 12 | 5 superclasses | 500 Hz, 10 s recordings | about 3.2 GB |

The sampling rate is each dataset's default: recordings are resampled to it from their native
rate (300 Hz for PhysioNet 2017, 250 Hz for MIT-BIH AFDB, 128 Hz for LTAFDB) so that the AF
datasets share an input shape. Pass `sampling_rate` to change it. `deepecg list-datasets` prints
the same table together with where each dataset is stored and whether it has been downloaded.

## ECGDataModule

`ECGDataModule` handles dataset creation, train/val/test splitting, and DataLoader construction:

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset

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

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

Options specific to one dataset go in `dataset_kwargs`:

```python
data_module = ECGDataModule(
    dataset_class=get_dataset("ltafdb"),
    dataset_kwargs={"binary_classification": True, "segment_duration_seconds": 30.0},
    download=True,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_class` | None | Dataset class to instantiate |
| `dataset` | None | Pre-instantiated dataset (alternative to `dataset_class`) |
| `data_dir` | None | Data directory (uses dataset default if None) |
| `batch_size` | 32 | Batch size for DataLoaders |
| `val_split` | 0.2 | Validation fraction |
| `test_split` | 0.1 | Test fraction |
| `num_workers` | 0 | DataLoader workers |
| `seed` | 42 | Random seed for splitting |
| `stratify` | True | Stratified splitting based on labels |
| `download` | False | Download dataset if not present |
| `sampling_rate` | None | Sampling rate to resample to (Hz); None keeps the dataset's own default |
| `dataset_kwargs` | None | Extra keyword arguments for the dataset class |

After `setup()`, `data_module.sampling_rate` holds the rate in effect and
`data_module.describe()` returns everything a checkpoint records about the data: dataset name,
preprocessing, class names, sample shape and split protocol.

`num_workers` defaults to 0 because the built-in datasets keep their signals in memory, which
makes worker processes pure overhead. If you raise it in a script on macOS or Windows, guard the
script with `if __name__ == "__main__":`.

## Using the Registry

Look up datasets by their registry name:

```python
from deepecgkit.registry import get_dataset, get_dataset_info, get_dataset_names

print(get_dataset_names())

info = get_dataset_info("af-classification")
print(f"Channels: {info['input_channels']}, Classes: {info['num_classes']}")

dataset_class = get_dataset("af-classification")
```

## Auto-Download

Datasets are downloaded on first use when `download=True` (the CLI always passes it). They are
stored under `~/.deepecgkit/datasets`, one folder per dataset; set the `DEEPECGKIT_HOME`
environment variable to move that root, or pass `data_dir` for a single dataset. `unified-af`
keeps nothing of its own and reads its three sources from their own folders, so a copy fetched
for one of them is shared; its `data_dir` is the folder that holds those three folders.

Files come from PhysioNet's open-data mirror, with the archives on physionet.org as the
fallback. Only what the loaders read is fetched. A download that is interrupted resumes where
it stopped the next time the dataset is requested, and a directory left incomplete is detected
and completed rather than loaded as it is. Use `force_download=True` (CLI: `--force-download`)
to discard a local copy and fetch it again. That deletes the dataset's folder first, so a
`data_dir` of your own is only cleared when it is empty or recognisably a copy of the dataset;
anything else is refused rather than deleted.

```python
from deepecgkit.datasets import PTBXLDataset

PTBXLDataset.is_downloaded()          # is a usable copy present?
PTBXLDataset.get_default_data_dir()   # where it lives
PTBXLDataset.download_to()            # fetch it without loading it
```

## BaseECGDataset

All datasets inherit from `BaseECGDataset`, which extends `torch.utils.data.Dataset` and defines the interface:

- `__getitem__(idx)` → `(signal_tensor, label_tensor)`, the signal shaped `(leads, samples)`
- `__len__()` → number of samples
- `class_names` / `num_classes` → the label vocabulary
- `get_class_distribution()` → sample count per class
- `group_ids` → one identifier per sample, used to keep a recording or patient inside one split
- `get_preprocessing()` → sampling rate, segment length and normalisation of the samples

## Preprocessing Utilities

### ECGStandardizer

Resamples and normalizes ECG signals of shape `(leads, samples)`:

```python
from deepecgkit.datasets import ECGStandardizer

standardizer = ECGStandardizer(target_sampling_rate=500, normalization="zscore")
signal = standardizer(raw_signal, original_sampling_rate=360)
```

### ECGSegmenter

Segments long ECG recordings into fixed-length windows:

```python
from deepecgkit.datasets import ECGSegmenter

segmenter = ECGSegmenter(segment_duration_seconds=10.0, sampling_rate=300, overlap=0.5)
segments, start_indices = segmenter.segment(long_signal)
```

### RhythmAnnotationExtractor

Converts WFDB rhythm annotations to a label per signal sample, and then per segment:

```python
from deepecgkit.datasets import RhythmAnnotationExtractor

extractor = RhythmAnnotationExtractor(sampling_rate=300)
labels = extractor.extract_labels(annotation, signal_length=long_signal.shape[-1])
segment_labels = extractor.segment_with_labels(labels, start_indices, segmenter.segment_length)
```

### Preparing a recording for a trained model

`prepare_recording` applies a dataset's preprocessing to a recording that is not part of it —
the same resampling, segmenting and normalising, in the same order. Checkpoints record the
preprocessing of their training data, so this is how a new recording is made ready for one:

```python
from deepecgkit.datasets import prepare_recording, read_ecg_recording_with_rate

signal, rate = read_ecg_recording_with_rate("record.hea")   # rate is None for .npy and .csv
prepared = prepare_recording(signal, dataset.get_preprocessing(), source_sampling_rate=rate)

prepared.segments        # (num_segments, leads, samples), ready for the model
prepared.start_seconds   # where each segment starts in the recording
prepared.notes           # what was applied, in words
```

## Adding Your Own Dataset

Subclass `BaseECGDataset`, fill the standard storage attributes in `_load_data`, and register
the class:

```python
import numpy as np

from deepecgkit.datasets import BaseECGDataset
from deepecgkit.registry import register_dataset


@register_dataset("my-dataset", input_channels=1, description="My recordings")
class MyDataset(BaseECGDataset):
    CLASS_LABELS = ["Normal", "Abnormal"]

    def _load_data(self):
        self.signals = [...]       # one float32 array of shape (leads, samples) per sample
        self.labels = [...]        # one integer class index per sample
        self.record_names = [...]  # one source-recording name per sample
```

`record_names` matters: samples sharing a name are kept in the same split, which is what stops
segments of one recording from landing in both train and test. Implement `_download(cls,
data_dir, verbose)` if the data can be fetched automatically. Constructor arguments named
`sampling_rate`, `normalization`, `segment_duration_seconds`, `binary_classification` or
`multi_label` are picked up by the matching CLI flags.

The class works with `ECGDataModule(dataset_class=MyDataset, data_dir=...)` straight away, and
with the CLI through [`--import`](cli.md#using-your-own-models-and-datasets).
