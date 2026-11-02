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

## PTB-XL: labels and split

PTB-XL follows the protocol of the PTB-XL benchmark (Strodthoff et al., 2021) by default, in its
labels and in its split.

**Labels.** With `label_rule="benchmark"`, the default, a diagnostic superclass is positive when
the record carries any statement that maps to it, whatever likelihood the statement is given, and
records with no superclass of the task are left out. `label_rule="likelihood"` keeps the rule of
releases before 1.0: a statement counts only at a likelihood of at least 50, and records without a
superclass stay in the dataset with every label negative. The rule applies to multi-label targets
only. On PTB-XL 1.0.3 the benchmark rule leaves out 411 records and keeps 21,388.

**Split.** With `use_recommended_folds=True`, the default, the dataset assigns every record to the
partition its stratified fold recommends: folds 1 to 8 for training, fold 9 for validation and
fold 10 for testing. The folds were drawn per patient, so the split is patient-level, and it is the
same for every seed and every architecture; the seed still sets the initialisation and the data
order. On PTB-XL 1.0.3 the partitions hold 17,084, 2,146 and 2,158 records.
`use_recommended_folds=False` hands the split back to `ECGDataModule`, which then draws its seeded,
patient-grouped split with `val_split` and `test_split`. From the command line:

```bash
deepecg train -m resnet -d ptbxl                  # multi-label, recommended folds, benchmark labels
deepecg train -m resnet -d ptbxl --seeded-split   # seeded 70/20/10 split by patient
deepecg train -m resnet -d ptbxl --label-rule likelihood
deepecg train -m resnet -d ptbxl --no-multi-label # one label per record, the primary diagnosis
```

Checkpoints record both options (see [Checkpointing](training.md#checkpointing)), and the commands
that start from a checkpoint rebuild PTB-XL with them. A checkpoint written before these options
existed is rebuilt the way it was trained: likelihood rule and seeded split.

## Long-term AF datasets

In their four-class mode MIT-BIH AFDB and LTAFDB label segments Normal, AF, AFL or J from the
rhythm annotations, and every other annotated rhythm (ventricular tachycardia, bigeminy and the
rest, which LTAFDB has many of) counts as Normal. With `binary_classification` the two classes are
AF and everything else, which is what that grouping amounts to.

## ECGDataModule

`ECGDataModule` handles dataset creation, train/val/test splitting, and DataLoader construction:

```python
from deepecgkit import ECGDataModule

data_module = ECGDataModule(
    "af-classification",
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

# The shapes a model for this data needs
print(data_module.input_channels, data_module.num_classes, data_module.class_names)
```

The first argument is a name from `deepecg list-datasets`, a dataset class, or a dataset
instance of your own.

Options specific to one dataset go in `dataset_kwargs`:

```python
data_module = ECGDataModule(
    "ltafdb",
    dataset_kwargs={"binary_classification": True, "segment_duration_seconds": 30.0},
    download=True,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | None | Registered dataset name, dataset class, or dataset instance |
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
| `normalization_stats` | None | `"fit"` standardises every partition with one mean and standard deviation fitted on the training partition; a `(mean, std)` pair applies given ones |
| `train_crop_length` | None | Cut every training sample to a random window of this many samples, drawn anew each epoch |

A dataset that names a recommended partition for every sample (`predefined_partition()`, as
PTB-XL does by default) is split that way, and `val_split`, `test_split` and `stratify` are then
unused. Otherwise the split is drawn from `seed`, grouped by recording or patient and stratified by
label.

`input_channels`, `num_classes` and `class_names` describe the data a model has to fit; reading
any of them loads the dataset if `setup()` has not run yet. After `setup()`,
`data_module.sampling_rate` holds the rate in effect, `data_module.split_protocol`
says which split was used (`"predefined"` or `"seeded"`), and `data_module.describe()` returns
everything a checkpoint records about the data: dataset name, preprocessing, class names, sample
shape, labelling mode, the dataset options that decide labels and split (`dataset_options`), the
split protocol (`split_protocol`) and the split settings (`split`).

`normalization_stats` is the global standardisation of the PTB-XL benchmark: one mean and one
standard deviation over every value of every lead of the training partition, so that a signal keeps
its amplitude relative to the training data, which per-signal z-scoring removes. Build the dataset
with `normalization="none"` when using it. After `setup()` the attribute holds the statistics in
effect, and `describe()` records them as `"normalization": "global"` with `normalization_stats`, so
a checkpoint applies the same values to new data. `train_crop_length` leaves validation and test
samples whole; a trainer with an `inference_window` scores them on overlapping windows (see
[Training](training.md#recipes)).

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
for one of them is shared; its `data_dir` is the folder that holds those three folders. A source
that cannot be loaded is an error rather than a smaller dataset; `skip_unavailable=True` builds it
from the sources that do load.

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
- `predefined_partition()` → `"train"`, `"val"` or `"test"` per sample for a dataset distributed
  with a recommended split, or `None` (the default) to leave the split to `ECGDataModule`
- `get_dataset_options()` → the constructor options that decide labels or split, which
  checkpoints record (empty by default)

## Preprocessing Utilities

### ECGStandardizer

Resamples and normalizes ECG signals of shape `(leads, samples)`:

```python
from deepecgkit.datasets import ECGStandardizer

standardizer = ECGStandardizer(target_sampling_rate=500, normalization="zscore")
signal = standardizer(raw_signal, original_sampling_rate=360)
```

`normalization` is `"zscore"` or `"minmax"` (each lead of each signal on its own), `"none"`, or
`"global"`, which takes the training data's `normalization_stats=(mean, std)` from a checkpoint.

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

A dataset distributed with a fixed split can override `predefined_partition()` to return the
partition of every sample, and `get_dataset_options()` to name the constructor options that
produced its labels and split, so that a checkpoint rebuilds it the same way.

The class works with `ECGDataModule(MyDataset, data_dir=...)`, or by its registered name, straight
away, and
with the CLI through [`--import`](cli.md#using-your-own-models-and-datasets).
