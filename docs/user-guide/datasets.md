# Datasets

DeepECG-Kit provides 5 ECG datasets with automatic downloading, standardized interfaces, and preprocessing utilities.

## Available Datasets

| Registry Name | Class | Leads | Classes | Source |
|---------------|-------|-------|---------|--------|
| `af-classification` | `AFClassificationDataset` | 1 | 4 (N/A/O/~) | PhysioNet Challenge 2017 |
| `ltafdb` | `LTAFDBDataset` | 2 | 4 | PhysioNet LTAFDB |
| `mitbih-afdb` | `MITBIHAFDBDataset` | 2 | varies | MIT-BIH AFDB |
| `unified-af` | `UnifiedAFDataset` | 1 | varies | Multi-source AF |
| `ptbxl` | `PTBXLDataset` | 12 | 5 | PTB-XL |

## ECGDataModule

`ECGDataModule` handles dataset creation, train/val/test splitting, and DataLoader construction:

```python
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset

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

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
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
| `num_workers` | 4 | DataLoader workers |
| `seed` | 42 | Random seed for splitting |
| `stratify` | True | Stratified splitting based on labels |
| `download` | False | Download dataset if not present |
| `sampling_rate` | 500 | Target sampling rate (Hz) |

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

Datasets are automatically downloaded on first use when `download=True`. Data is cached in the dataset's default directory. Use `force_download=True` (CLI: `--force-download`) to re-download.

## BaseECGDataset

All datasets inherit from `BaseECGDataset`, which extends `torch.utils.data.Dataset` and defines the interface:

- `__getitem__(idx)` → `(signal_tensor, label_tensor)`
- `__len__()` → number of samples
- `get_labels()` → array of all labels (for stratified splitting)

## Preprocessing Utilities

### ECGStandardizer

Resamples and normalizes ECG signals:

```python
from deepecgkit.datasets import ECGStandardizer

standardizer = ECGStandardizer(target_fs=500, normalize="zscore")
signal = standardizer(raw_signal, original_fs=360)
```

### ECGSegmenter

Segments long ECG recordings into fixed-length windows:

```python
from deepecgkit.datasets import ECGSegmenter

segmenter = ECGSegmenter(segment_length=3000, overlap=0.5)
segments = segmenter(long_signal)
```

### RhythmAnnotationExtractor

Converts WFDB rhythm annotations to classification labels:

```python
from deepecgkit.datasets import RhythmAnnotationExtractor

extractor = RhythmAnnotationExtractor()
labels = extractor(annotation)
```
