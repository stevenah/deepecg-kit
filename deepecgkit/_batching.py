"""Shared input normalisation for model wrappers (internal).

FeatureExtractor and ECGExplainer accept the same family of inputs — arrays,
tensors, recording files, Datasets, DataLoaders, and ECGDataModules — and both
need them as a non-shuffling DataLoader so row order stays joinable to external
metadata. The dispatch lives here once.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

from deepecgkit.datasets.io import as_model_input, read_ecg_recording
from deepecgkit.datasets.modules import ECGDataModule

__all__ = ["as_inference_dataloader", "unpack_batch"]


def as_inference_dataloader(
    data: Any,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    input_channels: int | None,
    verb: str = "processed",
) -> DataLoader:
    """Normalise any supported input to a non-shuffling DataLoader.

    Args:
        data: An array, tensor, file path, Dataset, DataLoader, or ECGDataModule
        split: Split name, used only for an ECGDataModule; "all" selects the
            full dataset
        batch_size: Batch size for loaders constructed here
        num_workers: Worker count for loaders constructed here
        input_channels: Lead count used to shape raw arrays for the model
        verb: Past-tense verb naming the operation ("extracted", "explained"),
            used in the shuffling warning

    Returns:
        A loader yielding either signals alone or (signals, labels).

    Raises:
        TypeError: If the input type is not supported.
        RuntimeError: If a data module split is unavailable.
    """

    def build_loader(dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    if isinstance(data, ECGDataModule):
        # The module's own train_dataloader shuffles, which would scramble row
        # order, so the loader is constructed here instead.
        if data.dataset is None or data.train_dataset is None:
            data.setup()
        subset = data.dataset if split == "all" else getattr(data, f"{split}_dataset")
        if subset is None:
            raise RuntimeError(f"Data module has no '{split}' split available.")
        return build_loader(subset)

    if isinstance(data, DataLoader):
        if isinstance(data.sampler, RandomSampler):
            warnings.warn(
                f"The supplied DataLoader shuffles, so {verb} rows will not follow "
                "dataset order and cannot be joined to external metadata. Pass "
                "shuffle=False for a reproducible ordering.",
                RuntimeWarning,
                stacklevel=4,
            )
        return data

    if isinstance(data, Dataset):
        return build_loader(data)

    if isinstance(data, str | Path):
        array = as_model_input(read_ecg_recording(data), input_channels)
        return build_loader(TensorDataset(torch.as_tensor(array)))

    if isinstance(data, torch.Tensor):
        array = as_model_input(data.detach().cpu().numpy(), input_channels)
        return build_loader(TensorDataset(torch.as_tensor(array)))

    if isinstance(data, np.ndarray):
        array = as_model_input(data, input_channels)
        return build_loader(TensorDataset(torch.as_tensor(array)))

    raise TypeError(
        f"Unsupported input type: {type(data).__name__}. Pass an array, tensor, file path, "
        f"Dataset, DataLoader, or ECGDataModule."
    )


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a batch into signals and optional labels.

    Args:
        batch: A tensor, or a sequence whose first element is the signals

    Returns:
        Tuple of the signals and the labels, which are None when absent.
    """
    if isinstance(batch, list | tuple):
        signals = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return signals, labels
    return batch, None
