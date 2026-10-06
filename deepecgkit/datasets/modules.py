"""
Data module for ECG datasets.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepecgkit.datasets.base import BaseECGDataset
from deepecgkit.datasets.splitting import DataSplitter

logger = logging.getLogger(__name__)


class ECGDataModule:
    """Data module for ECG datasets.

    Handles dataset creation, train/val/test splitting, and DataLoader construction.
    """

    def __init__(
        self,
        dataset: BaseECGDataset | Dataset | None = None,
        dataset_class: type[BaseECGDataset] | None = None,
        data_dir: str | None = None,
        sampling_rate: int | None = None,
        leads: list | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        download: bool = False,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
        stratify: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        verbose: bool = True,
        dataset_kwargs: dict | None = None,
    ):
        """Initialize the ECG data module.

        Args:
            dataset: PyTorch dataset instance (optional)
            dataset_class: Class of dataset to create (optional)
            data_dir: Directory containing ECG data files (optional, uses dataset's default if None)
            sampling_rate: Rate the dataset resamples its signals to (Hz). None keeps the
                dataset class's own default, which is its native or recommended rate;
                after setup() the attribute holds the rate actually in effect
            leads: List of leads to use (e.g., ['I', 'II', 'III'] for standard leads)
            transform: Optional transform to be applied to the ECG signals
            target_transform: Optional transform to be applied to the labels
            download: Whether to download the dataset if it doesn't exist
            batch_size: Batch size for data loaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            num_workers: Number of worker processes for data loading. The built-in
                datasets hold their signals in memory, so the default of 0 is the
                fastest setting; with more, guard the calling script with
                ``if __name__ == "__main__":`` on macOS and Windows
            seed: Random seed for reproducibility
            stratify: Whether to use stratified splitting based on labels
            dataset_kwargs: Additional keyword arguments to pass to the dataset class
        """
        self.dataset = dataset
        self.dataset_class = dataset_class
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.leads = leads
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.seed = seed
        self.stratify = stratify
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.verbose = verbose
        self.dataset_kwargs = dataset_kwargs or {}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if self.dataset is None:
            if self.dataset_class is None:
                raise ValueError("dataset_class must be provided if dataset is None")

            if self.data_dir is None:
                self.data_dir = self.dataset_class.get_default_data_dir()

            init_kwargs = {
                "data_dir": self.data_dir,
                "transform": self.transform,
                "target_transform": self.target_transform,
                "download": self.download,
            }

            # Only an explicit rate overrides the dataset's own default: a
            # blanket value here would silently resample every dataset.
            if self.sampling_rate is not None:
                init_kwargs["sampling_rate"] = self.sampling_rate

            if self.leads is not None:
                init_kwargs["leads"] = self.leads

            init_kwargs.update(self.dataset_kwargs)

            self.dataset = self.dataset_class(**init_kwargs)

        effective_rate = getattr(self.dataset, "sampling_rate", None)
        if effective_rate is not None:
            self.sampling_rate = effective_rate

        stratify_labels = None
        if (
            self.stratify and len(self.dataset) >= 8
        ):  # Need at least 2 samples per class for 4 classes
            try:
                all_labels = []
                for i in range(len(self.dataset)):
                    _, label = self.dataset[i]
                    all_labels.append(label)
                stacked = torch.stack(all_labels).numpy()

                # For multi-label (2D), convert rows to string keys for stratification
                if stacked.ndim > 1:
                    stratify_labels = np.array(
                        ["_".join(str(int(v)) for v in row) for row in stacked]
                    )
                else:
                    stratify_labels = stacked

                # Check if we have enough samples per class for stratification
                _, counts = np.unique(stratify_labels, return_counts=True)
                if np.min(counts) < 2:
                    warnings.warn(
                        "Stratified splitting disabled: at least one class has fewer "
                        "than 2 samples.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    stratify_labels = None
            except (RuntimeError, ValueError, TypeError, IndexError) as error:
                warnings.warn(
                    f"Stratified splitting disabled: could not collect labels ({error}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                stratify_labels = None

        # Extract groups for patient-level splitting (prevents data leakage)
        groups = None
        group_ids = getattr(self.dataset, "group_ids", None) or getattr(
            self.dataset, "record_names", None
        )
        if group_ids:
            if len(group_ids) != len(self.dataset):
                raise ValueError(
                    f"{type(self.dataset).__name__} exposes {len(group_ids)} group ids "
                    f"for {len(self.dataset)} samples. Grouped splitting cannot be "
                    "performed safely."
                )
            groups = np.array(group_ids)
            if self.verbose:
                n_groups = len(np.unique(groups))
                logger.info(f"Using grouped splitting ({n_groups} groups)")
        else:
            warnings.warn(
                f"{type(self.dataset).__name__} exposes no group ids, so the split is "
                "performed at sample level. If samples are segments of longer recordings, "
                "segments of the same recording will appear in both train and test and the "
                "reported scores will be optimistically biased.",
                RuntimeWarning,
                stacklevel=2,
            )

        splitter = DataSplitter(
            dataset=self.dataset,
            val_split=self.val_split,
            test_split=self.test_split,
            seed=self.seed,
            stratify=stratify_labels,
            groups=groups,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = splitter.split()

        if self.verbose:
            logger.info(f"Dataset size: {len(self.train_dataset)}")
            logger.info(f"Validation set size: {len(self.val_dataset)}")
            logger.info(f"Test set size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before using the data module")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before using the data module")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before using the data module")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def describe(self) -> dict[str, Any]:
        """Describe the data a model is trained on, in the form checkpoints record.

        ECGTrainer stores this as its data_config, which is what lets a
        checkpoint be used on new recordings: it names the preprocessing the
        inputs need and the classes the outputs stand for.

        Returns:
            Mapping with the registered dataset name, its preprocessing
            (sampling rate, segment length, normalisation), the class names,
            the sample shape, the labelling mode, and the split protocol.
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before describing the data module")
        dataset = self.dataset

        input_channels = input_length = None
        if len(dataset) > 0:
            sample = dataset[0][0]
            if getattr(sample, "ndim", 0) >= 2:
                input_channels, input_length = int(sample.shape[0]), int(sample.shape[-1])

        get_preprocessing = getattr(dataset, "get_preprocessing", None)
        class_names = getattr(dataset, "class_names", None)
        return {
            "dataset": getattr(type(dataset), "_registry_name", None),
            "preprocessing": get_preprocessing() if callable(get_preprocessing) else None,
            "class_names": list(class_names) if class_names else None,
            "input_channels": input_channels,
            "input_length": input_length,
            "multi_label": bool(getattr(dataset, "multi_label", False)),
            "binary_classification": bool(getattr(dataset, "binary_classification", False)),
            "split": {
                "seed": self.seed,
                "val_split": self.val_split,
                "test_split": self.test_split,
            },
        }

    def get_metadata(self) -> dict:
        """Get metadata about the dataset.

        Returns:
            Dictionary containing dataset metadata
        """
        if self.dataset is None:
            raise RuntimeError("Call setup() before getting metadata")
        if isinstance(self.dataset, BaseECGDataset):
            return self.dataset.get_metadata()
        return {
            "dataset_size": len(self.dataset),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def print_metadata(self) -> None:
        """Print metadata about the dataset."""
        print(self.get_metadata())
