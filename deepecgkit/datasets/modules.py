"""
PyTorch Lightning data module for ECG datasets.
"""

from typing import Dict, Optional, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from deepecgkit.datasets.base import BaseECGDataset
from deepecgkit.datasets.splitting import DataSplitter


class ECGDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for ECG datasets."""

    def __init__(
        self,
        dataset: Optional[Union[BaseECGDataset, Dataset]] = None,
        dataset_class: Optional[Type[BaseECGDataset]] = None,
        data_dir: Optional[str] = None,
        sampling_rate: int = 500,
        leads: Optional[list] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
        stratify: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        verbose: bool = True,
        dataset_kwargs: Optional[Dict] = None,
    ):
        """Initialize the ECG data module.

        Args:
            dataset: PyTorch dataset instance (optional)
            dataset_class: Class of dataset to create (optional)
            data_dir: Directory containing ECG data files (optional, uses dataset's default if None)
            sampling_rate: Sampling rate of the ECG signals (Hz)
            leads: List of leads to use (e.g., ['I', 'II', 'III'] for standard leads)
            transform: Optional transform to be applied to the ECG signals
            target_transform: Optional transform to be applied to the labels
            download: Whether to download the dataset if it doesn't exist
            batch_size: Batch size for data loaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            num_workers: Number of worker processes for data loading
            seed: Random seed for reproducibility
            stratify: Whether to use stratified splitting based on labels
            dataset_kwargs: Additional keyword arguments to pass to the dataset class
        """
        super().__init__()
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

    def setup(self, stage: Optional[str] = None):
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
                "sampling_rate": self.sampling_rate,
                "transform": self.transform,
                "target_transform": self.target_transform,
                "download": self.download,
            }

            if self.leads is not None:
                init_kwargs["leads"] = self.leads

            init_kwargs.update(self.dataset_kwargs)

            self.dataset = self.dataset_class(**init_kwargs)

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
                unique_labels, counts = np.unique(stratify_labels, return_counts=True)
                if np.min(counts) < 2:
                    stratify_labels = None  # Disable stratification
            except Exception:
                stratify_labels = None  # Fallback to no stratification

        splitter = DataSplitter(
            dataset=self.dataset,
            val_split=self.val_split,
            test_split=self.test_split,
            seed=self.seed,
            stratify=stratify_labels,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = splitter.split()

        if self.verbose:
            print(f"Dataset size: {len(self.train_dataset)}")
            print(f"Validation set size: {len(self.val_dataset)}")
            print(f"Test set size: {len(self.test_dataset)}")

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

    def get_metadata(self) -> Dict:
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

    def print_metadata(self):
        """Print metadata about the dataset."""
        print(self.get_metadata())
