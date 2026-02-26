"""
Data splitting and loader creation utilities.

This module provides classes for splitting datasets and creating data loaders.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, random_split


class DataSplitter:
    """Handles dataset splitting into train, validation, and test sets."""

    def __init__(
        self,
        dataset: Dataset,
        val_split: float = 0.2,
        test_split: float = 0.1,
        seed: Optional[int] = 42,
        stratify: Optional[np.ndarray] = None,
    ):
        """Initialize the data splitter.

        Args:
            dataset: PyTorch dataset to split
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
            stratify: Array of labels for stratified splitting (optional)
        """
        self.dataset = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.stratify = stratify

        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be less than 1.0")

    def split(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Split the dataset into train, validation, and test sets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        total_size = len(self.dataset)

        # Handle small datasets by ensuring at least 1 sample per split if possible
        val_size = max(1, int(total_size * self.val_split)) if self.val_split > 0 else 0
        test_size = max(1, int(total_size * self.test_split)) if self.test_split > 0 else 0

        # Adjust for very small datasets
        if total_size <= 2:
            val_size = 0
            test_size = 0
        elif total_size == 3:
            val_size = 1
            test_size = 0
        elif val_size + test_size >= total_size:
            # Reduce splits for small datasets
            val_size = 1 if self.val_split > 0 else 0
            test_size = 1 if self.test_split > 0 and total_size > val_size + 1 else 0

        train_size = total_size - val_size - test_size

        if self.stratify is not None and total_size > 2:
            indices = np.arange(total_size)
            if val_size + test_size > 0:
                train_idx, temp_idx = train_test_split(
                    indices,
                    test_size=(val_size + test_size) / total_size,
                    stratify=self.stratify,
                    random_state=self.seed,
                )
                if test_size > 0 and val_size > 0:
                    val_idx, test_idx = train_test_split(
                        temp_idx,
                        test_size=test_size / (val_size + test_size),
                        stratify=self.stratify[temp_idx] if self.stratify is not None else None,
                        random_state=self.seed,
                    )
                elif test_size > 0:
                    test_idx = temp_idx
                    val_idx = []
                else:
                    val_idx = temp_idx
                    test_idx = []
            else:
                train_idx = indices
                val_idx = []
                test_idx = []

            train_dataset = Subset(self.dataset, train_idx)
            val_dataset = (
                Subset(self.dataset, val_idx) if len(val_idx) > 0 else Subset(self.dataset, [])
            )
            test_dataset = (
                Subset(self.dataset, test_idx) if len(test_idx) > 0 else Subset(self.dataset, [])
            )
        # Use random splitting
        elif val_size + test_size == 0:
            train_dataset = self.dataset
            val_dataset = Subset(self.dataset, [])
            test_dataset = Subset(self.dataset, [])
        else:
            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        return train_dataset, val_dataset, test_dataset
