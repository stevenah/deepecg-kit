"""
Data splitting and loader creation utilities.

This module provides classes for splitting datasets and creating data loaders.
"""

from collections import Counter
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
        groups: Optional[np.ndarray] = None,
    ):
        """Initialize the data splitter.

        Args:
            dataset: PyTorch dataset to split
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            seed: Random seed for reproducibility
            stratify: Array of labels for stratified splitting (optional)
            groups: Array of group identifiers (e.g. patient/record IDs) to keep
                all samples from the same group in the same split (optional).
                Prevents data leakage from correlated samples.
        """
        self.dataset = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.stratify = stratify
        self.groups = groups

        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be less than 1.0")

    def split(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Split the dataset into train, validation, and test sets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.groups is not None and len(self.groups) > 0:
            return self._split_by_groups()

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
            try:
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
                            stratify=self.stratify[temp_idx],
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
                    Subset(self.dataset, test_idx)
                    if len(test_idx) > 0
                    else Subset(self.dataset, [])
                )
                return train_dataset, val_dataset, test_dataset
            except ValueError:
                pass  # Fall through to random splitting

        if val_size + test_size == 0:
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

    def _split_by_groups(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Split by groups so all samples from one group stay in the same split.

        This implements inter-patient splitting for medical datasets where
        segments from the same recording/patient are highly correlated.
        """
        unique_groups = np.unique(self.groups)
        n_groups = len(unique_groups)

        if n_groups < 3:
            raise ValueError(f"Need at least 3 groups for train/val/test split, got {n_groups}")

        val_n = max(1, int(n_groups * self.val_split)) if self.val_split > 0 else 0
        test_n = max(1, int(n_groups * self.test_split)) if self.test_split > 0 else 0

        if val_n + test_n >= n_groups:
            val_n = 1 if self.val_split > 0 else 0
            test_n = 1 if self.test_split > 0 and n_groups > val_n + 1 else 0

        # Compute majority label per group for stratified group splitting
        group_stratify = None
        if self.stratify is not None:
            group_to_labels: dict = {}
            for g, label in zip(self.groups, self.stratify):
                group_to_labels.setdefault(g, []).append(label)
            group_stratify = np.array(
                [Counter(group_to_labels[g]).most_common(1)[0][0] for g in unique_groups]
            )
            _, counts = np.unique(group_stratify, return_counts=True)
            if np.min(counts) < 2:
                group_stratify = None

        group_indices = np.arange(n_groups)

        try:
            train_gi, temp_gi = train_test_split(
                group_indices,
                test_size=(val_n + test_n) / n_groups,
                stratify=group_stratify,
                random_state=self.seed,
            )
            if val_n > 0 and test_n > 0:
                temp_stratify = group_stratify[temp_gi] if group_stratify is not None else None
                if temp_stratify is not None:
                    _, tc = np.unique(temp_stratify, return_counts=True)
                    if np.min(tc) < 2:
                        temp_stratify = None
                val_gi, test_gi = train_test_split(
                    temp_gi,
                    test_size=test_n / (val_n + test_n),
                    stratify=temp_stratify,
                    random_state=self.seed,
                )
            elif test_n > 0:
                test_gi = temp_gi
                val_gi = np.array([], dtype=int)
            else:
                val_gi = temp_gi
                test_gi = np.array([], dtype=int)
        except ValueError:
            # Stratification failed, split without it
            train_gi, temp_gi = train_test_split(
                group_indices,
                test_size=(val_n + test_n) / n_groups,
                random_state=self.seed,
            )
            if val_n > 0 and test_n > 0:
                val_gi, test_gi = train_test_split(
                    temp_gi,
                    test_size=test_n / (val_n + test_n),
                    random_state=self.seed,
                )
            elif test_n > 0:
                test_gi = temp_gi
                val_gi = np.array([], dtype=int)
            else:
                val_gi = temp_gi
                test_gi = np.array([], dtype=int)

        # Map group indices back to sample indices
        train_groups = set(unique_groups[train_gi])
        val_groups = set(unique_groups[val_gi]) if len(val_gi) > 0 else set()
        test_groups = set(unique_groups[test_gi]) if len(test_gi) > 0 else set()

        train_idx = [i for i, g in enumerate(self.groups) if g in train_groups]
        val_idx = [i for i, g in enumerate(self.groups) if g in val_groups]
        test_idx = [i for i, g in enumerate(self.groups) if g in test_groups]

        return (
            Subset(self.dataset, train_idx),
            Subset(self.dataset, val_idx) if val_idx else Subset(self.dataset, []),
            Subset(self.dataset, test_idx) if test_idx else Subset(self.dataset, []),
        )
