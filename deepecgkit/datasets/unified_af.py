import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from deepecgkit.registry import register_dataset

from .af_classification import AFClassificationDataset
from .base import BaseECGDataset
from .ltafdb import LTAFDBDataset
from .mitbih_afdb import MITBIHAFDBDataset

logger = logging.getLogger(__name__)

# PhysioNet 2017 uses labels: N=0, A=1, O=2, ~=3
# Unified 4-class scheme:     Normal=0, AF=1, AFL=2, J=3
# "Other" (O) and "Noisy" (~) have no unified equivalent → dropped
_PHYSIONET2017_LABEL_REMAP = {
    0: 0,  # N (Normal) → Normal
    1: 1,  # A (AF) → AF
    2: None,  # O (Other) → drop
    3: None,  # ~ (Noisy) → drop
}

# Binary scheme: Non-AF=0, AF=1
# "Other" maps to Non-AF, "Noisy" is still dropped
_PHYSIONET2017_BINARY_REMAP = {
    0: 0,  # N (Normal) → Non-AF
    1: 1,  # A (AF) → AF
    2: 0,  # O (Other) → Non-AF
    3: None,  # ~ (Noisy) → drop
}


class _RemappedDataset(Dataset):
    """Wraps a dataset with label remapping and sample filtering.

    Samples whose labels map to None are excluded from the dataset.
    """

    def __init__(
        self,
        dataset: Any,
        label_map: dict[int, int | None],
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self._dataset = dataset
        self._label_map = label_map
        self._target_transform = target_transform

        # Filter out samples whose labels map to None
        if hasattr(dataset, "labels"):
            self._indices = [
                i for i, lbl in enumerate(dataset.labels) if label_map.get(int(lbl)) is not None
            ]
            self.labels = [label_map[int(dataset.labels[i])] for i in self._indices]
        else:
            self._indices = list(range(len(dataset)))
            self.labels = []

        source_records = getattr(dataset, "record_names", None)
        if source_records:
            self.record_names = [source_records[i] for i in self._indices]
        else:
            self.record_names = []

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal, label = self._dataset[self._indices[idx]]
        new_label = torch.tensor(self._label_map[label.item()], dtype=torch.long)
        if self._target_transform is not None:
            new_label = self._target_transform(new_label)
        return signal, new_label

    def __len__(self) -> int:
        return len(self._indices)


@register_dataset(
    name="unified-af",
    input_channels=1,
    num_classes=4,
    description="Unified AF dataset combining PhysioNet 2017, MIT-BIH, LTAFDB",
)
class UnifiedAFDataset(BaseECGDataset):
    """Unified AF Dataset combining multiple PhysioNet AF databases.

    Combines samples from the PhysioNet 2017 Challenge, MIT-BIH AFDB, and LTAFDB
    into a single dataset for AF classification. Supports both binary (AF vs Non-AF)
    and 4-class (Normal, AF, AFL, J) classification modes.

    PhysioNet 2017 labels are remapped to the unified scheme:
      - Normal (N) → Normal, AF (A) → AF
      - Other (O) and Noisy (~) are dropped in 4-class mode
      - Other (O) → Non-AF and Noisy (~) is dropped in binary mode
    """

    CLASS_LABELS: ClassVar[list[str]] = ["Normal", "AF", "AFL", "J"]
    LEADS: ClassVar[list[str]] = ["ECG"]

    AVAILABLE_DATASETS: ClassVar[dict[str, type[BaseECGDataset]]] = {
        "physionet2017": AFClassificationDataset,
        "mitbih_afdb": MITBIHAFDBDataset,
        "ltafdb": LTAFDBDataset,
    }
    DOWNLOAD_SIZE: ClassVar[str | None] = "about 4.4 GB across its three sources"

    @classmethod
    def is_downloaded(cls, data_dir: str | Path | None = None) -> bool:
        """Report whether every source dataset is present in its own directory."""
        return all(
            dataset_class.is_downloaded(dataset_class.get_default_data_dir())
            for dataset_class in cls.AVAILABLE_DATASETS.values()
        )

    def _resolve_dataset_dir(self, dataset_name: str) -> Path:
        dataset_class = self.AVAILABLE_DATASETS[dataset_name]
        return dataset_class.get_default_data_dir()

    def __init__(
        self,
        data_dir: str | Path | None = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        datasets: list[str] | None = None,
        binary_classification: bool = False,
        normalization: str = "zscore",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        download: bool = False,
        force_download: bool = False,
        verbose: bool = True,
        dataset_kwargs: dict[str, dict] | None = None,
    ):
        self.segment_duration_seconds = segment_duration_seconds
        self.binary_classification = binary_classification
        self.normalization = normalization
        self.verbose = verbose
        self.dataset_kwargs = dataset_kwargs or {}

        if datasets is None:
            datasets = ["physionet2017", "mitbih_afdb", "ltafdb"]

        for dataset_name in datasets:
            if dataset_name not in self.AVAILABLE_DATASETS:
                raise ValueError(
                    f"Unknown dataset: {dataset_name}. "
                    f"Available: {list(self.AVAILABLE_DATASETS.keys())}"
                )

        self.dataset_names = datasets
        self.datasets: list[Any] = []
        self.dataset_sizes: list[int] = []
        self.loaded_dataset_names: list[str] = []
        self.record_names: list[str] = []

        super().__init__(
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            leads=self.LEADS,
            transform=transform,
            target_transform=target_transform,
            download=download,
            force_download=force_download,
        )

    def download(self) -> None:
        for dataset_name in self.dataset_names:
            dataset_class = self.AVAILABLE_DATASETS[dataset_name]
            dataset_dir = self._resolve_dataset_dir(dataset_name)

            if dataset_class.is_downloaded(dataset_dir):
                if self.verbose:
                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"Skipping {dataset_name}: already exists at {dataset_dir}")
                    logger.info(f"{'=' * 60}")
                continue

            if self.verbose:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Downloading dataset: {dataset_name}")
                logger.info(f"{'=' * 60}")

            dataset_class.download_to(dataset_dir, verbose=self.verbose)

    def _get_dataset_kwargs(self, dataset_name: str) -> dict:
        base_kwargs = {
            "sampling_rate": self.sampling_rate,
            "transform": self.transform,
            "segment_duration_seconds": self.segment_duration_seconds,
            "normalization": self.normalization,
            "download": False,
        }

        if dataset_name in ["mitbih_afdb", "ltafdb"]:
            base_kwargs["target_transform"] = self.target_transform
            base_kwargs["binary_classification"] = self.binary_classification
        elif dataset_name == "physionet2017":
            # target_transform is omitted — applied after label remapping
            # in _RemappedDataset (see _load_data)
            pass

        if dataset_name in self.dataset_kwargs:
            base_kwargs.update(self.dataset_kwargs[dataset_name])

        return base_kwargs

    def _load_data(self) -> None:
        if self.verbose:
            logger.info("\nLoading unified AF dataset...")

        for dataset_name in self.dataset_names:
            if self.verbose:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Loading: {dataset_name}")
                logger.info(f"{'=' * 60}")

            dataset_class = self.AVAILABLE_DATASETS[dataset_name]
            dataset_dir = self._resolve_dataset_dir(dataset_name)

            kwargs = self._get_dataset_kwargs(dataset_name)
            kwargs["data_dir"] = dataset_dir
            kwargs["verbose"] = self.verbose

            try:
                dataset = dataset_class(**kwargs)

                # Remap PhysioNet 2017 labels to the unified scheme
                if dataset_name == "physionet2017":
                    remap = (
                        _PHYSIONET2017_BINARY_REMAP
                        if self.binary_classification
                        else _PHYSIONET2017_LABEL_REMAP
                    )
                    original_len = len(dataset)
                    dataset = _RemappedDataset(
                        dataset, remap, target_transform=self.target_transform
                    )
                    dropped = original_len - len(dataset)
                    if dropped > 0 and self.verbose:
                        logger.info(
                            f"  Dropped {dropped} samples with unmappable labels (Other/Noisy)"
                        )

                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))
                self.loaded_dataset_names.append(dataset_name)

                if self.verbose:
                    logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
                    if hasattr(dataset, "get_class_distribution"):
                        dist = dataset.get_class_distribution()
                        logger.info(f"Class distribution: {dist}")

            except Exception as e:
                logger.warning("Failed to load %s: %s", dataset_name, e)
                continue

        if len(self.datasets) == 0:
            raise RuntimeError("No datasets were successfully loaded")

        self.concat_dataset: ConcatDataset = ConcatDataset(self.datasets)
        self._collect_record_names()

        if self.verbose:
            logger.info(f"\n{'=' * 60}")
            logger.info("Unified dataset created")
            logger.info(f"Total datasets: {len(self.datasets)}")
            logger.info(f"Total samples: {len(self.concat_dataset)}")
            logger.info(f"{'=' * 60}")
            self._print_overall_distribution()

    def _collect_record_names(self) -> None:
        """Build globally unique record identifiers across the merged source datasets.

        Record names are namespaced by source dataset so that identically named
        records in different sources are never treated as the same group. These
        identifiers are what ECGDataModule uses to keep every segment of a
        recording inside a single split; without them the merged dataset would be
        split at segment level and leak between train and test.

        Raises:
            RuntimeError: If any source dataset does not expose usable record names.
        """
        record_names: list[str] = []
        for dataset_name, dataset in zip(self.loaded_dataset_names, self.datasets, strict=False):
            source_records = getattr(dataset, "record_names", None)
            if not source_records or len(source_records) != len(dataset):
                raise RuntimeError(
                    f"Source dataset '{dataset_name}' exposes "
                    f"{0 if not source_records else len(source_records)} record names for "
                    f"{len(dataset)} samples. Record-level splitting cannot be performed "
                    "safely, and segment-level splitting would leak recordings between "
                    "train and test."
                )
            record_names.extend(f"{dataset_name}/{name}" for name in source_records)

        self.record_names = record_names

    def _print_overall_distribution(self) -> None:
        if len(self.datasets) == 0:
            return

        all_labels = []
        for dataset in self.datasets:
            if hasattr(dataset, "labels"):
                all_labels.extend(dataset.labels)

        if len(all_labels) == 0:
            return

        unique, counts = np.unique(all_labels, return_counts=True)
        logger.info("\nOverall class distribution:")
        for label_idx, count in zip(unique, counts, strict=False):
            class_name = self.class_names[label_idx]
            percentage = (count / len(all_labels)) * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.concat_dataset[idx]

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def get_preprocessing(self) -> dict:
        """Describe the preprocessing shared by every source dataset."""
        preprocessing = super().get_preprocessing()
        preprocessing["normalization"] = self.normalization
        return preprocessing

    def get_dataset_info(self) -> dict:
        return {
            "num_datasets": len(self.datasets),
            "dataset_names": self.dataset_names,
            "dataset_sizes": self.dataset_sizes,
            "total_samples": len(self),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

    def get_class_distribution(self) -> dict[str, int]:
        all_labels = []
        for dataset in self.datasets:
            if hasattr(dataset, "labels"):
                all_labels.extend(dataset.labels)

        if len(all_labels) == 0:
            return {}

        unique, counts = np.unique(all_labels, return_counts=True)
        return {
            self.class_names[int(label)]: int(count)
            for label, count in zip(unique, counts, strict=False)
        }
