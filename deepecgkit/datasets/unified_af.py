import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Type, Union

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
        dataset,
        label_map: Dict[int, Optional[int]],
        target_transform: Optional[callable] = None,
    ):
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

    CLASS_LABELS: ClassVar[List[str]] = ["Normal", "AF", "AFL", "J"]
    LEADS: ClassVar[List[str]] = ["ECG"]

    AVAILABLE_DATASETS: ClassVar[Dict[str, Type[BaseECGDataset]]] = {
        "physionet2017": AFClassificationDataset,
        "mitbih_afdb": MITBIHAFDBDataset,
        "ltafdb": LTAFDBDataset,
    }

    def _resolve_dataset_dir(self, dataset_name: str) -> Path:
        dataset_class = self.AVAILABLE_DATASETS[dataset_name]
        return dataset_class.get_default_data_dir()

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        datasets: Optional[List[str]] = None,
        binary_classification: bool = False,
        normalization: str = "zscore",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        force_download: bool = False,
        verbose: bool = True,
        dataset_kwargs: Optional[Dict[str, Dict]] = None,
    ):
        self.segment_duration_seconds = segment_duration_seconds
        self.binary_classification = binary_classification
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
        self.datasets = []
        self.dataset_sizes = []

        super().__init__(
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            leads=self.LEADS,
            transform=transform,
            target_transform=target_transform,
            download=download,
            force_download=force_download,
        )

    def download(self):
        for dataset_name in self.dataset_names:
            dataset_class = self.AVAILABLE_DATASETS[dataset_name]
            dataset_dir = self._resolve_dataset_dir(dataset_name)

            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                if self.verbose:
                    print(f"\n{'=' * 60}")
                    print(f"Skipping {dataset_name}: already exists at {dataset_dir}")
                    print(f"{'=' * 60}")
                continue

            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Downloading dataset: {dataset_name}")
                print(f"{'=' * 60}")

            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Minimal instance for download only — sub-dataset download()
            # methods only require self.data_dir and self.verbose.
            instance = object.__new__(dataset_class)
            instance.data_dir = dataset_dir
            instance.verbose = self.verbose
            instance.download()

    def _get_dataset_kwargs(self, dataset_name: str) -> Dict:
        base_kwargs = {
            "sampling_rate": self.sampling_rate,
            "transform": self.transform,
            "segment_duration_seconds": self.segment_duration_seconds,
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

    def _load_data(self):
        if self.verbose:
            print("\nLoading unified AF dataset...")

        for dataset_name in self.dataset_names:
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Loading: {dataset_name}")
                print(f"{'=' * 60}")

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
                        print(f"  Dropped {dropped} samples with unmappable labels (Other/Noisy)")

                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))

                if self.verbose:
                    print(f"Loaded {len(dataset)} samples from {dataset_name}")
                    if hasattr(dataset, "get_class_distribution"):
                        dist = dataset.get_class_distribution()
                        print(f"Class distribution: {dist}")

            except Exception as e:
                logger.warning("Failed to load %s: %s", dataset_name, e)
                if self.verbose:
                    print(f"Failed to load {dataset_name}: {e}")
                continue

        if len(self.datasets) == 0:
            raise RuntimeError("No datasets were successfully loaded")

        self.concat_dataset = ConcatDataset(self.datasets)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print("Unified dataset created")
            print(f"Total datasets: {len(self.datasets)}")
            print(f"Total samples: {len(self.concat_dataset)}")
            print(f"{'=' * 60}")
            self._print_overall_distribution()

    def _print_overall_distribution(self):
        if len(self.datasets) == 0:
            return

        all_labels = []
        for dataset in self.datasets:
            if hasattr(dataset, "labels"):
                all_labels.extend(dataset.labels)

        if len(all_labels) == 0:
            return

        unique, counts = np.unique(all_labels, return_counts=True)
        print("\nOverall class distribution:")
        for label_idx, count in zip(unique, counts):
            class_name = self.class_names[label_idx]
            percentage = (count / len(all_labels)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.concat_dataset[idx]

    def __len__(self) -> int:
        return len(self.concat_dataset)

    @property
    def num_classes(self) -> int:
        return 2 if self.binary_classification else len(self.CLASS_LABELS)

    @property
    def class_names(self) -> List[str]:
        if self.binary_classification:
            return ["Non-AF", "AF"]
        return self.CLASS_LABELS

    def get_dataset_info(self) -> Dict:
        return {
            "num_datasets": len(self.datasets),
            "dataset_names": self.dataset_names,
            "dataset_sizes": self.dataset_sizes,
            "total_samples": len(self),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        all_labels = []
        for dataset in self.datasets:
            if hasattr(dataset, "labels"):
                all_labels.extend(dataset.labels)

        if len(all_labels) == 0:
            return {}

        unique, counts = np.unique(all_labels, return_counts=True)
        return {self.class_names[int(label)]: int(count) for label, count in zip(unique, counts)}
