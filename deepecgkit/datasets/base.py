import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class BaseECGDataset(Dataset, ABC):
    """Base class for all ECG datasets in deepecg-kit.

    This class defines the common interface and functionality for all ECG datasets.
    Each specific dataset implementation should inherit from this class and implement
    the required methods.
    """

    @classmethod
    def get_default_data_dir(cls) -> Path:
        """Get the default data directory for this dataset.

        Returns:
            Path to the default data directory
        """
        return Path.home() / ".deepecgkit" / "datasets" / cls.__name__.lower()

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        sampling_rate: int = 500,
        leads: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        force_download: bool = False,
    ):
        """Initialize the base ECG dataset.

        Args:
            data_dir: Directory where the dataset is stored or will be downloaded.
                     If None, uses the default data directory.
            sampling_rate: Sampling rate of the ECG signals (Hz)
            leads: List of leads to use (e.g., ['I', 'II', 'III'] for standard leads)
            transform: Optional transform to be applied to the ECG signals
            target_transform: Optional transform to be applied to the labels
            download: Whether to download the dataset if it doesn't exist locally.
            force_download: Whether to force re-download even if the dataset exists
        """
        self.data_dir = Path(data_dir) if data_dir is not None else self.get_default_data_dir()
        self.sampling_rate = sampling_rate
        self.leads = leads
        self.transform = transform
        self.target_transform = target_transform
        self.home_dir = Path.home()

        if force_download:
            self._clear_dataset()
            self.download()
        elif not self.data_dir.exists():
            if download:
                self.download()

        self._load_data()

    def _clear_dataset(self):
        """Clear the dataset directory for re-download."""
        if self.data_dir.exists():
            print(f"Clearing existing dataset at {self.data_dir}...")
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def download(self):
        """Download the dataset if it doesn't exist."""
        pass

    @abstractmethod
    def _load_data(self):
        """Load the dataset into memory."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple of (ecg_signal, label) where:
                - ecg_signal: Tensor of shape (num_leads, signal_length)
                - label: Tensor of shape (num_classes,) for classification or (num_beats,) for segmentation
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Get the names of the classes in the dataset."""
        pass

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        return {}

    def _print_class_distribution(self):
        """Print class distribution statistics."""
        distribution = self.get_class_distribution()
        if not distribution:
            return
        print("\nClass distribution:")
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}")

    def get_metadata(self) -> Dict:
        """Get metadata about the dataset.

        Returns:
            Dictionary containing metadata such as:
            - sampling_rate: Sampling rate of the signals
            - num_leads: Number of leads
            - lead_names: Names of the leads
            - num_classes: Number of classes
            - class_names: Names of the classes
            - dataset_size: Number of samples
            - signal_length: Length of each signal
        """
        return {
            "sampling_rate": self.sampling_rate,
            "num_leads": len(self.leads) if self.leads else None,
            "lead_names": self.leads,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "dataset_size": len(self),
        }
