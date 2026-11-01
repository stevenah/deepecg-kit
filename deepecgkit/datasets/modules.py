"""
Data module for ECG datasets.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from deepecgkit.datasets.base import BaseECGDataset
from deepecgkit.datasets.splitting import DataSplitter
from deepecgkit.registry import get_dataset

logger = logging.getLogger(__name__)

#: How the partitions were formed: the dataset's own recommended assignment,
#: or ECGDataModule's seeded, grouped and stratified split.
SPLIT_PREDEFINED = "predefined"
SPLIT_SEEDED = "seeded"
PARTITIONS = ("train", "val", "test")

#: normalization_stats value asking setup() to fit the statistics on the training partition.
FIT_STATS = "fit"


class GlobalStandardize:
    """Subtract one mean and divide by one standard deviation, the same for every lead.

    Unlike per-signal z-scoring, the amplitude of a signal relative to the
    training data is kept.

    Args:
        mean: Mean of every value of the training partition
        std: Standard deviation of every value of the training partition
        then: Transform applied after standardising, if any
    """

    def __init__(
        self,
        mean: float,
        std: float,
        then: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.mean = mean
        self.std = std
        self.then = then

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        signal = (signal - self.mean) / self.std
        return self.then(signal) if self.then is not None else signal


class RandomCrop(Dataset):
    """A view of a dataset whose signals are cut to random windows of fixed length.

    A new window is drawn each time a sample is read, from torch's random
    number generator, so a seeded run draws the same windows.

    Args:
        dataset: The dataset to crop, yielding (signal, label) pairs
        length: Window length in samples
    """

    def __init__(self, dataset: Dataset, length: int):
        self.dataset = dataset
        self.length = length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        signal, label = self.dataset[index]
        excess = signal.shape[-1] - self.length
        if excess < 0:
            raise ValueError(
                f"Cannot crop {self.length} samples from a signal of {signal.shape[-1]}."
            )
        start = int(torch.randint(0, excess + 1, (1,)).item()) if excess else 0
        return signal[..., start : start + self.length], label


class ECGDataModule:
    """Data module for ECG datasets.

    Handles dataset creation, train/val/test splitting, and DataLoader construction.

    Example:
        >>> data = ECGDataModule("af-classification", download=True)
        >>> model = build_model("kanres", data.input_channels, data.num_classes)
    """

    def __init__(
        self,
        dataset: str | type[BaseECGDataset] | BaseECGDataset | Dataset | None = None,
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
        normalization_stats: str | tuple[float, float] | None = None,
        train_crop_length: int | None = None,
    ):
        """Initialize the ECG data module.

        Args:
            dataset: A registered dataset name such as "af-classification"
                (see get_dataset_names), a dataset class, or a dataset instance
            dataset_class: Class of dataset to create, the same as passing the
                class as dataset
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
            normalization_stats: Normalise every partition with one mean and one
                standard deviation over all values: "fit" computes them on the
                training partition, a (mean, std) pair applies given ones, as a
                checkpoint records them. None leaves the dataset's own
                normalisation, which should then be "none". After setup() the
                attribute holds the statistics in effect
            train_crop_length: Cut every training sample to a random window of
                this many samples, drawn anew each epoch; validation and test
                samples are left whole

        Raises:
            KeyError: If dataset is a name nobody registered.
            ValueError: If a dataset is given both as dataset and as dataset_class.
        """
        if isinstance(dataset, str):
            dataset = get_dataset(dataset)
        if isinstance(dataset, type):
            if dataset_class is not None:
                raise ValueError("Pass the dataset once, as dataset or as dataset_class.")
            dataset_class, dataset = dataset, None

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
        self.normalization_stats = normalization_stats
        self._fit_stats = normalization_stats == FIT_STATS
        self._base_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._base_transform_saved = False
        self.train_crop_length = train_crop_length

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.split_protocol: str | None = None
        self._split_settings: tuple | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets.

        The dataset is loaded and split on the first call. Later calls, which
        the trainer makes for every stage, return at once unless the dataset or
        a split setting has changed since.

        A dataset that names a recommended partition for every sample (see
        BaseECGDataset.predefined_partition) is split that way, and the split
        fractions are then unused; the seed still sets initialisation and
        shuffling in the trainer.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if self.dataset is None:
            if self.dataset_class is None:
                raise ValueError(
                    "No dataset given: pass a dataset name such as 'af-classification', "
                    "a dataset class, or a dataset instance."
                )

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

        split_settings = (
            id(self.dataset),
            self.val_split,
            self.test_split,
            self.seed,
            self.stratify,
        )
        if self.train_dataset is not None and split_settings == self._split_settings:
            return
        self._split_settings = split_settings

        partition = self._predefined_partition()
        if partition is not None:
            self.split_protocol = SPLIT_PREDEFINED
            self.train_dataset, self.val_dataset, self.test_dataset = (
                Subset(self.dataset, np.flatnonzero(partition == name).tolist())
                for name in PARTITIONS
            )
            if self.verbose:
                logger.info("Using the dataset's recommended split")
                logger.info(f"Training set size: {len(self.train_dataset)}")
                logger.info(f"Validation set size: {len(self.val_dataset)}")
                logger.info(f"Test set size: {len(self.test_dataset)}")
            self._apply_normalization_stats()
            return
        self.split_protocol = SPLIT_SEEDED

        stratify_labels = None
        if (
            self.stratify and len(self.dataset) >= 8
        ):  # Need at least 2 samples per class for 4 classes
            try:
                stacked = self._collect_labels()

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
            logger.info(f"Training set size: {len(self.train_dataset)}")
            logger.info(f"Validation set size: {len(self.val_dataset)}")
            logger.info(f"Test set size: {len(self.test_dataset)}")
        self._apply_normalization_stats()

    def _apply_normalization_stats(self) -> None:
        """Standardise every partition with one mean and standard deviation.

        The statistics are fitted on the training partition when asked to,
        over every value of every lead, so nothing of the validation or test
        data enters them. They are applied through the dataset's transform,
        ahead of any transform it already had.

        Raises:
            ValueError: If the training values have no spread.
        """
        if self.normalization_stats is None:
            return
        if not isinstance(self.dataset, BaseECGDataset) and not hasattr(self.dataset, "transform"):
            raise ValueError(
                f"normalization_stats is applied through the dataset's transform, which "
                f"{type(self.dataset).__name__} does not have."
            )
        if not self._base_transform_saved:
            self._base_transform = getattr(self.dataset, "transform", None)
            self._base_transform_saved = True
        self.dataset.transform = self._base_transform
        if self._fit_stats:
            total = total_squares = 0.0
            count = 0
            for index in range(len(self.train_dataset)):
                values = torch.as_tensor(self.train_dataset[index][0], dtype=torch.float64)
                total += float(values.sum())
                total_squares += float((values * values).sum())
                count += values.numel()
            mean = total / count
            std = max(total_squares / count - mean * mean, 0.0) ** 0.5
            if std == 0.0:
                raise ValueError(
                    "The training signals are constant, so they cannot be standardised."
                )
            self.normalization_stats = (mean, std)
            if self.verbose:
                logger.info(f"Standardising with the training mean {mean:.6g} and std {std:.6g}")
        mean, std = self.normalization_stats
        self.dataset.transform = GlobalStandardize(float(mean), float(std), self._base_transform)

    def _predefined_partition(self) -> np.ndarray | None:
        """Read the dataset's recommended partition, if it defines one.

        Raises:
            ValueError: If the partition does not name one known partition per sample.
        """
        getter = getattr(self.dataset, "predefined_partition", None)
        partition = getter() if callable(getter) else None
        if partition is None:
            return None
        partition = np.asarray(partition)
        if len(partition) != len(self.dataset) or not np.isin(partition, PARTITIONS).all():
            raise ValueError(
                f"{type(self.dataset).__name__}.predefined_partition() must name one of "
                f"{PARTITIONS} for each of its {len(self.dataset)} samples."
            )
        return partition

    def _collect_labels(self) -> np.ndarray:
        """Collect every sample's label, for stratifying the split.

        A dataset that can hand its labels over directly is asked first;
        otherwise every sample is read for its label.
        """
        size = len(self.dataset)
        direct = getattr(self.dataset, "stratification_labels", None)
        labels: np.ndarray | None = direct() if callable(direct) else None
        if labels is None or len(labels) != size:
            labels = torch.stack([self.dataset[i][1] for i in range(size)]).numpy()
        return labels

    def _loader(self, dataset: Any, shuffle: bool) -> DataLoader:
        """Build a loader over one split.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if dataset is None:
            raise RuntimeError("Call setup() before using the data module")
        options: dict[str, Any] = {}
        if self.num_workers > 0:
            # These two only mean something with worker processes, and PyTorch
            # before 2.0 rejects a prefetch_factor given without any.
            options["persistent_workers"] = self.persistent_workers
            options["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **options,
        )

    def train_dataloader(self) -> DataLoader:
        """Get the training data loader, cropping the samples when train_crop_length is set."""
        if self.train_crop_length and self.train_dataset is not None:
            return self._loader(RandomCrop(self.train_dataset, self.train_crop_length), True)
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get the validation data loader."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Get the test data loader."""
        return self._loader(self.test_dataset, shuffle=False)

    def _loaded_dataset(self) -> Any:
        """The dataset, loading and splitting it first if setup() has not run."""
        if self.dataset is None:
            self.setup()
        return self.dataset

    @property
    def input_channels(self) -> int:
        """Leads in each sample, the input_channels a model for this data takes.

        Reading it loads the dataset if that has not happened yet.
        """
        return int(self._loaded_dataset()[0][0].shape[0])

    @property
    def class_names(self) -> list[str] | None:
        """Names of the classes, in label order, where the dataset gives them.

        Reading it loads the dataset if that has not happened yet.
        """
        names = getattr(self._loaded_dataset(), "class_names", None)
        return list(names) if names else None

    @property
    def num_classes(self) -> int:
        """Number of classes, the output_size a model for this data needs.

        Reading it loads the dataset if that has not happened yet.

        Raises:
            ValueError: If the dataset says neither how many classes it has
                nor what they are called.
        """
        dataset = self._loaded_dataset()
        count = getattr(dataset, "num_classes", None)
        if count is None and self.class_names:
            count = len(self.class_names)
        if count is None:
            raise ValueError(
                f"{type(dataset).__name__} has neither num_classes nor class_names, so the "
                f"number of classes is unknown."
            )
        return int(count)

    def describe(self) -> dict[str, Any]:
        """Describe the data a model is trained on, in the form checkpoints record.

        ECGTrainer stores this as its data_config, which is what lets a
        checkpoint be used on new recordings: it names the preprocessing the
        inputs need and the classes the outputs stand for.

        Returns:
            Mapping with the registered dataset name, its preprocessing
            (sampling rate, segment length, normalisation), the class names,
            the sample shape, the labelling mode, the options that decide labels
            and split, and the split protocol.
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
        get_options = getattr(dataset, "get_dataset_options", None)
        class_names = getattr(dataset, "class_names", None)
        preprocessing = get_preprocessing() if callable(get_preprocessing) else None
        if self.normalization_stats is not None and not self._stats_pending():
            preprocessing = {
                **(preprocessing or {}),
                "normalization": "global",
                "normalization_stats": [float(value) for value in self.normalization_stats],
            }
        return {
            "dataset": getattr(type(dataset), "_registry_name", None),
            "preprocessing": preprocessing,
            "class_names": list(class_names) if class_names else None,
            "input_channels": input_channels,
            "input_length": input_length,
            "multi_label": bool(getattr(dataset, "multi_label", False)),
            "binary_classification": bool(getattr(dataset, "binary_classification", False)),
            "dataset_options": get_options() if callable(get_options) else {},
            "split_protocol": self.split_protocol,
            "split": {
                "seed": self.seed,
                "val_split": self.val_split,
                "test_split": self.test_split,
            },
        }

    def _stats_pending(self) -> bool:
        """Whether statistics were asked to be fitted but setup() has not fitted them yet."""
        return isinstance(self.normalization_stats, str)

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
