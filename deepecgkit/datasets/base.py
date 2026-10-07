import logging
import shutil
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch.utils.data import Dataset

from deepecgkit.datasets.preprocessing import convert_to_tensor
from deepecgkit.utils.download import (
    IN_PROGRESS_MARKER,
    download_and_extract_zip,
    download_files,
    download_lock,
    marking_download,
    partial_path,
)
from deepecgkit.utils.paths import DATA_HOME_ENV, get_data_home

logger = logging.getLogger(__name__)


class BaseECGDataset(Dataset, ABC):
    """Base class for all ECG datasets in deepecg-kit.

    Subclasses implement ``_download`` and ``_load_data``. The accessor layer
    (``__getitem__``, ``__len__``, ``num_classes``, ``class_names``,
    ``get_record_info``, ``get_class_distribution``) is provided here for any
    dataset that fills the standard storage attributes during ``_load_data``:

    - ``self.signals``: sequence of per-sample arrays (list or ndarray)
    - ``self.labels``: sequence of integer class labels
    - ``self.record_names``: one source-record name per sample
    - ``CLASS_LABELS``: class-level list of class names
    - ``binary_classification``: optional bool collapsing labels to Non-AF/AF

    Datasets with a different storage model (multi-label PTB-XL, the composed
    unified-af) override the relevant accessors.
    """

    #: Class names for the standard multi-class labelling.
    CLASS_LABELS: ClassVar[list[str]] = []
    #: Class names used when binary_classification is enabled.
    BINARY_CLASS_LABELS: ClassVar[list[str]] = ["Non-AF", "AF"]
    #: On-disk directory name under ~/.deepecgkit/datasets. Defaults to the
    #: lowercased class name; pin it before renaming a class so existing
    #: downloads keep being found.
    DIR_NAME: ClassVar[str | None] = None
    #: Approximate size of what download() fetches, for `deepecg list-datasets`.
    DOWNLOAD_SIZE: ClassVar[str | None] = None

    binary_classification: bool = False
    #: Re-raise per-record load errors instead of skipping the record.
    strict: bool = False

    #: Exception types treated as a corrupt/unreadable source record.
    RECORD_ERRORS: ClassVar[tuple[type[Exception], ...]] = (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        NotImplementedError,
    )

    @classmethod
    def get_default_data_dir(cls) -> Path:
        """Get the default data directory for this dataset.

        Returns:
            ``<data home>/datasets/<name>``, where the data home is
            ``~/.deepecgkit`` unless the DEEPECGKIT_HOME environment variable
            points elsewhere.
        """
        return get_data_home() / "datasets" / (cls.DIR_NAME or cls.__name__.lower())

    @classmethod
    def is_downloaded(cls, data_dir: str | Path | None = None) -> bool:
        """Report whether a usable copy of the dataset is present.

        The default treats any non-empty directory as the dataset. Datasets
        that download archives override this to look for the files their
        loader needs, so an interrupted download is detected and repeated
        rather than mistaken for the dataset.

        Args:
            data_dir: Directory to inspect; the default data directory when None

        Returns:
            True when the dataset can be loaded from data_dir as it stands.
        """
        target = Path(data_dir) if data_dir is not None else cls.get_default_data_dir()
        return target.is_dir() and any(target.iterdir())

    @classmethod
    def _download_records(
        cls,
        data_dir: Path,
        names: list[str] | Callable[[], list[str]],
        *,
        mirror: str,
        desc: str,
        archive_url: str,
        archive_name: str,
        workers: int = 1,
        keep_alive: bool = True,
        verbose: bool = True,
        **archive_kwargs: Any,
    ) -> None:
        """Fetch a dataset's files from a mirror, or its archive when that fails.

        PhysioNet's own servers hand out whole-project archives slowly; the
        open-data mirror serves the same files individually and much faster,
        so it is tried first. The archive stays as the fallback because it
        needs nothing but physionet.org to be reachable.

        Args:
            data_dir: Directory to download into
            names: File paths relative to the mirror's base URL, or a function
                returning them once it has fetched whatever index lists them
            mirror: Base URL of the mirrored project directory
            desc: Description for the progress bars
            archive_url: URL of the whole-project zip archive
            archive_name: File name to save the archive under
            workers: Files fetched at a time from the mirror
            keep_alive: Reuse connections across files; right for many small
                files, wrong for large ones that should resume individually
            verbose: Print progress messages
            **archive_kwargs: Forwarded to download_and_extract_zip
        """
        archive = data_dir / archive_name
        with marking_download(data_dir):
            try:
                resolved = names() if callable(names) else names
                download_files(
                    [mirror],
                    resolved,
                    data_dir,
                    desc=desc,
                    workers=workers,
                    keep_alive=keep_alive,
                )
                # An archive download abandoned on an earlier run would
                # otherwise keep the directory looking unfinished.
                for leftover in (archive, partial_path(archive)):
                    leftover.unlink(missing_ok=True)
                return
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                logger.warning(
                    f"The mirror at {mirror} could not be used ({error}). "
                    f"Falling back to the archive on physionet.org, which is slower."
                )
            download_and_extract_zip(
                archive_url,
                data_dir,
                zip_name=archive_name,
                desc=desc,
                verbose=verbose,
                **archive_kwargs,
            )

    @classmethod
    def _not_found_message(cls, data_dir: Path, expected: str) -> str:
        """Build the error text for a dataset that is absent or incomplete.

        Args:
            data_dir: Directory that was searched
            expected: What the loader looked for there

        Returns:
            A message naming the directory and the ways to fix it.
        """
        return (
            f"{cls.__name__} data was not found in {data_dir} (expected {expected}). "
            f"Pass download=True to fetch it, or force_download=True to replace an "
            f"incomplete copy (CLI: --force-download). To keep datasets elsewhere, pass "
            f"data_dir or set the {DATA_HOME_ENV} environment variable."
        )

    def __init__(
        self,
        data_dir: str | Path | None = None,
        sampling_rate: int = 500,
        leads: list[str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
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

        if force_download:
            with download_lock(self.data_dir):
                self._clear_dataset()
                self.download()
        elif download and not self.is_downloaded(self.data_dir):
            with download_lock(self.data_dir):
                # Another process may have fetched it while this one waited.
                if not self.is_downloaded(self.data_dir):
                    self.download()

        self._load_data()

    def _data_dir_is_empty(self) -> bool:
        """True when the data directory is missing or holds no entries."""
        return not self.data_dir.exists() or not any(self.data_dir.iterdir())

    @classmethod
    def _require_clearable(cls, directory: Path) -> None:
        """Refuse to delete a directory that is not recognisably this dataset.

        A forced download removes the whole directory first. The default
        directory is the package's own, but a data_dir of the caller's
        choosing may hold anything, so it is only cleared when it is empty, a
        complete copy of the dataset, or a download this package began.

        Raises:
            RuntimeError: If the directory holds something else.
        """
        recognised = (
            directory.resolve() == cls.get_default_data_dir().resolve()
            or not any(directory.iterdir())
            or (directory / IN_PROGRESS_MARKER).exists()
            or cls.is_downloaded(directory)
        )
        if not recognised:
            raise RuntimeError(
                f"Refusing to clear {directory} for a forced download: it is not empty and "
                f"does not look like a copy of {cls.__name__}, and everything in it would be "
                f"deleted. Remove it yourself, or choose an empty data directory."
            )

    def _clear_dataset(self) -> None:
        """Clear the dataset directory for re-download."""
        if self.data_dir.exists():
            self._require_clearable(self.data_dir)
            logger.info(f"Clearing existing dataset at {self.data_dir}...")
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _download(cls, data_dir: Path, verbose: bool = True) -> None:
        """Fetch the dataset archive into data_dir. Implemented per dataset."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement _download(), so it cannot be fetched "
            f"automatically. Place the data in {data_dir} and pass download=False."
        )

    @classmethod
    def download_to(cls, data_dir: str | Path | None = None, verbose: bool = True) -> Path:
        """Download the dataset without constructing (and loading) an instance.

        Args:
            data_dir: Target directory; the default data directory when None
            verbose: Print progress messages

        Returns:
            The directory the dataset was downloaded into.
        """
        target = Path(data_dir) if data_dir is not None else cls.get_default_data_dir()
        with download_lock(target):
            cls._download(target, verbose=verbose)
        return target

    def download(self) -> None:
        """Download the dataset into this instance's data directory."""
        self._download(self.data_dir, verbose=getattr(self, "verbose", True))

    @abstractmethod
    def _load_data(self) -> None:
        """Load the dataset into memory."""

    def _record_load_error(self, record_name: str, error: Exception) -> None:
        """Note a source record that failed to load; re-raise in strict mode.

        Args:
            record_name: Name of the failing record
            error: The exception it raised
        """
        if self.strict:
            raise error
        if not hasattr(self, "_load_errors"):
            self._load_errors: list[tuple[str, Exception]] = []
        self._load_errors.append((record_name, error))
        if getattr(self, "verbose", True):
            logger.warning(f"Error loading {record_name}: {error}")

    def _finish_load(self, loaded_records: int, total_records: int) -> None:
        """Surface skipped records loudly once loading completes.

        A partially loaded dataset must never look like a fully loaded one:
        skips produce an unconditional warning (independent of verbose), and
        a load in which every record failed raises.

        Args:
            loaded_records: Number of records that loaded successfully
            total_records: Number of records that were attempted
        """
        errors = getattr(self, "_load_errors", [])
        if errors:
            first_name, first_error = errors[0]
            warnings.warn(
                f"Skipped {len(errors)}/{total_records} records while loading "
                f"{type(self).__name__}; first error ({first_name}): {first_error}. "
                f"Pass strict=True to raise instead.",
                RuntimeWarning,
                stacklevel=3,
            )
        if total_records and loaded_records == 0 and errors:
            raise RuntimeError(
                f"All {total_records} records failed to load for {type(self).__name__}; "
                f"first error ({errors[0][0]}): {errors[0][1]}"
            )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple of (ecg_signal, label) where ecg_signal has shape
            (num_leads, signal_length).
        """
        signal = convert_to_tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform is not None:
            signal = self.transform(signal)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return signal, label

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self.class_names)

    @property
    def class_names(self) -> list[str]:
        """Get the names of the classes in the dataset."""
        if self.binary_classification:
            return self.BINARY_CLASS_LABELS
        return self.CLASS_LABELS

    @property
    def group_ids(self) -> list[str]:
        """Identifiers used to keep correlated samples inside a single split.

        Defaults to record_names, which is correct whenever one record corresponds
        to one subject. Datasets where a single subject contributes several records
        must override this to return subject-level identifiers, otherwise samples
        from the same subject can be divided across train and test.

        Returns:
            One identifier per sample, or an empty list if grouping is unavailable.
        """
        return list(getattr(self, "record_names", []) or [])

    def get_record_info(self, idx: int) -> dict:
        """Describe one sample: source record, label, class name, and shape."""
        label = int(self.labels[idx])
        return {
            "record_name": self.record_names[idx],
            "label": label,
            "class_name": self.class_names[label],
            "signal_shape": self.signals[idx].shape,
        }

    def get_class_distribution(self) -> dict[str, int]:
        """Count samples per class name."""
        labels = getattr(self, "labels", None)
        if labels is None or len(labels) == 0:
            return {}
        unique, counts = np.unique(labels, return_counts=True)
        return {
            self.class_names[int(label)]: int(count)
            for label, count in zip(unique, counts, strict=False)
        }

    def _print_class_distribution(self) -> None:
        """Print class distribution statistics."""
        distribution = self.get_class_distribution()
        if not distribution:
            return
        logger.info("\nClass distribution:")
        for class_name, count in distribution.items():
            logger.info(f"  {class_name}: {count}")

    def get_preprocessing(self) -> dict[str, Any]:
        """Describe how a raw recording becomes one of this dataset's samples.

        Checkpoints record this (see ECGTrainer.data_config) so that a new
        recording can be put through the same steps before it reaches the
        model: resample to sampling_rate, cut into segments of
        segment_duration_seconds, then normalise each segment.

        Returns:
            Mapping with sampling_rate, segment_duration_seconds (None when
            recordings are not segmented), normalization, and leads.
        """
        standardizer = getattr(self, "standardizer", None)
        return {
            "sampling_rate": self.sampling_rate,
            "segment_duration_seconds": getattr(self, "segment_duration_seconds", None),
            "normalization": getattr(standardizer, "normalization", None),
            "leads": list(self.leads) if self.leads else None,
        }

    def get_metadata(self) -> dict:
        """Get metadata about the dataset.

        Returns:
            Dictionary containing metadata such as:
            - sampling_rate: Sampling rate of the signals
            - num_leads: Number of leads
            - lead_names: Names of the leads
            - num_classes: Number of classes
            - class_names: Names of the classes
            - dataset_size: Number of samples
        """
        return {
            "sampling_rate": self.sampling_rate,
            "num_leads": len(self.leads) if self.leads else None,
            "lead_names": self.leads,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "dataset_size": len(self),
        }
