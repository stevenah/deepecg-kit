import logging
import shutil
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

from deepecgkit.registry import register_dataset

from ..utils.download import (
    IN_PROGRESS_MARKER,
    download_file,
    flatten_directory,
    marking_download,
)
from .base import BaseECGDataset
from .preprocessing import ECGSegmenter, ECGStandardizer

logger = logging.getLogger(__name__)


@register_dataset(
    name="af-classification",
    input_channels=1,
    description="PhysioNet 2017 AF Classification (4 classes, single-lead)",
)
class AFClassificationDataset(BaseECGDataset):
    """PhysioNet/Computing in Cardiology Challenge 2017 AF Classification Dataset.

    This dataset contains over 10,000 single-lead ECG recordings of 30-60 seconds duration
    for atrial fibrillation (AF) classification. Each recording is labeled as one of four
    categories: Normal (N), Atrial Fibrillation (A), Other rhythm (O), or Noisy (~).

    The recordings are from AliveCor device and represent patient-initiated recordings.

    Reference:
        Clifford GD, Liu C, Moody B, Li-wei HL, Silva I, Li Q, Johnson AE, Mark RG.
        AF classification from a short single lead ECG recording: The PhysioNet/computing
        in cardiology challenge 2017. In 2017 Computing in Cardiology (CinC) 2017 Sep 24 (pp. 1-4). IEEE.

    URL:
        https://physionet.org/content/challenge-2017/1.0.0/
    """

    CLASS_LABELS: ClassVar[list[str]] = [
        "Normal",
        "AF",
        "Other",
        "Noisy",
    ]

    REFERENCE_FILE: ClassVar[str] = "REFERENCE-v3.csv"

    LABEL_MAPPING: ClassVar[dict[str, int]] = {
        "N": 0,
        "A": 1,
        "O": 2,
        "~": 3,
    }

    LEADS: ClassVar[list[str]] = ["ECG"]
    DOWNLOAD_URL: ClassVar[str] = "https://physionet.org/content/challenge-2017/get-zip/1.0.0/"
    #: Mirrors of the project's file directory, tried in order. The AWS
    #: open-data mirror comes first because physionet.org is often throttled.
    DOWNLOAD_MIRRORS: ClassVar[tuple[str, ...]] = (
        "https://physionet-open.s3.amazonaws.com/challenge-2017/1.0.0/",
        "https://physionet.org/files/challenge-2017/1.0.0/",
    )
    #: The files the loader reads. The whole-project archive behind
    #: DOWNLOAD_URL is 1.4 GB and repeats the same recordings several times.
    DOWNLOAD_FILES: ClassVar[tuple[str, ...]] = (
        "REFERENCE-v3.csv",
        "training2017.zip",
        "sample2017.zip",
    )
    DOWNLOAD_SIZE: ClassVar[str | None] = "about 100 MB"
    INNER_ARCHIVES: ClassVar[tuple[str, ...]] = ("training2017.zip", "sample2017.zip")
    #: Archive name used by releases that downloaded the whole project.
    LEGACY_ARCHIVE: ClassVar[str] = "challenge-2017-1.0.0.zip"
    REFERENCE_FALLBACKS: ClassVar[tuple[str, ...]] = (
        "REFERENCE.csv",
        "REFERENCE-v2.csv",
        "REFERENCE-v1.csv",
    )
    NATIVE_SAMPLING_RATE: ClassVar[int] = 300

    def __init__(
        self,
        data_dir: str | Path | None = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        segment_overlap: float = 0.0,
        normalization: str = "zscore",
        strict: bool = False,
        subset: str = "training",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        download: bool = False,
        force_download: bool = False,
        version: str = "1.0.0",
        verbose: bool = True,
    ):
        if subset not in ["training", "validation"]:
            raise ValueError("subset must be either 'training' or 'validation'")

        self.version = version
        self.subset = subset
        self.strict = strict
        self.verbose = verbose
        self.segment_duration_seconds = segment_duration_seconds
        self.segment_overlap = segment_overlap

        self.standardizer = ECGStandardizer(
            target_sampling_rate=sampling_rate,
            normalization=normalization,
        )

        self.segmenter = ECGSegmenter(
            segment_duration_seconds=segment_duration_seconds,
            sampling_rate=sampling_rate,
            overlap=segment_overlap,
        )

        self.signals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_names: list[str] = []
        self.reference_data: pd.DataFrame | None = None

        super().__init__(
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            leads=self.LEADS,
            transform=transform,
            target_transform=target_transform,
            download=download,
            force_download=force_download,
        )

    @classmethod
    def _find_reference(cls, data_dir: Path) -> Path | None:
        """Locate the label file, preferring the corrected v3 annotations."""
        for name in (cls.REFERENCE_FILE, *cls.REFERENCE_FALLBACKS):
            candidate = data_dir / name
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def is_downloaded(cls, data_dir: str | Path | None = None) -> bool:
        """Report whether the labels and the training recordings are present.

        A directory holding only an interrupted download (a partial file, or a
        truncated archive) does not count, so download=True repeats the
        download instead of failing on the leftovers.
        """
        target = Path(data_dir) if data_dir is not None else cls.get_default_data_dir()
        if not target.is_dir() or cls._find_reference(target) is None:
            return False
        if (target / IN_PROGRESS_MARKER).exists():
            return False
        archive = target / "training2017.zip"
        if archive.exists():
            # Archives are deleted once extracted, so one that is still here
            # marks an extraction that never finished; it is usable only if
            # it is a whole archive rather than a truncated one.
            return zipfile.is_zipfile(archive)
        return (target / "training2017").is_dir()

    @classmethod
    def _download(cls, data_dir: Path, verbose: bool = True) -> None:
        """Download the label file and the recording archives from PhysioNet."""
        if verbose:
            logger.info(f"Downloading PhysioNet Challenge 2017 dataset to {data_dir}")
            logger.info(f"Note: this fetches {cls.DOWNLOAD_SIZE}; an interrupted download resumes.")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Earlier releases downloaded the whole project into one archive; an
        # interrupted run left a truncated copy that nothing ever cleaned up.
        legacy_archive = data_dir / cls.LEGACY_ARCHIVE
        if legacy_archive.exists() and not zipfile.is_zipfile(legacy_archive):
            if verbose:
                logger.info(f"Removing incomplete download {legacy_archive.name}...")
            legacy_archive.unlink()

        with marking_download(data_dir):
            for name in cls.DOWNLOAD_FILES:
                already_extracted = (data_dir / Path(name).stem).is_dir()
                if (
                    name in cls.INNER_ARCHIVES
                    and already_extracted
                    and not (data_dir / name).exists()
                ):
                    continue
                download_file(
                    [mirror + name for mirror in cls.DOWNLOAD_MIRRORS],
                    data_dir / name,
                    desc=f"Downloading {name}",
                    max_retries=5,
                )

            cls._flatten_nested_dir(data_dir, verbose)
            cls._extract_inner_zips(data_dir, verbose)
        if verbose:
            logger.info("Download complete!")

    @classmethod
    def _flatten_nested_dir(cls, data_dir: Path, verbose: bool = True) -> None:
        """Flatten a nested extraction directory if present from a previous download."""
        if not data_dir.is_dir():
            return
        for child in data_dir.iterdir():
            if not child.is_dir() or child.name in ("__MACOSX", "training2017", "sample2017"):
                continue
            holds_dataset = (
                any(child.glob("REFERENCE*.csv"))
                or any(child.glob("*2017.zip"))
                or (child / "training2017").is_dir()
            )
            if not holds_dataset:
                continue
            if verbose:
                logger.info(f"Flattening nested directory {child.name}...")
            flatten_directory(child, data_dir)
            return

    @classmethod
    def _extract_inner_zips(cls, data_dir: Path, verbose: bool = True) -> None:
        """Extract training2017.zip and sample2017.zip, then delete them.

        An archive that is still present next to its extracted folder means
        the previous extraction was cut short, so the folder is rebuilt.

        Raises:
            FileNotFoundError: If an archive is truncated or corrupted.
        """
        if not data_dir.is_dir():
            return
        for name in cls.INNER_ARCHIVES:
            zf_path = data_dir / name
            if not zf_path.exists():
                continue
            target_dir = data_dir / zf_path.stem
            if target_dir.exists():
                shutil.rmtree(target_dir)
            if verbose:
                logger.info(f"Extracting {zf_path.name}...")
            try:
                with zipfile.ZipFile(zf_path, "r") as zf:
                    zf.extractall(data_dir)
            except (zipfile.BadZipFile, zlib.error, EOFError) as error:
                raise FileNotFoundError(
                    cls._not_found_message(
                        data_dir, f"a complete {name}, but the one there is truncated or corrupted"
                    )
                ) from error
            zf_path.unlink()

    def _load_data(self) -> None:
        """Load the AF Classification dataset into memory."""
        if self.verbose:
            logger.info(f"Loading AF Classification dataset ({self.subset} subset)...")

        self._flatten_nested_dir(self.data_dir, self.verbose)
        self._extract_inner_zips(self.data_dir, self.verbose)

        if self.subset == "training":
            expected = f"{self.REFERENCE_FILE} and a training2017/ folder"
            data_folder = self.data_dir / "training2017"
            reference_path = self._find_reference(self.data_dir)
        else:
            # sample2017.zip keeps its recordings and their own labels in a
            # validation/ subfolder.
            expected = "a sample2017/ folder with its REFERENCE.csv"
            data_folder = self.data_dir / "sample2017"
            if (data_folder / "validation").is_dir():
                data_folder = data_folder / "validation"
            reference_path = data_folder / "REFERENCE.csv"
            if not reference_path.exists():
                reference_path = self._find_reference(self.data_dir)

        if reference_path is None or not reference_path.exists() or not data_folder.is_dir():
            raise FileNotFoundError(self._not_found_message(self.data_dir, expected))

        self.reference_data = pd.read_csv(
            reference_path, header=None, names=["record_name", "label"]
        )

        iterator = (
            tqdm(
                self.reference_data.iterrows(),
                total=len(self.reference_data),
                desc="Loading records",
            )
            if self.verbose
            else self.reference_data.iterrows()
        )

        attempted_records = 0
        loaded_records = 0
        for _, row in iterator:
            record_name = row["record_name"]
            label = row["label"]

            mat_file = data_folder / f"{record_name}.mat"

            if mat_file.exists():
                attempted_records += 1
                try:
                    mat_data = scipy.io.loadmat(str(mat_file))

                    if "val" in mat_data:
                        signal = mat_data["val"].flatten()
                    elif "ecg" in mat_data:
                        signal = mat_data["ecg"].flatten()
                    elif "data" in mat_data:
                        signal = mat_data["data"].flatten()
                    else:
                        signal_key = [k for k in mat_data.keys() if not k.startswith("__")][0]
                        signal = mat_data[signal_key].flatten()

                    if label not in self.LABEL_MAPPING:
                        if self.verbose:
                            logger.warning(f"Unknown label '{label}' for record {record_name}")
                        continue

                    label_idx = self.LABEL_MAPPING[label]

                    signal = signal[np.newaxis, :]
                    resampled = self.standardizer.resample(signal, self.NATIVE_SAMPLING_RATE)

                    segments, _ = self.segmenter.segment(resampled)
                    if len(segments) == 0:
                        continue

                    for segment in segments:
                        normalized = self.standardizer.normalize(segment)
                        self.signals.append(normalized.astype(np.float32))
                        self.labels.append(label_idx)
                        self.record_names.append(record_name)
                    loaded_records += 1

                except self.RECORD_ERRORS as e:
                    self._record_load_error(record_name, e)
                    continue

        self._finish_load(loaded_records, attempted_records)

        if self.verbose:
            logger.info(
                f"Successfully loaded {len(self.signals)} segments from {self.subset} subset"
            )
