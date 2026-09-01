import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

from deepecgkit.registry import register_dataset

from ..utils.download import download_and_extract_zip, flatten_directory
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
    def _download(cls, data_dir: Path, verbose: bool = True) -> None:
        """Download the AF Classification dataset from PhysioNet as a single ZIP."""
        if verbose:
            logger.info(f"Downloading PhysioNet Challenge 2017 dataset to {data_dir}")
            logger.info("Note: This is a large dataset (~1.4GB). Download may take a while.")
        download_and_extract_zip(
            cls.DOWNLOAD_URL,
            data_dir,
            zip_name="challenge-2017-1.0.0.zip",
            desc="Downloading PhysioNet Challenge 2017",
            nested_dirs=(),
            flatten_files_dir=False,
            verbose=verbose,
        )
        cls._flatten_nested_dir(data_dir, verbose)
        cls._extract_inner_zips(data_dir, verbose)
        if verbose:
            logger.info("Download complete!")

    @classmethod
    def _flatten_nested_dir(cls, data_dir: Path, verbose: bool = True) -> None:
        """Flatten a nested extraction directory if present from a previous download."""
        for child in data_dir.iterdir():
            if not child.is_dir() or child.name in ("__MACOSX", "training2017", "sample2017"):
                continue
            if verbose:
                logger.info(f"Flattening nested directory {child.name}...")
            flatten_directory(child, data_dir)
            return

    @classmethod
    def _extract_inner_zips(cls, data_dir: Path, verbose: bool = True) -> None:
        """Extract inner zip files like training2017.zip."""
        for zf_path in data_dir.glob("*.zip"):
            folder_name = zf_path.stem
            target_dir = data_dir / folder_name
            if target_dir.exists():
                continue
            if verbose:
                logger.info(f"Extracting {zf_path.name}...")
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(data_dir)
            zf_path.unlink()

    def _load_data(self) -> None:
        """Load the AF Classification dataset into memory."""
        if self.verbose:
            logger.info(f"Loading AF Classification dataset ({self.subset} subset)...")

        self._flatten_nested_dir(self.data_dir, self.verbose)
        self._extract_inner_zips(self.data_dir, self.verbose)

        reference_path = self.data_dir / self.REFERENCE_FILE
        if not reference_path.exists():
            for ref_file in ["REFERENCE.csv", "REFERENCE-v2.csv", "REFERENCE-v1.csv"]:
                alt_path = self.data_dir / ref_file
                if alt_path.exists():
                    reference_path = alt_path
                    break

        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found in {self.data_dir}")

        self.reference_data = pd.read_csv(
            reference_path, header=None, names=["record_name", "label"]
        )

        if self.subset == "training":
            data_folder = self.data_dir / "training2017"
        else:
            data_folder = self.data_dir / "sample2017"

        if not data_folder.exists():
            raise FileNotFoundError(
                f"Data folder {data_folder} not found. Please download the dataset first."
            )

        iterator = (
            tqdm(self.reference_data.iterrows(), desc="Loading records")
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
                        self.signals.append(normalized)
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
