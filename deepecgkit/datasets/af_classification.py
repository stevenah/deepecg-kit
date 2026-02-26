import os
import shutil
import zipfile
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

from deepecgkit.registry import register_dataset

from ..utils.download import download_file
from .base import BaseECGDataset
from .preprocessing import ECGSegmenter, ECGStandardizer, convert_to_tensor


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

    CLASS_LABELS: ClassVar[List[str]] = [
        "Normal",
        "AF",
        "Other",
        "Noisy",
    ]

    REFERENCE_FILE: ClassVar[str] = "REFERENCE-v3.csv"

    LABEL_MAPPING: ClassVar[Dict[str, int]] = {
        "N": 0,
        "A": 1,
        "O": 2,
        "~": 3,
    }

    LEADS: ClassVar[List[str]] = ["ECG"]
    DOWNLOAD_URL: ClassVar[str] = "https://physionet.org/content/challenge-2017/get-zip/1.0.0/"
    NATIVE_SAMPLING_RATE: ClassVar[int] = 300

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        segment_overlap: float = 0.0,
        normalization: str = "zscore",
        subset: str = "training",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        force_download: bool = False,
        version: str = "1.0.0",
        verbose: bool = True,
    ):
        if subset not in ["training", "validation"]:
            raise ValueError("subset must be either 'training' or 'validation'")

        self.version = version
        self.subset = subset
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

        self.signals: List[np.ndarray] = []
        self.labels: List[int] = []
        self.record_names: List[str] = []
        self.reference_data: Optional[pd.DataFrame] = None

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
        """Download the AF Classification dataset from PhysioNet as a single ZIP."""
        if self.verbose:
            print(f"Downloading PhysioNet Challenge 2017 dataset to {self.data_dir}")
            print("Note: This is a large dataset (~1.4GB). Download may take a while.")

        os.makedirs(self.data_dir, exist_ok=True)

        zip_path = self.data_dir / "challenge-2017-1.0.0.zip"
        download_file(
            self.DOWNLOAD_URL,
            zip_path,
            desc="Downloading PhysioNet Challenge 2017",
            max_retries=5,
        )

        if self.verbose:
            print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

        self._flatten_nested_dir()
        self._extract_inner_zips()

        zip_path.unlink(missing_ok=True)

        if self.verbose:
            print("Download complete!")

    def _flatten_nested_dir(self):
        """Flatten a nested extraction directory if present from a previous download."""
        for child in self.data_dir.iterdir():
            if not child.is_dir() or child.name in ("__MACOSX", "training2017", "sample2017"):
                continue
            if self.verbose:
                print(f"Flattening nested directory {child.name}...")
            for item in child.iterdir():
                dest = self.data_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            child.rmdir()
            return

    def _extract_inner_zips(self):
        """Extract inner zip files like training2017.zip."""
        for zf_path in self.data_dir.glob("*.zip"):
            folder_name = zf_path.stem
            target_dir = self.data_dir / folder_name
            if target_dir.exists():
                continue
            if self.verbose:
                print(f"Extracting {zf_path.name}...")
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(self.data_dir)
            zf_path.unlink()

    def _load_data(self):
        """Load the AF Classification dataset into memory."""
        if self.verbose:
            print(f"Loading AF Classification dataset ({self.subset} subset)...")

        self._flatten_nested_dir()
        self._extract_inner_zips()

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

        for _, row in iterator:
            record_name = row["record_name"]
            label = row["label"]

            mat_file = data_folder / f"{record_name}.mat"

            if mat_file.exists():
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
                            print(f"Warning: Unknown label '{label}' for record {record_name}")
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

                except Exception as e:
                    if self.verbose:
                        print(f"Error loading record {record_name}: {e}")
                    continue

        if self.verbose:
            print(f"Successfully loaded {len(self.signals)} segments from {self.subset} subset")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]

        signal = convert_to_tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            signal = self.transform(signal)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return signal, label

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self.CLASS_LABELS)

    @property
    def class_names(self) -> List[str]:
        """Get the names of the classes in the dataset."""
        return self.CLASS_LABELS

    def get_record_info(self, idx: int) -> Dict:
        record_name = self.record_names[idx]
        label_idx = self.labels[idx]

        return {
            "record_name": record_name,
            "label": label_idx,
            "class_name": self.CLASS_LABELS[label_idx],
            "signal_shape": self.signals[idx].shape,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.class_names[int(label)]: int(count) for label, count in zip(unique, counts)}
