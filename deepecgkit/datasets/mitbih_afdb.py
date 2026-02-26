import os
import shutil
import zipfile
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wfdb

from deepecgkit.registry import register_dataset

from ..utils.download import download_file
from .base import BaseECGDataset
from .preprocessing import (
    ECGSegmenter,
    ECGStandardizer,
    RhythmAnnotationExtractor,
    convert_to_tensor,
)


@register_dataset(
    name="mitbih-afdb",
    input_channels=2,
    num_classes=4,
    description="MIT-BIH AF Database (2-lead, binary or 4-class)",
)
class MITBIHAFDBDataset(BaseECGDataset):
    """MIT-BIH Atrial Fibrillation Database (AFDB) Dataset.

    Contains 25 long-term (10-hour) two-lead ECG recordings from subjects with
    atrial fibrillation (mostly paroxysmal). Rhythm annotations indicate Normal,
    AF, Atrial Flutter, and Junctional rhythm segments.

    Reference:
        Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R
        intervals. Computers in Cardiology. 1983;10:227-230.

    URL:
        https://physionet.org/content/afdb/1.0.0/
    """

    CLASS_LABELS: ClassVar[List[str]] = ["Normal", "AF", "AFL", "J"]
    LABEL_MAPPING: ClassVar[Dict[str, int]] = {
        "(N": 0,
        "(AFIB": 1,
        "(AFL": 2,
        "(J": 3,
    }
    LEADS: ClassVar[List[str]] = ["ECG1", "ECG2"]
    SAMPLING_RATE: ClassVar[int] = 250
    DOWNLOAD_URL: ClassVar[str] = "https://physionet.org/content/afdb/get-zip/1.0.0/"

    RECORD_NAMES: ClassVar[List[str]] = [
        "04015",
        "04043",
        "04048",
        "04126",
        "04746",
        "04908",
        "04936",
        "05091",
        "05121",
        "05261",
        "06426",
        "06453",
        "06995",
        "07162",
        "07859",
        "07879",
        "07910",
        "08215",
        "08219",
        "08378",
        "08405",
        "08434",
        "08455",
    ]

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        segment_overlap: float = 0.0,
        binary_classification: bool = False,
        use_both_leads: bool = False,
        normalization: str = "zscore",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        force_download: bool = False,
        verbose: bool = True,
    ):
        self.segment_duration_seconds = segment_duration_seconds
        self.segment_overlap = segment_overlap
        self.binary_classification = binary_classification
        self.use_both_leads = use_both_leads
        self.verbose = verbose

        self.standardizer = ECGStandardizer(
            target_sampling_rate=sampling_rate,
            normalization=normalization,
        )

        self.segmenter = ECGSegmenter(
            segment_duration_seconds=segment_duration_seconds,
            sampling_rate=sampling_rate,
            overlap=segment_overlap,
        )

        self.rhythm_extractor = RhythmAnnotationExtractor(
            sampling_rate=sampling_rate, binary_classification=binary_classification
        )

        self.signals: List[np.ndarray] = []
        self.labels: List[int] = []
        self.record_names: List[str] = []

        super().__init__(
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            leads=self.LEADS if use_both_leads else [self.LEADS[0]],
            transform=transform,
            target_transform=target_transform,
            download=download,
            force_download=force_download,
        )

    def download(self):
        if self.verbose:
            print(f"Downloading MIT-BIH AFDB to {self.data_dir}")
            print("Note: This is a ~440MB download.")

        os.makedirs(self.data_dir, exist_ok=True)

        zip_path = self.data_dir / "afdb-1.0.0.zip"
        download_file(
            self.DOWNLOAD_URL,
            zip_path,
            desc="Downloading MIT-BIH AFDB",
            max_retries=5,
        )

        if self.verbose:
            print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

        nested_dir = self.data_dir / "afdb-1.0.0"
        if nested_dir.is_dir():
            for item in nested_dir.iterdir():
                dest = self.data_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            nested_dir.rmdir()

        zip_path.unlink(missing_ok=True)

        if self.verbose:
            print("Download complete!")

    def _load_data(self):
        if self.verbose:
            print("Loading MIT-BIH AFDB data...")

        if not any(self.data_dir.glob("*.hea")):
            raise FileNotFoundError(
                f"No record files found in {self.data_dir}. Expected .hea files from MIT-BIH AFDB."
            )

        for record_name in self.RECORD_NAMES:
            record_path = self.data_dir / record_name

            if not (self.data_dir / f"{record_name}.hea").exists():
                if self.verbose:
                    print(f"Skipping {record_name} (not found)")
                continue

            try:
                record = wfdb.rdrecord(str(record_path))
                annotation = wfdb.rdann(str(record_path), "atr")

                signals = record.p_signal.T

                if not self.use_both_leads:
                    signals = signals[0:1, :]

                standardized_signal = self.standardizer.resample(signals, self.SAMPLING_RATE)

                labels = self.rhythm_extractor.extract_labels(
                    annotation, standardized_signal.shape[-1], self.SAMPLING_RATE
                )

                segments, start_indices = self.segmenter.segment(standardized_signal)

                if len(segments) == 0:
                    continue

                segment_labels = self.rhythm_extractor.segment_with_labels(
                    labels, start_indices, self.segmenter.segment_length
                )

                for segment, label in zip(segments, segment_labels):
                    normalized_segment = self.standardizer.normalize(segment)
                    self.signals.append(normalized_segment)
                    self.labels.append(label)
                    self.record_names.append(record_name)

                if self.verbose:
                    print(f"Loaded {record_name}: {len(segments)} segments")

            except Exception as e:
                if self.verbose:
                    print(f"Error loading {record_name}: {e}")

        if self.verbose:
            print(f"Total segments loaded: {len(self.signals)}")
            self._print_class_distribution()

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
        return 2 if self.binary_classification else len(self.CLASS_LABELS)

    @property
    def class_names(self) -> List[str]:
        if self.binary_classification:
            return ["Non-AF", "AF"]
        return self.CLASS_LABELS

    def get_record_info(self, idx: int) -> Dict:
        return {
            "record_name": self.record_names[idx],
            "label": self.labels[idx],
            "class_name": self.class_names[self.labels[idx]],
            "signal_shape": self.signals[idx].shape,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.class_names[int(label)]: int(count) for label, count in zip(unique, counts)}
