import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from deepecgkit.registry import register_dataset

from ..utils.download import download_and_extract_zip
from .base import BaseECGDataset
from .preprocessing import (
    ECGSegmenter,
    ECGStandardizer,
    RhythmAnnotationExtractor,
    read_rhythm_record,
)

logger = logging.getLogger(__name__)


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

    CLASS_LABELS: ClassVar[list[str]] = ["Normal", "AF", "AFL", "J"]
    LEADS: ClassVar[list[str]] = ["ECG1", "ECG2"]
    SAMPLING_RATE: ClassVar[int] = 250
    DOWNLOAD_URL: ClassVar[str] = "https://physionet.org/content/afdb/get-zip/1.0.0/"

    RECORD_NAMES: ClassVar[list[str]] = [
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
        data_dir: str | Path | None = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        segment_overlap: float = 0.0,
        binary_classification: bool = False,
        use_both_leads: bool = False,
        normalization: str = "zscore",
        strict: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        download: bool = False,
        force_download: bool = False,
        verbose: bool = True,
    ):
        self.segment_duration_seconds = segment_duration_seconds
        self.segment_overlap = segment_overlap
        self.binary_classification = binary_classification
        self.use_both_leads = use_both_leads
        self.strict = strict
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

        self.signals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_names: list[str] = []

        super().__init__(
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            leads=self.LEADS if use_both_leads else [self.LEADS[0]],
            transform=transform,
            target_transform=target_transform,
            download=download,
            force_download=force_download,
        )

    @classmethod
    def _download(cls, data_dir: Path, verbose: bool = True) -> None:
        if verbose:
            logger.info(f"Downloading MIT-BIH AFDB to {data_dir}")
            logger.info("Note: This is a ~440MB download.")
        download_and_extract_zip(
            cls.DOWNLOAD_URL,
            data_dir,
            zip_name="afdb-1.0.0.zip",
            desc="Downloading MIT-BIH AFDB",
            verbose=verbose,
        )
        if verbose:
            logger.info("Download complete!")

    def _load_data(self) -> None:
        if self.verbose:
            logger.info("Loading MIT-BIH AFDB data...")

        # PhysioNet zips may nest records inside a files/ subdirectory
        files_dir = self.data_dir / "files"
        if (
            not any(self.data_dir.glob("*.hea"))
            and files_dir.is_dir()
            and any(files_dir.glob("*.hea"))
        ):
            self.data_dir = files_dir

        if not any(self.data_dir.glob("*.hea")):
            raise FileNotFoundError(
                f"No record files found in {self.data_dir}. Expected .hea files from MIT-BIH AFDB."
            )

        attempted_records = 0
        loaded_records = 0
        for record_name in self.RECORD_NAMES:
            record_path = self.data_dir / record_name

            if not (self.data_dir / f"{record_name}.hea").exists():
                if self.verbose:
                    logger.info(f"Skipping {record_name} (not found)")
                continue

            attempted_records += 1
            try:
                segments, start_indices, label_timeline = read_rhythm_record(
                    record_path,
                    standardizer=self.standardizer,
                    segmenter=self.segmenter,
                    rhythm_extractor=self.rhythm_extractor,
                    native_sampling_rate=self.SAMPLING_RATE,
                    use_both_leads=self.use_both_leads,
                )

                if len(segments) == 0:
                    continue

                segment_labels = self.rhythm_extractor.segment_with_labels(
                    label_timeline, start_indices, self.segmenter.segment_length
                )

                for segment, label in zip(segments, segment_labels, strict=False):
                    normalized_segment = self.standardizer.normalize(segment)
                    self.signals.append(normalized_segment)
                    self.labels.append(label)
                    self.record_names.append(record_name)

                loaded_records += 1
                if self.verbose:
                    logger.info(f"Loaded {record_name}: {len(segments)} segments")

            except self.RECORD_ERRORS as e:
                self._record_load_error(record_name, e)

        self._finish_load(loaded_records, attempted_records)

        if self.verbose:
            logger.info(f"Total segments loaded: {len(self.signals)}")
            self._print_class_distribution()
