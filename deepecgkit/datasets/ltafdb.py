import hashlib
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
    name="ltafdb",
    input_channels=2,
    num_classes=4,
    description="Long-Term AF Database (2-lead, binary or 4-class)",
)
class LTAFDBDataset(BaseECGDataset):
    """Long-Term AF Database (LTAFDB) Dataset.

    Contains 84 long-term (typically 24-hour) two-lead ECG recordings of subjects
    with paroxysmal or sustained atrial fibrillation. Rhythm annotations indicate
    Normal, AF, Atrial Flutter, and Junctional rhythm segments.

    Reference:
        Petrutiu S, Sahakian AV, Swiryn S. Abrupt changes in fibrillatory wave
        characteristics at the termination of paroxysmal atrial fibrillation in humans.
        Europace. 2007;9(7):466-470.

    URL:
        https://physionet.org/content/ltafdb/1.0.0/
    """

    CLASS_LABELS: ClassVar[List[str]] = ["Normal", "AF", "AFL", "J"]
    LABEL_MAPPING: ClassVar[Dict[str, int]] = {
        "(N": 0,
        "(AFIB": 1,
        "(AFL": 2,
        "(J": 3,
    }
    LEADS: ClassVar[List[str]] = ["ECG1", "ECG2"]
    SAMPLING_RATE: ClassVar[int] = 128

    DOWNLOAD_URL: ClassVar[str] = "https://physionet.org/content/ltafdb/get-zip/1.0.0/"

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        segment_overlap: float = 0.0,
        binary_classification: bool = False,
        use_both_leads: bool = False,
        normalization: str = "zscore",
        max_segments_per_record: Optional[int] = None,
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
        self.max_segments_per_record = max_segments_per_record
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

        self.signals: np.ndarray = np.array([])
        self.labels: np.ndarray = np.array([], dtype=np.int64)
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

    def _cache_key(self) -> str:
        params = (
            f"sr{self.sampling_rate}_dur{self.segment_duration_seconds}_"
            f"ovlp{self.segment_overlap}_bin{self.binary_classification}_"
            f"leads{self.use_both_leads}_norm{self.standardizer.normalization}_"
            f"max{self.max_segments_per_record}"
        )
        return hashlib.md5(params.encode()).hexdigest()[:12]

    def _cache_dir(self) -> Path:
        return self.data_dir / f"cache_{self._cache_key()}"

    def download(self):
        if self.verbose:
            print(f"Downloading Long-Term AF Database to {self.data_dir}")
            print("Note: This is a large dataset (~1.7GB). Download may take a while.")

        os.makedirs(self.data_dir, exist_ok=True)

        zip_path = self.data_dir / "ltafdb-1.0.0.zip"
        download_file(
            self.DOWNLOAD_URL,
            zip_path,
            desc="Downloading LTAFDB",
            max_retries=5,
        )

        if self.verbose:
            print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

        nested_dir = self.data_dir / "ltafdb-1.0.0"
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

    def _resolve_record_dir(self) -> Path:
        """Find the directory containing .hea/.atr record files."""
        if any(self.data_dir.glob("*.hea")):
            return self.data_dir
        files_dir = self.data_dir / "files"
        if files_dir.is_dir() and any(files_dir.glob("*.hea")):
            return files_dir
        return self.data_dir

    def _discover_records(self, record_dir: Path) -> List[str]:
        """Discover available records by finding files with both .hea and .atr extensions."""
        hea_stems = {p.stem for p in record_dir.glob("*.hea")}
        atr_stems = {p.stem for p in record_dir.glob("*.atr")}
        return sorted(hea_stems & atr_stems, key=lambda x: (len(x), x))

    def _load_data(self):
        if self.verbose:
            print("Loading Long-Term AF Database...")

        cache_dir = self._cache_dir()
        signals_path = cache_dir / "signals.npy"
        labels_path = cache_dir / "labels.npy"
        record_names_path = cache_dir / "record_names.npy"

        if signals_path.exists() and labels_path.exists():
            if self.verbose:
                print(f"Loading from cache: {cache_dir.name}")
            self.signals = np.load(signals_path, mmap_mode="r")
            self.labels = np.load(labels_path, mmap_mode="r")
            if record_names_path.exists():
                self.record_names = np.load(record_names_path, allow_pickle=True).tolist()
            else:
                self.record_names = ["unknown"] * len(self.labels)
            if self.verbose:
                print(f"Loaded {len(self.labels)} segments from cache (memory-mapped)")
                self._print_class_distribution()
            return

        record_dir = self._resolve_record_dir()
        record_names = self._discover_records(record_dir)
        if not record_names:
            raise FileNotFoundError(
                f"No valid records found in {self.data_dir}. "
                "Expected .hea and .atr files from LTAFDB."
            )

        if self.verbose:
            print(f"Found {len(record_names)} records — processing (first run only)...")

        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_signals = cache_dir / "signals_tmp.npy"

        num_leads = 2 if self.use_both_leads else 1
        seg_len = self.segmenter.segment_length

        writer = None
        write_offset = 0
        label_chunks: List[np.ndarray] = []
        all_record_names: List[str] = []
        total_segments = 0

        for record_name in record_names:
            record_path = record_dir / record_name

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

                if (
                    self.max_segments_per_record is not None
                    and len(segments) > self.max_segments_per_record
                ):
                    indices = np.random.choice(
                        len(segments),
                        self.max_segments_per_record,
                        replace=False,
                    )
                    segments = segments[indices]
                    start_indices = start_indices[indices]

                segment_labels = self.rhythm_extractor.segment_with_labels(
                    labels, start_indices, self.segmenter.segment_length
                )

                normalized = np.stack([self.standardizer.normalize(seg) for seg in segments])

                if writer is None:
                    estimated_total = len(record_names) * len(segments)
                    writer = np.lib.format.open_memmap(
                        str(tmp_signals),
                        mode="w+",
                        dtype=np.float32,
                        shape=(estimated_total, num_leads, seg_len),
                    )

                needed = write_offset + len(normalized)
                if needed > writer.shape[0]:
                    new_size = max(needed, writer.shape[0] * 2)
                    writer = np.lib.format.open_memmap(
                        str(tmp_signals),
                        mode="r+",
                        dtype=np.float32,
                        shape=(new_size, num_leads, seg_len),
                    )

                writer[write_offset : write_offset + len(normalized)] = normalized
                label_chunks.append(np.array(segment_labels, dtype=np.int64))
                all_record_names.extend([record_name] * len(segments))
                write_offset += len(normalized)
                total_segments += len(segments)

                if self.verbose:
                    print(
                        f"Loaded {record_name}: {len(segments)} segments (total: {total_segments})"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"Error loading {record_name}: {e}")
                continue

        if write_offset == 0:
            shutil.rmtree(cache_dir)
            self.signals = np.array([])
            self.labels = np.array([], dtype=np.int64)
            self.record_names = []
            return

        del writer

        final_signals = np.lib.format.open_memmap(
            str(tmp_signals), mode="r+", shape=(write_offset, num_leads, seg_len)
        )
        np.save(signals_path, final_signals[:write_offset])
        del final_signals
        tmp_signals.unlink(missing_ok=True)

        all_labels = np.concatenate(label_chunks)
        np.save(labels_path, all_labels)
        np.save(record_names_path, np.array(all_record_names))

        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.record_names = all_record_names

        if self.verbose:
            print(f"\nCached {len(self.labels)} segments to {cache_dir.name} (memory-mapped)")
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
            "label": int(self.labels[idx]),
            "class_name": self.class_names[int(self.labels[idx])],
            "signal_shape": self.signals[idx].shape,
        }

    def get_class_distribution(self) -> Dict[str, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return {self.class_names[int(label)]: int(count) for label, count in zip(unique, counts)}
