import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from deepecgkit.registry import register_dataset

from ..utils.download import download_file, has_unfinished_download
from .base import BaseECGDataset
from .preprocessing import (
    ECGSegmenter,
    ECGStandardizer,
    RhythmAnnotationExtractor,
    read_rhythm_record,
)

logger = logging.getLogger(__name__)


@register_dataset(
    name="ltafdb",
    input_channels=1,
    num_classes=4,
    description="Long-Term AF Database (first lead by default, binary or 4-class)",
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

    CLASS_LABELS: ClassVar[list[str]] = ["Normal", "AF", "AFL", "J"]
    LEADS: ClassVar[list[str]] = ["ECG1", "ECG2"]
    SAMPLING_RATE: ClassVar[int] = 128

    DOWNLOAD_URL: ClassVar[str] = "https://physionet.org/content/ltafdb/get-zip/1.0.0/"
    DOWNLOAD_MIRROR: ClassVar[str] = "https://physionet-open.s3.amazonaws.com/ltafdb/1.0.0/"
    DOWNLOAD_SIZE: ClassVar[str | None] = "about 3.7 GB"
    RECORD_SUFFIXES: ClassVar[tuple[str, ...]] = (".hea", ".atr", ".dat")
    #: Part of the cache key; raised whenever cached segments or labels would
    #: differ for the same parameters, so stale caches are never read back.
    #: 2: rhythm labels are carried across the per-beat annotations (version 1
    #: caches label almost every segment Normal).
    CACHE_VERSION: ClassVar[int] = 2
    #: Written into every cache; a cache folder without it predates version 2.
    CACHE_INFO_FILE: ClassVar[str] = "cache_info.json"
    #: The records are tens of megabytes each, so a few are fetched at a time.
    DOWNLOAD_WORKERS: ClassVar[int] = 4

    def __init__(
        self,
        data_dir: str | Path | None = None,
        sampling_rate: int = 300,
        segment_duration_seconds: float = 10.0,
        segment_overlap: float = 0.0,
        binary_classification: bool = False,
        use_both_leads: bool = False,
        normalization: str = "zscore",
        max_segments_per_record: int | None = None,
        subsample_seed: int = 0,
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
        self.max_segments_per_record = max_segments_per_record
        self.subsample_seed = subsample_seed
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

        self.signals: np.ndarray = np.array([])
        self.labels: np.ndarray = np.array([], dtype=np.int64)
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

    def _cache_key(self) -> str:
        params = (
            f"sr{self.sampling_rate}_dur{self.segment_duration_seconds}_"
            f"ovlp{self.segment_overlap}_bin{self.binary_classification}_"
            f"leads{self.use_both_leads}_norm{self.standardizer.normalization}_"
            f"max{self.max_segments_per_record}"
        )
        if self.max_segments_per_record is not None:
            # The seed shapes which segments survive subsampling, so it is part
            # of the cache identity. Unsubsampled caches are seed-independent
            # and share one key.
            params += f"_seed{self.subsample_seed}"
        params += f"_v{self.CACHE_VERSION}"
        return hashlib.md5(params.encode()).hexdigest()[:12]

    def _cache_dir(self) -> Path:
        return self.data_dir / f"cache_{self._cache_key()}"

    @staticmethod
    def _grow_memmap(path: Path, writer: np.memmap, new_rows: int) -> np.memmap:
        """Grow an on-disk .npy memmap to new_rows, preserving its content.

        open_memmap in r+ mode reads the shape from the file header and
        ignores a passed shape, so "reopening larger" silently keeps the old
        capacity — writes past it then raise, and the per-record error
        handling drops those records. A real grow needs a new file.

        Args:
            path: Path of the memmap file being grown
            writer: The currently open memmap over path
            new_rows: New first-dimension size, larger than the current one

        Returns:
            A writable memmap over the grown file.
        """
        old_rows = writer.shape[0]
        tail_shape = writer.shape[1:]
        dtype = writer.dtype
        grow_path = path.with_suffix(".grow.npy")
        grown = np.lib.format.open_memmap(
            str(grow_path), mode="w+", dtype=dtype, shape=(new_rows, *tail_shape)
        )
        grown[:old_rows] = writer[:old_rows]
        grown.flush()
        del writer
        del grown
        grow_path.replace(path)
        return np.lib.format.open_memmap(str(path), mode="r+")

    @classmethod
    def is_downloaded(cls, data_dir: str | Path | None = None) -> bool:
        """Report whether record files, or a finished segment cache, are present.

        A directory still holding the archive or a partial download does not
        count: the archive is deleted once extraction finishes.
        """
        target = Path(data_dir) if data_dir is not None else cls.get_default_data_dir()
        if has_unfinished_download(target):
            return False
        if any(target.glob("cache_*/signals.npy")):
            return True
        return any(
            any(folder.glob("*.hea")) and any(folder.glob("*.atr"))
            for folder in (target, target / "files")
        )

    @classmethod
    def _download(cls, data_dir: Path, verbose: bool = True) -> None:
        if verbose:
            logger.info(f"Downloading Long-Term AF Database to {data_dir}")
            logger.info(f"Note: this fetches {cls.DOWNLOAD_SIZE}; an interrupted download resumes.")
        cls._download_records(
            data_dir,
            lambda: cls._mirrored_files(data_dir),
            mirror=cls.DOWNLOAD_MIRROR,
            desc="Downloading LTAFDB",
            archive_url=cls.DOWNLOAD_URL,
            archive_name="ltafdb-1.0.0.zip",
            workers=cls.DOWNLOAD_WORKERS,
            keep_alive=False,
            verbose=verbose,
        )
        if verbose:
            logger.info("Download complete!")

    @classmethod
    def _mirrored_files(cls, data_dir: Path) -> list[str]:
        """List the record files to fetch, from the project's own index.

        Raises:
            RuntimeError: If the index cannot be fetched.
            ValueError: If the index names no records.
        """
        index = data_dir / "RECORDS"
        download_file(cls.DOWNLOAD_MIRROR + "RECORDS", index, progress=False)
        records = [line.strip() for line in index.read_text().splitlines() if line.strip()]
        if not records or not all(record.isalnum() for record in records):
            raise ValueError(f"unexpected content in the record index {index}")
        return [f"{record}{suffix}" for record in records for suffix in cls.RECORD_SUFFIXES]

    def _resolve_record_dir(self) -> Path:
        """Find the directory containing .hea/.atr record files."""
        if any(self.data_dir.glob("*.hea")):
            return self.data_dir
        files_dir = self.data_dir / "files"
        if files_dir.is_dir() and any(files_dir.glob("*.hea")):
            return files_dir
        return self.data_dir

    def _discover_records(self, record_dir: Path) -> list[str]:
        """Discover available records by finding files with both .hea and .atr extensions."""
        hea_stems = {p.stem for p in record_dir.glob("*.hea")}
        atr_stems = {p.stem for p in record_dir.glob("*.atr")}
        return sorted(hea_stems & atr_stems, key=lambda x: (len(x), x))

    def _warn_about_stale_caches(self, current: Path) -> None:
        """Point out cache folders that are no longer read and only take up space.

        Caches written before CACHE_VERSION 2 carry labels in which almost
        every segment is Normal. Their keys differ from today's, so they are
        never loaded, but each holds several gigabytes.

        Args:
            current: The cache folder this instance uses
        """
        if not self.data_dir.is_dir():
            return
        stale = [
            folder
            for folder in sorted(self.data_dir.glob("cache_*"))
            if folder.is_dir()
            and folder != current
            and (folder / "signals.npy").exists()
            and not (folder / self.CACHE_INFO_FILE).exists()
        ]
        if not stale:
            return
        size = sum(f.stat().st_size for folder in stale for f in folder.iterdir() if f.is_file())
        logger.warning(
            f"{len(stale)} LTAFDB cache folder(s) in {self.data_dir} were written by an earlier "
            f"release that labelled almost every segment Normal. They are no longer read and "
            f"can be deleted to free {size / 1e9:.1f} GB: {', '.join(f.name for f in stale)}"
        )

    def _load_data(self) -> None:
        if self.verbose:
            logger.info("Loading Long-Term AF Database...")

        cache_dir = self._cache_dir()
        signals_path = cache_dir / "signals.npy"
        labels_path = cache_dir / "labels.npy"
        record_names_path = cache_dir / "record_names.npy"
        self._warn_about_stale_caches(cache_dir)

        if signals_path.exists() and labels_path.exists():
            if self.verbose:
                logger.info(f"Loading from cache: {cache_dir.name}")
            self.signals = np.load(signals_path, mmap_mode="r")
            self.labels = np.load(labels_path, mmap_mode="r")
            if record_names_path.exists():
                self.record_names = np.load(record_names_path, allow_pickle=True).tolist()
            else:
                self.record_names = ["unknown"] * len(self.labels)
            if self.verbose:
                logger.info(f"Loaded {len(self.labels)} segments from cache (memory-mapped)")
                self._print_class_distribution()
            return

        record_dir = self._resolve_record_dir()
        record_names = self._discover_records(record_dir)
        if not record_names:
            raise FileNotFoundError(
                self._not_found_message(self.data_dir, ".hea and .atr record files from LTAFDB")
            )

        if self.verbose:
            logger.info(f"Found {len(record_names)} records — processing (first run only)...")

        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_signals = cache_dir / "signals_tmp.npy"

        num_leads = 2 if self.use_both_leads else 1
        seg_len = self.segmenter.segment_length

        writer = None
        write_offset = 0
        label_chunks: list[np.ndarray] = []
        all_record_names: list[str] = []
        total_segments = 0
        loaded_records = 0

        for record_name in record_names:
            record_path = record_dir / record_name

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

                if (
                    self.max_segments_per_record is not None
                    and len(segments) > self.max_segments_per_record
                ):
                    rng = np.random.default_rng(self.subsample_seed)
                    indices = rng.choice(
                        len(segments),
                        self.max_segments_per_record,
                        replace=False,
                    )
                    segments = segments[indices]
                    start_indices = start_indices[indices]

                segment_labels = self.rhythm_extractor.segment_with_labels(
                    label_timeline, start_indices, self.segmenter.segment_length
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
                    writer = self._grow_memmap(tmp_signals, writer, new_size)

                writer[write_offset : write_offset + len(normalized)] = normalized
                label_chunks.append(np.array(segment_labels, dtype=np.int64))
                all_record_names.extend([record_name] * len(segments))
                write_offset += len(normalized)
                total_segments += len(segments)
                loaded_records += 1

                if self.verbose:
                    logger.info(
                        f"Loaded {record_name}: {len(segments)} segments (total: {total_segments})"
                    )

            except self.RECORD_ERRORS as e:
                self._record_load_error(record_name, e)
                continue

        self._finish_load(loaded_records, len(record_names))

        if write_offset == 0:
            shutil.rmtree(cache_dir)
            self.signals = np.array([])
            self.labels = np.array([], dtype=np.int64)
            self.record_names = []
            return

        del writer

        final_signals = np.lib.format.open_memmap(str(tmp_signals), mode="r")
        np.save(signals_path, final_signals[:write_offset])
        del final_signals
        tmp_signals.unlink(missing_ok=True)

        all_labels = np.concatenate(label_chunks)
        np.save(labels_path, all_labels)
        np.save(record_names_path, np.array(all_record_names))
        (cache_dir / self.CACHE_INFO_FILE).write_text(
            json.dumps({"cache_version": self.CACHE_VERSION, "segments": int(write_offset)}) + "\n"
        )

        self.signals = np.load(signals_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.record_names = all_record_names

        if self.verbose:
            logger.info(f"\nCached {len(self.labels)} segments to {cache_dir.name} (memory-mapped)")
            self._print_class_distribution()
