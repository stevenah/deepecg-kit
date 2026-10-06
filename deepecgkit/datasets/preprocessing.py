"""Signal standardisation, segmentation, and rhythm-annotation expansion.

These are the building blocks every dataset loader composes: resample to a
target rate, normalise, cut into fixed-length segments, and expand wfdb
rhythm annotations into per-sample label timelines.
"""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import wfdb
from scipy import signal


class ECGStandardizer:
    """Resamples, length-standardises, and normalises raw ECG signals.

    Args:
        target_sampling_rate: Sampling rate signals are resampled to (Hz)
        target_length: Fixed output length in samples; mutually exclusive
            with target_duration_seconds, None leaves lengths unchanged
        target_duration_seconds: Fixed output duration; converted to samples
            at the target rate
        normalization: "zscore", "minmax", or "none"
        clip_method: Where clipping/padding takes place: "center", "start",
            or "end"
    """

    def __init__(
        self,
        target_sampling_rate: int = 300,
        target_length: int | None = None,
        target_duration_seconds: float | None = None,
        normalization: str = "zscore",
        clip_method: str = "center",
    ):
        self.target_sampling_rate = target_sampling_rate
        self.normalization = normalization
        self.clip_method = clip_method

        if target_length is not None and target_duration_seconds is not None:
            raise ValueError("Cannot specify both target_length and target_duration_seconds")

        if target_duration_seconds is not None:
            self.target_length = int(target_duration_seconds * target_sampling_rate)
        else:
            self.target_length = target_length

    def resample(self, ecg_signal: np.ndarray, original_sampling_rate: int) -> np.ndarray:
        """Resample a (leads, samples) or 1-D signal to the target rate."""
        if original_sampling_rate == self.target_sampling_rate:
            return ecg_signal

        num_samples = ecg_signal.shape[-1]
        target_samples = int(num_samples * self.target_sampling_rate / original_sampling_rate)

        if ecg_signal.ndim == 1:
            return signal.resample(ecg_signal, target_samples)
        else:
            resampled = np.zeros((ecg_signal.shape[0], target_samples))
            for i in range(ecg_signal.shape[0]):
                resampled[i] = signal.resample(ecg_signal[i], target_samples)
            return resampled

    def normalize(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Normalise per lead using the configured method."""
        if self.normalization == "zscore":
            mean = np.mean(ecg_signal, axis=-1, keepdims=True)
            std = np.std(ecg_signal, axis=-1, keepdims=True)
            return (ecg_signal - mean) / (std + 1e-8)

        elif self.normalization == "minmax":
            min_val = np.min(ecg_signal, axis=-1, keepdims=True)
            max_val = np.max(ecg_signal, axis=-1, keepdims=True)
            range_val = max_val - min_val
            return np.where(
                range_val > 1e-8, (ecg_signal - min_val) / range_val, ecg_signal - min_val
            )

        elif self.normalization == "none":
            return ecg_signal

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

    def clip_or_pad(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Bring the signal to target_length by clipping or zero-padding."""
        if self.target_length is None or ecg_signal.shape[-1] == self.target_length:
            return ecg_signal

        current_length = ecg_signal.shape[-1]

        if ecg_signal.ndim == 1:
            ecg_signal = ecg_signal[np.newaxis, :]

        if self.clip_method not in ("center", "start", "end"):
            raise ValueError(f"Unknown clip_method: {self.clip_method}")

        if current_length < self.target_length:
            return self._pad_signal(ecg_signal, current_length)
        return self._clip_signal(ecg_signal, current_length)

    def _pad_signal(self, ecg_signal: np.ndarray, current_length: int) -> np.ndarray:
        pad_width = self.target_length - current_length
        if self.clip_method == "center":
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            return np.pad(ecg_signal, ((0, 0), (pad_left, pad_right)), mode="constant")
        if self.clip_method == "start":
            return np.pad(ecg_signal, ((0, 0), (0, pad_width)), mode="constant")
        return np.pad(ecg_signal, ((0, 0), (pad_width, 0)), mode="constant")

    def _clip_signal(self, ecg_signal: np.ndarray, current_length: int) -> np.ndarray:
        excess = current_length - self.target_length
        if self.clip_method == "center":
            start = excess // 2
            return ecg_signal[:, start : start + self.target_length]
        if self.clip_method == "start":
            return ecg_signal[:, : self.target_length]
        return ecg_signal[:, -self.target_length :]

    def __call__(self, ecg_signal: np.ndarray, original_sampling_rate: int) -> np.ndarray:
        """Apply resample, clip_or_pad, and normalize in order."""
        ecg_signal = self.resample(ecg_signal, original_sampling_rate)
        ecg_signal = self.clip_or_pad(ecg_signal)
        ecg_signal = self.normalize(ecg_signal)
        return ecg_signal


class ECGSegmenter:
    """Cuts a continuous recording into fixed-length, optionally overlapping segments.

    Args:
        segment_duration_seconds: Length of each segment in seconds
        sampling_rate: Sampling rate of the incoming signal (Hz)
        overlap: Fraction of a segment shared with its successor, in [0, 1)
    """

    def __init__(
        self,
        segment_duration_seconds: float,
        sampling_rate: int,
        overlap: float = 0.0,
    ):
        self.segment_duration_seconds = segment_duration_seconds
        self.sampling_rate = sampling_rate
        self.overlap = overlap
        self.segment_length = int(segment_duration_seconds * sampling_rate)
        self.stride = int(self.segment_length * (1 - overlap))

    def segment(self, ecg_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cut the signal into segments.

        Args:
            ecg_signal: Signal of shape (leads, samples) or (samples,)

        Returns:
            Tuple of (segments, start indices); both empty when the signal is
            shorter than one segment.
        """
        if ecg_signal.ndim == 1:
            ecg_signal = ecg_signal[np.newaxis, :]

        _, signal_length = ecg_signal.shape

        if signal_length < self.segment_length:
            return np.array([]), np.array([])

        segments = []
        start_indices = []

        for start in range(0, signal_length - self.segment_length + 1, self.stride):
            segment = ecg_signal[:, start : start + self.segment_length]
            segments.append(segment)
            start_indices.append(start)

        return np.array(segments), np.array(start_indices)


class RhythmAnnotationExtractor:
    """Expands wfdb rhythm annotations into per-sample label timelines.

    Args:
        sampling_rate: Rate of the (possibly resampled) signal the labels
            must align with (Hz)
        binary_classification: Collapse all rhythms to AF (1) vs Non-AF (0)
    """

    RHYTHM_MAP: ClassVar[dict[str, int]] = {
        "(AFIB": 1,
        "(AFL": 2,
        "(J": 3,
        "(N": 0,
    }
    #: What the aux_note of a rhythm-change annotation starts with.
    RHYTHM_PREFIX: ClassVar[str] = "("

    def __init__(
        self,
        sampling_rate: int,
        binary_classification: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.binary_classification = binary_classification

    def extract_labels(
        self,
        annotation: Any,
        signal_length: int,
        original_sampling_rate: int | None = None,
    ) -> np.ndarray:
        """Expand a wfdb annotation into one label per signal sample.

        Only rhythm-change annotations, whose aux_note opens a rhythm such as
        ``(AFIB`` or ``(N``, move the label: each one sets the rhythm from its
        own position until the next rhythm change. Some annotation files
        (LTAFDB's among them) also hold an annotation for every beat, with an
        empty aux_note. Those say nothing about the rhythm and are skipped;
        treating them as changes would end every episode at its first beat.

        Args:
            annotation: wfdb annotation with sample and aux_note attributes
            signal_length: Length of the (resampled) signal in samples
            original_sampling_rate: Rate the annotation indices refer to,
                used to rescale them onto the resampled signal

        Returns:
            Integer label per sample, 0 where no rhythm is annotated.
        """
        labels = np.zeros(signal_length, dtype=np.int64)

        if not hasattr(annotation, "aux_note") or not hasattr(annotation, "sample"):
            return labels

        scale = 1.0
        if original_sampling_rate is not None and original_sampling_rate != self.sampling_rate:
            scale = self.sampling_rate / original_sampling_rate

        changes: list[tuple[int, int]] = []
        for sample_idx, aux_note in zip(annotation.sample, annotation.aux_note, strict=False):
            rhythm = aux_note.strip("\x00 \t\r\n")
            if not rhythm.startswith(self.RHYTHM_PREFIX):
                continue

            scaled_idx = int(sample_idx * scale)
            if scaled_idx >= signal_length:
                break

            rhythm_code = self.RHYTHM_MAP.get(rhythm, 0)
            if self.binary_classification:
                rhythm_code = 1 if rhythm_code == 1 else 0
            changes.append((scaled_idx, rhythm_code))

        for position, (start, rhythm_code) in enumerate(changes):
            end = changes[position + 1][0] if position + 1 < len(changes) else signal_length
            labels[start:end] = rhythm_code

        return labels

    def segment_with_labels(
        self, labels: np.ndarray, segment_start_indices: np.ndarray, segment_length: int
    ) -> np.ndarray:
        """Assign each segment the majority label of its samples.

        Args:
            labels: Per-sample label timeline from extract_labels
            segment_start_indices: Segment start positions from ECGSegmenter
            segment_length: Segment length in samples

        Returns:
            One label per segment.
        """
        segment_labels = []

        for start_idx in segment_start_indices:
            segment_label_region = labels[start_idx : start_idx + segment_length]

            unique, counts = np.unique(segment_label_region, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            segment_labels.append(majority_label)

        return np.array(segment_labels)


def convert_to_tensor(
    data: np.ndarray | torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert an array to a tensor of the given dtype, passing tensors through."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    return torch.from_numpy(data).to(dtype)


def read_rhythm_record(
    record_path: str | Path,
    *,
    standardizer: "ECGStandardizer",
    segmenter: "ECGSegmenter",
    rhythm_extractor: "RhythmAnnotationExtractor",
    native_sampling_rate: int,
    use_both_leads: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one rhythm-annotated wfdb record and segment it.

    This is the pipeline shared by the rhythm-annotation datasets (LTAFDB,
    MIT-BIH AFDB): read the record and its .atr annotation, select leads,
    resample to the target rate, expand the rhythm annotation into a
    per-sample label timeline, and cut the signal into fixed-length segments.

    Callers finish the job per storage strategy: optional subsampling, then
    ``rhythm_extractor.segment_with_labels`` and per-segment normalisation.

    Args:
        record_path: Path to the record, without extension
        standardizer: Resampler/normaliser configured for the target rate
        segmenter: Fixed-length segmenter configured for the target rate
        rhythm_extractor: Annotation-to-label expander
        native_sampling_rate: Sampling rate the record was stored at
        use_both_leads: Keep both leads instead of only the first

    Returns:
        Tuple of (segments, segment start indices, per-sample label timeline).
    """
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    signals = record.p_signal.T
    if not use_both_leads:
        signals = signals[0:1, :]

    standardized = standardizer.resample(signals, native_sampling_rate)
    label_timeline = rhythm_extractor.extract_labels(
        annotation, standardized.shape[-1], native_sampling_rate
    )
    segments, start_indices = segmenter.segment(standardized)
    return segments, start_indices, label_timeline
