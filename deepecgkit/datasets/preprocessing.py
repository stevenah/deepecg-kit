from typing import ClassVar, Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy import signal


class ECGStandardizer:
    def __init__(
        self,
        target_sampling_rate: int = 300,
        target_length: Optional[int] = None,
        target_duration_seconds: Optional[float] = None,
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
        ecg_signal = self.resample(ecg_signal, original_sampling_rate)
        ecg_signal = self.clip_or_pad(ecg_signal)
        ecg_signal = self.normalize(ecg_signal)
        return ecg_signal


class ECGSegmenter:
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

    def segment(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if ecg_signal.ndim == 1:
            ecg_signal = ecg_signal[np.newaxis, :]

        num_leads, signal_length = ecg_signal.shape

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
    RHYTHM_MAP: ClassVar[Dict[str, int]] = {
        "(AFIB": 1,
        "(AFL": 2,
        "(J": 3,
        "(N": 0,
    }

    def __init__(
        self,
        sampling_rate: int,
        binary_classification: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.binary_classification = binary_classification

    def extract_labels(
        self,
        annotation,
        signal_length: int,
        original_sampling_rate: Optional[int] = None,
    ) -> np.ndarray:
        labels = np.zeros(signal_length, dtype=np.int64)

        if not hasattr(annotation, "aux_note") or not hasattr(annotation, "sample"):
            return labels

        scale = 1.0
        if original_sampling_rate is not None and original_sampling_rate != self.sampling_rate:
            scale = self.sampling_rate / original_sampling_rate

        for i, (sample_idx, aux_note) in enumerate(zip(annotation.sample, annotation.aux_note)):
            scaled_idx = int(sample_idx * scale)
            if scaled_idx >= signal_length:
                break

            rhythm_code = self.RHYTHM_MAP.get(aux_note.strip(), 0)

            if self.binary_classification:
                rhythm_code = 1 if rhythm_code == 1 else 0

            raw_next = annotation.sample[i + 1] if i + 1 < len(annotation.sample) else signal_length
            next_sample = min(int(raw_next * scale), signal_length)

            labels[scaled_idx:next_sample] = rhythm_code

        return labels

    def segment_with_labels(
        self, labels: np.ndarray, segment_start_indices: np.ndarray, segment_length: int
    ) -> np.ndarray:
        segment_labels = []

        for start_idx in segment_start_indices:
            segment_label_region = labels[start_idx : start_idx + segment_length]

            unique, counts = np.unique(segment_label_region, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            segment_labels.append(majority_label)

        return np.array(segment_labels)


def convert_to_tensor(
    data: Union[np.ndarray, torch.Tensor], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    return torch.from_numpy(data).to(dtype)
