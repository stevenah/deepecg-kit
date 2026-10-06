"""Reading ECG recordings from files into the layout models consume.

Every entry point that accepts a path rather than a dataset needs the same
steps: decode the file, reshape it to (batch, leads, time), and put it through
the preprocessing the model was trained on. Keeping them here means the
prediction, feature-extraction and explanation commands cannot drift apart in
how they interpret the same file.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from deepecgkit.datasets.preprocessing import ECGSegmenter, ECGStandardizer

__all__ = [
    "SUPPORTED_ECG_SUFFIXES",
    "PreparedRecording",
    "as_model_input",
    "load_recordings",
    "prepare_recording",
    "read_ecg_recording",
    "read_ecg_recording_with_rate",
]

SUPPORTED_ECG_SUFFIXES: tuple[str, ...] = (".npy", ".csv", ".dat", ".hea")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _read_csv_signal(path: Path) -> np.ndarray:
    """Read a CSV of samples, with or without a header row.

    A first row holding anything non-numeric is taken as column names;
    otherwise every row is data, so a headerless export does not lose its
    first sample.
    """
    with open(path) as handle:
        first_row = [token.strip() for token in handle.readline().split(",")]
    has_header = any(token and not _is_number(token) for token in first_row)
    return pd.read_csv(str(path), header=0 if has_header else None).to_numpy()


def read_ecg_recording_with_rate(path: str | Path) -> tuple[np.ndarray, float | None]:
    """Read an ECG recording together with its sampling rate, where the file records one.

    Args:
        path: Path to a .npy, .csv, or WFDB (.dat/.hea) recording

    Returns:
        Tuple of the decoded signal array and its sampling rate in Hz. WFDB
        records carry their rate in the header; .npy and .csv files carry none,
        so the rate is None for them. WFDB records are transposed to put leads
        first; .npy and .csv arrays are returned as stored.

    Raises:
        ValueError: If the file suffix is not one of SUPPORTED_ECG_SUFFIXES.
    """
    file_path = Path(path)
    if file_path.suffix not in SUPPORTED_ECG_SUFFIXES:
        raise ValueError(
            f"Unsupported input format: {file_path.suffix}. "
            f"Supported formats: .npy, .csv, .dat/.hea (WFDB)"
        )

    if file_path.suffix == ".npy":
        return np.load(str(file_path)), None
    if file_path.suffix == ".csv":
        return _read_csv_signal(file_path), None

    record = wfdb.rdrecord(str(file_path.with_suffix("")))
    return record.p_signal.T, float(record.fs)


def read_ecg_recording(path: str | Path) -> np.ndarray:
    """Read an ECG recording from a supported file format.

    Args:
        path: Path to a .npy, .csv, or WFDB (.dat/.hea) recording

    Returns:
        The decoded signal array. WFDB records are transposed to put leads
        first; .npy and .csv arrays are returned as stored.

    Raises:
        ValueError: If the file suffix is not one of SUPPORTED_ECG_SUFFIXES.
    """
    return read_ecg_recording_with_rate(path)[0]


def as_model_input(data: np.ndarray, input_channels: int | None = None) -> np.ndarray:
    """Reshape an ECG array to the (batch, leads, time) layout models expect.

    A one-dimensional array is treated as a single single-lead recording. A
    two-dimensional array is ambiguous, so the lead count decides the
    orientation: whichever axis matches input_channels becomes the lead axis.
    When the lead count is unknown or one, rows are treated as independent
    single-lead recordings.

    Args:
        data: Signal array of one, two, or three dimensions
        input_channels: Number of leads the target model consumes, used to
            disambiguate two-dimensional input

    Returns:
        Array of shape (batch, leads, time).

    Raises:
        ValueError: If a two-dimensional array matches neither orientation, or
            the array has more than three dimensions.
    """
    if data.ndim == 1:
        return data.reshape(1, 1, -1)

    if data.ndim == 2:
        if input_channels is not None and data.shape[0] == input_channels:
            return data.reshape(1, data.shape[0], -1)
        if input_channels is not None and data.shape[1] == input_channels:
            return data.T.reshape(1, input_channels, -1)
        if input_channels in (None, 1):
            return data.reshape(data.shape[0], 1, -1)
        raise ValueError(
            f"Input of shape {data.shape} is not compatible with a model expecting "
            f"{input_channels} leads."
        )

    if data.ndim == 3:
        return data

    raise ValueError(
        f"Input of shape {data.shape} has {data.ndim} dimensions; expected at most three "
        f"(batch, leads, time)."
    )


def load_recordings(
    path: str | Path, input_channels: int | None = None
) -> tuple[np.ndarray, float | None, list[str]]:
    """Read a recording file as a batch of recordings for a model.

    A WFDB record knows which axis holds its leads, so one with more leads
    than the model takes is narrowed to the first ones, the way the datasets
    select leads. Arrays from .npy and .csv files follow as_model_input.

    Args:
        path: Path to a .npy, .csv, or WFDB (.dat/.hea) recording
        input_channels: Number of leads the target model consumes

    Returns:
        Tuple of the recordings shaped (batch, leads, time), the file's
        sampling rate in Hz (None when the format carries none), and notes
        describing any lead selection, for reporting to the user.

    Raises:
        ValueError: If the format is unsupported or the array cannot be
            arranged for a model with input_channels leads.
    """
    data, rate = read_ecg_recording_with_rate(path)
    data = np.asarray(data)
    notes: list[str] = []
    is_wfdb = Path(path).suffix in (".dat", ".hea")
    if is_wfdb and input_channels and data.ndim == 2 and data.shape[0] > input_channels:
        verb = "is" if input_channels == 1 else "are"
        notes.append(
            f"the record has {data.shape[0]} leads, of which the first {input_channels} {verb} used"
        )
        data = data[:input_channels]
    return as_model_input(data, input_channels), rate, notes


@dataclass
class PreparedRecording:
    """One recording after the preprocessing a model was trained with.

    Attributes:
        segments: Array of shape (num_segments, leads, samples), float32
        start_seconds: Start of each segment within the recording, in seconds
        sampling_rate: Rate of the segments in Hz, or None when unknown
        notes: What was applied or assumed, for reporting to the user
    """

    segments: np.ndarray
    start_seconds: np.ndarray
    sampling_rate: float | None = None
    notes: list[str] = field(default_factory=list)


def prepare_recording(
    recording: np.ndarray,
    preprocessing: Mapping[str, Any] | None,
    source_sampling_rate: float | None = None,
) -> PreparedRecording:
    """Put one raw recording through a model's training-time preprocessing.

    The steps mirror what the dataset loaders do: resample to the training
    rate, cut into training-length segments, and normalise each segment.

    Args:
        recording: Signal of shape (leads, samples)
        preprocessing: Mapping as returned by BaseECGDataset.get_preprocessing
            and recorded in checkpoints, with sampling_rate,
            segment_duration_seconds and normalization. None leaves the
            recording untouched and returns it as a single segment.
        source_sampling_rate: Rate the recording was sampled at, in Hz. None
            assumes it is already at the training rate.

    Returns:
        The prepared segments, with notes describing what was done.
    """
    signal = np.asarray(recording, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    notes: list[str] = []

    if not preprocessing:
        return PreparedRecording(
            segments=signal[np.newaxis].astype(np.float32),
            start_seconds=np.zeros(1),
            sampling_rate=source_sampling_rate,
            notes=notes,
        )

    target_rate = preprocessing.get("sampling_rate")
    normalization = preprocessing.get("normalization") or "none"
    duration = preprocessing.get("segment_duration_seconds")
    standardizer = ECGStandardizer(
        target_sampling_rate=target_rate or 0, normalization=normalization
    )

    if target_rate and source_sampling_rate and source_sampling_rate != target_rate:
        signal = standardizer.resample(signal, source_sampling_rate)
        notes.append(f"resampled from {source_sampling_rate:g} Hz to {target_rate:g} Hz")
    elif target_rate and not source_sampling_rate:
        notes.append(f"assumed to be sampled at {target_rate:g} Hz, the rate the model saw")

    starts = np.zeros(1)
    segments = signal[np.newaxis]
    if duration and target_rate:
        segmenter = ECGSegmenter(segment_duration_seconds=duration, sampling_rate=target_rate)
        cut, start_indices = segmenter.segment(signal)
        if len(cut):
            segments = cut
            starts = np.asarray(start_indices, dtype=np.float64) / float(target_rate)
            if len(cut) > 1:
                notes.append(f"cut into {len(cut)} segments of {duration:g} s")
        else:
            seconds = signal.shape[-1] / float(target_rate)
            notes.append(
                f"only {seconds:.1f} s long, shorter than the {duration:g} s segments the "
                f"model was trained on, so it is used whole"
            )

    if normalization != "none":
        segments = np.stack([standardizer.normalize(segment) for segment in segments])
        notes.append(f"{normalization} normalised")

    return PreparedRecording(
        segments=segments.astype(np.float32),
        start_seconds=starts,
        sampling_rate=float(target_rate) if target_rate else source_sampling_rate,
        notes=notes,
    )
