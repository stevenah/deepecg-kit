"""Reading ECG recordings from files into the layout models consume.

Every entry point that accepts a path rather than a dataset needs the same two
steps: decode the file, then reshape it to (batch, leads, time). Keeping them
here means the prediction and feature-extraction commands cannot drift apart in
how they interpret the same file.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb

__all__ = [
    "SUPPORTED_ECG_SUFFIXES",
    "as_model_input",
    "read_ecg_recording",
]

SUPPORTED_ECG_SUFFIXES: Tuple[str, ...] = (".npy", ".csv", ".dat", ".hea")


def read_ecg_recording(path: Union[str, Path]) -> np.ndarray:
    """Read an ECG recording from a supported file format.

    Args:
        path: Path to a .npy, .csv, or WFDB (.dat/.hea) recording

    Returns:
        The decoded signal array. WFDB records are transposed to put leads
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
        return np.load(str(file_path))
    if file_path.suffix == ".csv":
        return pd.read_csv(str(file_path)).values

    record = wfdb.rdrecord(str(file_path.with_suffix("")))
    return record.p_signal.T


def as_model_input(data: np.ndarray, input_channels: Optional[int] = None) -> np.ndarray:
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
