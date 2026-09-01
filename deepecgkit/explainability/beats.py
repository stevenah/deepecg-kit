"""Aligning attribution to the cardiac cycle.

A per-timestep attribution map over a ten-second recording is hard to argue
with, because nothing in it is anchored to anything a cardiologist names. This
module anchors it: it finds the R-peaks, cuts one window per beat, averages
those windows, and reports how much of the resulting profile falls in the P,
QRS and T regions.

That last number is the point. "This model attends to the P-wave region" stops
being a description of a picture and becomes a fraction that can be wrong.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

__all__ = [
    "BEAT_WINDOW_MS",
    "PQRST_WINDOWS_MS",
    "beat_aligned_attribution",
    "detect_r_peaks",
    "pqrst_enrichment",
    "pqrst_fractions",
    "region_duration_fractions",
]

PQRST_WINDOWS_MS: Dict[str, Tuple[int, int]] = {
    "P": (-250, -50),
    "QRS": (-50, 50),
    "T": (100, 400),
}

BEAT_WINDOW_MS: Tuple[int, int] = (-300, 500)

_QRS_BANDPASS_HZ: Tuple[float, float] = (5.0, 15.0)
_MINIMUM_RR_SECONDS = 0.3
_THRESHOLD_FACTOR = 1.0
_INTEGRATION_SECONDS = 0.15


def detect_r_peaks(
    signal: np.ndarray,
    sampling_rate: float = 500.0,
    bandpass: Tuple[float, float] = _QRS_BANDPASS_HZ,
    min_rr_seconds: float = _MINIMUM_RR_SECONDS,
    threshold_factor: float = _THRESHOLD_FACTOR,
    integration_seconds: float = _INTEGRATION_SECONDS,
) -> np.ndarray:
    """Locate R-peaks in a single-lead signal.

    The pipeline is the classical Pan-Tompkins shape: band-pass to the QRS
    band, differentiate, square, integrate over a moving window, then take
    peaks above a threshold with a refractory distance between them. On
    synthetic ECG with known R positions this recovers every peak to the exact
    sample.

    The band is a parameter rather than a constant because it encodes an
    assumption about where QRS energy lives. Real ECG puts it near 5 to 15 Hz,
    but synthetic test signals often do not, and a detector that silently
    returns the wrong beats is worse than one that is asked to be told.

    Args:
        signal: One-dimensional signal of shape (time,)
        sampling_rate: Samples per second
        bandpass: Low and high cutoff in Hz for the QRS band
        min_rr_seconds: Refractory period, the closest two peaks may be
        threshold_factor: Peaks must exceed mean plus this many standard
            deviations of the integrated energy
        integration_seconds: Width of the moving-average window

    Returns:
        Sorted sample indices of the detected R-peaks, empty when the signal is
        too short or carries no energy.

    Raises:
        ValueError: If the signal is not one-dimensional, or the band is not a
            rising pair below the Nyquist frequency.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"detect_r_peaks expects a one-dimensional signal, got shape {values.shape}."
        )

    nyquist = 0.5 * sampling_rate
    low, high = bandpass
    if not 0.0 < low < high:
        raise ValueError(f"bandpass must be a rising pair of positive cutoffs, got {bandpass}.")
    if low >= nyquist:
        raise ValueError(
            f"bandpass low cutoff {low} Hz is at or above the Nyquist frequency {nyquist} Hz."
        )

    window = max(1, int(integration_seconds * sampling_rate))
    if values.size <= 3 * window:
        return np.empty(0, dtype=np.int64)

    numerator, denominator = scipy_signal.butter(
        2, [low / nyquist, min(high / nyquist, 0.99)], btype="band"
    )
    filtered = scipy_signal.filtfilt(numerator, denominator, values)
    energy = np.gradient(filtered) ** 2
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")

    spread = float(integrated.std())
    if spread <= 0.0:
        return np.empty(0, dtype=np.int64)

    peaks, _ = scipy_signal.find_peaks(
        integrated,
        height=float(integrated.mean()) + threshold_factor * spread,
        distance=max(1, int(min_rr_seconds * sampling_rate)),
    )
    return peaks.astype(np.int64)


def beat_aligned_attribution(
    values: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float = 500.0,
    window_ms: Tuple[int, int] = BEAT_WINDOW_MS,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Average attribution over R-peak-aligned windows.

    Beats whose window would run past either end of the recording are dropped
    rather than padded or clipped, and the surviving count is returned so a
    caller can report it. A truncated beat averaged in alongside whole ones
    would pull the profile toward the edges of the recording, which is an
    artefact of where the recording happened to start.

    Args:
        values: Attribution of shape (leads, time)
        r_peaks: Sample indices of the R-peaks
        sampling_rate: Samples per second
        window_ms: Start and end of the beat window in milliseconds relative to
            the R-peak, negative before it

    Returns:
        Tuple of the mean profile of shape (leads, window), the window's time
        axis in milliseconds, and the number of beats averaged. When no beat
        fits, the profile is all zeros and the count is zero.

    Raises:
        ValueError: If the attribution is not two-dimensional, or the window
            does not start before it ends.
    """
    attribution = np.asarray(values, dtype=np.float64)
    if attribution.ndim != 2:
        raise ValueError(
            f"beat_aligned_attribution expects attribution of shape (leads, time), got "
            f"shape {attribution.shape}."
        )

    start_ms, end_ms = window_ms
    if start_ms >= end_ms:
        raise ValueError(f"window_ms must start before it ends, got {window_ms}.")

    rate = float(sampling_rate)
    pre = round(abs(start_ms) / 1000.0 * rate)
    post = round(end_ms / 1000.0 * rate)
    span = pre + post
    times_ms = (np.arange(span) - pre) / rate * 1000.0

    leads, length = attribution.shape
    windows = [
        attribution[:, int(peak) - pre : int(peak) + post]
        for peak in r_peaks
        if int(peak) - pre >= 0 and int(peak) + post <= length
    ]

    if not windows:
        return np.zeros((leads, span), dtype=np.float32), times_ms, 0

    profile = np.mean(np.stack(windows, axis=0), axis=0)
    return profile.astype(np.float32), times_ms, len(windows)


def pqrst_fractions(
    profile: np.ndarray,
    times_ms: np.ndarray,
    windows_ms: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Split a beat profile into its P, QRS and T shares.

    Shares are taken against the total mass of the profile, not against the
    mass inside the three windows, so they do not sum to one. The remainder is
    the baseline between the named regions, and hiding it by renormalising
    would make a model that attends mostly to the isoelectric line look like one
    that attends to the P wave.

    Args:
        profile: Mean beat profile of shape (leads, window) or (window,)
        times_ms: Time axis of the window in milliseconds relative to the R-peak
        windows_ms: Region boundaries in milliseconds, defaulting to
            PQRST_WINDOWS_MS

    Returns:
        Fraction of total profile mass falling in each named region, plus a
        "baseline" entry holding the remainder. All zeros when the profile
        carries no mass.

    Raises:
        ValueError: If the time axis does not match the profile's last axis.
    """
    values = np.abs(np.asarray(profile, dtype=np.float64))
    if values.ndim == 2:
        values = values.mean(axis=0)

    axis = np.asarray(times_ms, dtype=np.float64)
    if axis.shape != values.shape:
        raise ValueError(
            f"times_ms of shape {axis.shape} does not match the profile window of shape "
            f"{values.shape}."
        )

    regions = windows_ms if windows_ms is not None else PQRST_WINDOWS_MS
    total = float(values.sum())
    if total <= 0.0:
        fractions = {name: 0.0 for name in regions}
        fractions["baseline"] = 0.0
        return fractions

    fractions = {}
    for name, (start_ms, end_ms) in regions.items():
        selected = (axis >= start_ms) & (axis < end_ms)
        fractions[name] = float(values[selected].sum() / total)

    fractions["baseline"] = float(max(0.0, 1.0 - sum(fractions.values())))
    return fractions


def region_duration_fractions(
    times_ms: np.ndarray,
    windows_ms: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Return each region's share of the beat window's duration.

    This is the profile a model spreading its attention evenly over the cardiac
    cycle would produce, and it is what makes a raw share interpretable. The
    P window is 200 ms of an 800 ms beat, so 25 per cent of the attribution
    falling in it is not evidence of anything.

    Args:
        times_ms: Time axis of the beat window in milliseconds relative to the
            R-peak
        windows_ms: Region boundaries in milliseconds, defaulting to
            PQRST_WINDOWS_MS

    Returns:
        Fraction of the window's samples falling in each named region, plus a
        "baseline" entry holding the remainder.
    """
    axis = np.asarray(times_ms, dtype=np.float64)
    regions = windows_ms if windows_ms is not None else PQRST_WINDOWS_MS
    total = float(axis.size)

    shares = {}
    for name, (start_ms, end_ms) in regions.items():
        selected = (axis >= start_ms) & (axis < end_ms)
        shares[name] = float(selected.sum() / total) if total else 0.0

    shares["baseline"] = float(max(0.0, 1.0 - sum(shares.values())))
    return shares


def pqrst_enrichment(
    fractions: Dict[str, float],
    times_ms: np.ndarray,
    windows_ms: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Compare each region's attribution share against its share of the beat.

    A value of one means the region received exactly the attention its width
    would earn by chance, above one means the model concentrates there, and
    below one means it avoids the region. This is the number the P-wave claim
    turns on: a P share of 0.09 sounds small until it is read against the
    P window's 0.25 share of the beat, at which point it says the model attends
    to the P-wave region at roughly a third of chance.

    Args:
        fractions: Output of pqrst_fractions
        times_ms: Time axis of the beat window in milliseconds
        windows_ms: Region boundaries in milliseconds, defaulting to
            PQRST_WINDOWS_MS

    Returns:
        Ratio of attribution share to duration share for each region. A region
        occupying no samples yields 0.0 rather than an infinity.
    """
    shares = region_duration_fractions(times_ms, windows_ms)
    return {
        name: float(fractions.get(name, 0.0) / share) if share > 0.0 else 0.0
        for name, share in shares.items()
    }
