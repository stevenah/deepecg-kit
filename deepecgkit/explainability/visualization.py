"""Drawing attribution over the signal it describes.

The plots follow the style of the evaluation visualisations: a ``plot_``
prefix, an optional ``save_path``, and nothing returned. They differ from
``plot_ecg_signals`` in one respect that matters. That function takes
(time, leads); these take (leads, time), matching the model input, the
attribution tensor, and every other array in the package. Following the
neighbouring function instead would put a transpose at every call site, and
each of those is somewhere to get the axes the wrong way round.
"""

import matplotlib.pyplot as plt
import numpy as np

from deepecgkit.explainability.beats import PQRST_WINDOWS_MS
from deepecgkit.utils.plotting import finalize_figure

__all__ = [
    "plot_attribution",
    "plot_beat_profile",
    "plot_lead_importance",
]

_ATTRIBUTION_CMAP = "Reds"
_REGION_COLOURS: dict[str, str] = {"P": "tab:blue", "QRS": "tab:orange", "T": "tab:green"}


def plot_attribution(
    signal: np.ndarray,
    values: np.ndarray,
    sampling_rate: float = 500.0,
    leads: list[str] | None = None,
    time_offset: float = 0.0,
    title: str = "ECG Attribution",
    save_path: str | None = None,
) -> None:
    """Plot each lead with its attribution shaded behind the trace.

    Attribution magnitude is drawn as a background band on one colour scale
    shared by every lead, so a lead the model ignored looks empty next to one it
    relied on. Scaling each lead independently would render those two
    identically, which is the misreading the lead-importance plot exists to
    prevent.

    Args:
        signal: ECG of shape (leads, time)
        values: Attribution of the same shape
        sampling_rate: Samples per second
        leads: Display names, one per lead
        time_offset: Seconds into the original recording where this window
            starts, so a centre-cropped attribution still reads correctly
        title: Figure title
        save_path: Where to write the figure, or None to show it

    Raises:
        ValueError: If the signal is not two-dimensional or the shapes differ.
    """
    trace = np.asarray(signal, dtype=np.float64)
    attribution = np.abs(np.asarray(values, dtype=np.float64))

    if trace.ndim != 2:
        raise ValueError(
            f"plot_attribution expects a signal of shape (leads, time), got shape {trace.shape}."
        )
    if trace.shape != attribution.shape:
        raise ValueError(
            f"Signal of shape {trace.shape} and attribution of shape {attribution.shape} "
            f"must match. Both are (leads, time)."
        )

    n_leads, n_samples = trace.shape
    names = leads or [f"Lead {i + 1}" for i in range(n_leads)]
    time = time_offset + np.arange(n_samples) / sampling_rate
    ceiling = float(attribution.max())

    _fig, axes = plt.subplots(n_leads, 1, figsize=(12, 2 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]

    mesh = None
    for i, ax in enumerate(axes):
        mesh = ax.imshow(
            attribution[i][np.newaxis, :],
            aspect="auto",
            cmap=_ATTRIBUTION_CMAP,
            vmin=0.0,
            vmax=ceiling if ceiling > 0.0 else 1.0,
            extent=(time[0], time[-1], float(trace[i].min()), float(trace[i].max())),
        )
        ax.plot(time, trace[i], color="black", linewidth=0.8)
        ax.set_ylabel(f"{names[i]} (mV)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()
    if mesh is not None:
        plt.colorbar(mesh, ax=axes, label="|attribution|", fraction=0.02, pad=0.01)

    finalize_figure(save_path)


def plot_lead_importance(
    importance: np.ndarray,
    leads: list[str] | None = None,
    title: str = "Lead Importance",
    save_path: str | None = None,
) -> None:
    """Plot the share of attribution each lead carries.

    Args:
        importance: Distribution of shape (leads,), or (batch, leads) to be
            averaged over the batch
        leads: Display names, one per lead
        title: Figure title
        save_path: Where to write the figure, or None to show it

    Raises:
        ValueError: If the distribution has more than two dimensions.
    """
    shares = np.asarray(importance, dtype=np.float64)
    if shares.ndim == 2:
        shares = shares.mean(axis=0)
    if shares.ndim != 1:
        raise ValueError(
            f"plot_lead_importance expects shape (leads,) or (batch, leads), got shape "
            f"{np.asarray(importance).shape}."
        )

    names = leads or [f"Lead {i + 1}" for i in range(shares.size)]
    positions = np.arange(shares.size)

    plt.figure(figsize=(8, max(4, 0.4 * shares.size)))
    plt.barh(positions, shares, color="tab:blue", alpha=0.8)
    plt.yticks(positions, names)
    plt.gca().invert_yaxis()
    plt.xlabel("Share of total attribution")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    finalize_figure(save_path)


def plot_beat_profile(
    profile: np.ndarray,
    times_ms: np.ndarray,
    leads: list[str] | None = None,
    windows_ms: dict[str, tuple[int, int]] | None = None,
    title: str = "Beat-Aligned Attribution",
    save_path: str | None = None,
) -> None:
    """Plot mean attribution across the cardiac cycle with the regions shaded.

    The P, QRS and T windows are drawn behind the curve so the reported shares
    can be checked against the shape they came from, rather than being taken on
    trust from a table.

    Args:
        profile: Mean beat profile of shape (leads, window) or (window,)
        times_ms: Time axis of the window in milliseconds relative to the R-peak
        leads: Display names, one per lead, used for the per-lead traces
        windows_ms: Region boundaries in milliseconds, defaulting to
            PQRST_WINDOWS_MS
        title: Figure title
        save_path: Where to write the figure, or None to show it

    Raises:
        ValueError: If the time axis does not match the profile window.
    """
    values = np.asarray(profile, dtype=np.float64)
    axis = np.asarray(times_ms, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != axis.size:
        raise ValueError(
            f"Profile of shape {np.asarray(profile).shape} does not match a time axis of "
            f"{axis.size} samples."
        )

    names = leads or [f"Lead {i + 1}" for i in range(values.shape[0])]
    regions = windows_ms if windows_ms is not None else PQRST_WINDOWS_MS

    plt.figure(figsize=(10, 5))
    for name, (start_ms, end_ms) in regions.items():
        plt.axvspan(
            start_ms,
            end_ms,
            color=_REGION_COLOURS.get(name, "tab:grey"),
            alpha=0.15,
            label=name,
        )

    for i in range(values.shape[0]):
        plt.plot(axis, np.abs(values[i]), linewidth=0.8, alpha=0.5, label=names[i])
    plt.plot(axis, np.abs(values).mean(axis=0), color="black", linewidth=2.0, label="Mean")

    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Time relative to R-peak (ms)")
    plt.ylabel("|attribution|")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()

    finalize_figure(save_path)
