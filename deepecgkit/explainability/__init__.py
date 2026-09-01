"""
Attribution for trained ECG models.

Provides:
- Gradient saliency, integrated gradients, and occlusion over the raw signal,
  each returning attribution in the input's own (batch, leads, time) shape
- Lead attribution, reducing that map to a normalised distribution over leads
- Beat-aligned saliency, averaging attribution over R-peak-aligned windows and
  reporting the share falling in the P, QRS and T regions
- Signal-length and lead-count preflight, so an architecture's ceiling produces
  a centre-crop rather than a shape error
- Plots of the signal with attribution overlaid, the lead distribution, and the
  beat profile

The plotting helpers are imported on first use rather than eagerly, so that
importing this package does not pull matplotlib into the path of every command.
"""

from typing import Any

from deepecgkit.explainability.attribution import (
    METHODS,
    Attribution,
    ECGExplainer,
    explain,
    integrated_gradients,
    lead_importance,
    occlusion,
    saliency,
)
from deepecgkit.explainability.beats import (
    BEAT_WINDOW_MS,
    PQRST_WINDOWS_MS,
    beat_aligned_attribution,
    detect_r_peaks,
    pqrst_enrichment,
    pqrst_fractions,
    region_duration_fractions,
)
from deepecgkit.explainability.constraints import (
    MAXIMUM_LENGTHS,
    MINIMUM_LENGTHS,
    fit_to_constraints,
    length_bounds,
)

_LAZY_PLOTS = frozenset({"plot_attribution", "plot_beat_profile", "plot_lead_importance"})

__all__ = [
    "BEAT_WINDOW_MS",
    "MAXIMUM_LENGTHS",
    "METHODS",
    "MINIMUM_LENGTHS",
    "PQRST_WINDOWS_MS",
    "Attribution",
    "ECGExplainer",
    "beat_aligned_attribution",
    "detect_r_peaks",
    "explain",
    "fit_to_constraints",
    "integrated_gradients",
    "lead_importance",
    "length_bounds",
    "occlusion",
    "plot_attribution",
    "plot_beat_profile",
    "plot_lead_importance",
    "pqrst_enrichment",
    "pqrst_fractions",
    "region_duration_fractions",
    "saliency",
]


def __getattr__(name: str) -> Any:
    """Import a plotting helper the first time it is asked for.

    Args:
        name: Attribute being looked up on the package

    Returns:
        The requested plotting function.

    Raises:
        AttributeError: If the name is not one this package provides.
    """
    if name in _LAZY_PLOTS:
        from deepecgkit.explainability import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
