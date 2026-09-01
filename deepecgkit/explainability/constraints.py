"""Signal-length and lead-count preflight for attribution.

Attribution runs a model many times over the same recording, so a shape error
raised deep inside the network is far more expensive here than during a single
forward pass. Four architectures carry a positional-encoding ceiling and one
carries a convolutional floor, and none of that is recorded in the registry.
This module holds the measured bounds and reshapes a signal to satisfy them
before any gradient work starts.

The same numbers are currently hard-coded in ``examples/feature_extraction.py``
and repeated as a table in the feature-extraction guide. This is the first
importable copy; the other two are left alone.
"""

import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "MAXIMUM_LENGTHS",
    "MINIMUM_LENGTHS",
    "fit_to_constraints",
    "length_bounds",
    "resolve_model_name",
]

MINIMUM_LENGTHS: Dict[str, int] = {
    "afmodel": 7971,
    "convnext-v2": 29,
    "crnn": 16,
    "deep-res-cnn": 127,
    "dualnet": 8,
    "fcn-wang": 2,
    "kanres-deep": 11,
    "mamba": 50,
    "medformer": 50,
    "simple-cnn": 8,
    "xresnet": 17,
}

MAXIMUM_LENGTHS: Dict[str, int] = {
    "dualnet": 5000,
    "mamba": 25049,
    "medformer": 5009,
    "transformer": 5000,
}


def resolve_model_name(model: nn.Module) -> Optional[str]:
    """Recover the registry name of an already-built model.

    ``register_model`` sets ``_registry_name`` on every registered class, so the
    bounds can be looked up for a model handed in directly rather than only for
    one built by name.

    Args:
        model: Any model, registered or not

    Returns:
        The registry name, or None for a model that was never registered.
    """
    name = getattr(model, "_registry_name", None)
    return name if isinstance(name, str) else None


def length_bounds(model: nn.Module, model_name: Optional[str] = None) -> Tuple[int, Optional[int]]:
    """Return the smallest and largest signal length a model accepts.

    A live ``min_input_length`` attribute wins over the table, because AFModel's
    floor is derived from its ``recording_length`` and is 7971 at 30 seconds but
    1393 at 6 or 10. The table cannot know which was built.

    Args:
        model: The model the attribution will run against
        model_name: Registry name, inferred from the model when omitted

    Returns:
        Tuple of the minimum length and the maximum length, where the maximum is
        None for the seventeen architectures that have no ceiling.
    """
    resolved = model_name or resolve_model_name(model)

    declared = getattr(model, "min_input_length", None)
    if isinstance(declared, int):
        minimum = declared
    else:
        minimum = MINIMUM_LENGTHS.get(resolved, 1) if resolved else 1

    maximum = MAXIMUM_LENGTHS.get(resolved) if resolved else None
    return minimum, maximum


def fit_to_constraints(
    signals: torch.Tensor, model: nn.Module, model_name: Optional[str] = None
) -> Tuple[torch.Tensor, int]:
    """Centre-crop a batch so the model can consume it, and report the offset.

    Cropping is centred to match ``ECGStandardizer(clip_method="center")``, so a
    recording trimmed here covers the same span it would have during training.
    The returned offset lets a plot label the time axis against the original
    recording rather than against the crop.

    A signal that is too short cannot be fixed by cropping, so that case raises
    instead of padding: padding would feed the model fabricated samples and the
    resulting attribution would describe the padding as much as the ECG.

    Args:
        signals: Batch of shape (batch, leads, time)
        model: The model the attribution will run against
        model_name: Registry name, inferred from the model when omitted

    Returns:
        Tuple of the possibly cropped batch and the index in the original signal
        where the returned batch begins.

    Raises:
        ValueError: If the batch is not three-dimensional, if its lead count
            does not match the model, or if it is shorter than the model's
            minimum.
    """
    if signals.dim() != 3:
        raise ValueError(
            f"Attribution expects input of shape (batch, leads, time), got a "
            f"{signals.dim()}-D tensor of shape {tuple(signals.shape)}."
        )

    resolved = model_name or resolve_model_name(model)
    label = resolved or type(model).__name__
    minimum, maximum = length_bounds(model, resolved)

    expected_leads = getattr(model, "input_channels", None)
    if isinstance(expected_leads, int) and signals.shape[1] != expected_leads:
        raise ValueError(
            f"{label} was built for {expected_leads} leads but received "
            f"{signals.shape[1]}. Rebuild the model with "
            f"input_channels={signals.shape[1]} or pass matching data."
        )

    length = signals.shape[-1]
    if length < minimum:
        raise ValueError(
            f"{label} needs at least {minimum} samples per lead but received {length}. "
            f"Pass a longer recording, or resample to a rate that reaches that length."
        )

    if maximum is None or length <= maximum:
        return signals, 0

    offset = (length - maximum) // 2
    warnings.warn(
        f"{label} accepts at most {maximum} samples per lead; centre-cropping {length} "
        f"to {maximum} (offset {offset}). Attribution describes only the cropped span.",
        RuntimeWarning,
        stacklevel=3,
    )
    return signals[..., offset : offset + maximum], offset
