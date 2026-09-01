"""Signal-length and lead-count preflight for attribution.

Attribution runs a model many times over the same recording, so a shape error
raised deep inside the network is far more expensive here than during a single
forward pass. Four architectures carry a positional-encoding ceiling and one
carries a convolutional floor. The measured bounds are recorded per model in
the registry (``register_model(min_input_length=..., max_input_length=...)``)
so that a new architecture declares them next to its definition; this module
reads them back and reshapes a signal to satisfy them before any gradient
work starts.
"""

import warnings
from typing import Any

import torch
from torch import nn

from deepecgkit.registry import get_model_info, get_model_names

__all__ = [
    "MAXIMUM_LENGTHS",  # noqa: F822  (module __getattr__ view over the registry)
    "MINIMUM_LENGTHS",  # noqa: F822
    "fit_to_constraints",
    "length_bounds",
    "resolve_model_name",
]


def __getattr__(name: str) -> Any:
    """Expose the registry bounds under the historical table names."""
    if name == "MINIMUM_LENGTHS":
        return {
            model: info["min_input_length"]
            for model in get_model_names()
            if (info := get_model_info(model)).get("min_input_length") is not None
        }
    if name == "MAXIMUM_LENGTHS":
        return {
            model: info["max_input_length"]
            for model in get_model_names()
            if (info := get_model_info(model)).get("max_input_length") is not None
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_model_name(model: nn.Module) -> str | None:
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


def length_bounds(model: nn.Module, model_name: str | None = None) -> tuple[int, int | None]:
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
    info = get_model_info(resolved) if resolved and resolved in get_model_names() else {}

    declared = getattr(model, "min_input_length", None)
    if isinstance(declared, int):
        minimum = declared
    else:
        minimum = info.get("min_input_length") or 1

    return minimum, info.get("max_input_length")


def fit_to_constraints(
    signals: torch.Tensor, model: nn.Module, model_name: str | None = None
) -> tuple[torch.Tensor, int]:
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
