"""Shared model-loading machinery for wrapper classes (internal).

FeatureExtractor and ECGExplainer both wrap a model they can load from
registered pretrained weights, and both need to split caller kwargs between
their own constructor and the model's. That logic lives here once.
"""

import inspect
from pathlib import Path
from typing import Any

from torch import nn

from deepecgkit.models.factory import infer_input_channels
from deepecgkit.models.pretrained import from_pretrained, model_name_for_weights

__all__ = [
    "load_pretrained_resolving_name",
    "model_input_channels",
    "split_wrapper_kwargs",
]


def model_input_channels(model: nn.Module) -> int | None:
    """Lead count declared by the model, or recovered from its weights.

    Args:
        model: The wrapped model

    Returns:
        The lead count, or None when unrecoverable.
    """
    declared = getattr(model, "input_channels", None)
    if isinstance(declared, int):
        return declared
    return infer_input_channels(model.state_dict())


def split_wrapper_kwargs(
    wrapper_cls: type, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split kwargs between a wrapper class's constructor and the model's.

    The wrapper's accepted names are read from its ``__init__`` signature, so
    adding a constructor parameter never requires updating a hand-maintained
    key list.

    Args:
        wrapper_cls: The wrapper class (FeatureExtractor, ECGExplainer, ...)
        kwargs: Mixed keyword arguments from the caller

    Returns:
        Tuple of (wrapper kwargs, model kwargs).
    """
    accepted = set(inspect.signature(wrapper_cls.__init__).parameters) - {
        "self",
        "model",
        "model_name",
    }
    wrapper_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in accepted}
    return wrapper_kwargs, model_kwargs


def load_pretrained_resolving_name(
    weights: str | Path,
    model_name: str | None = None,
    **model_kwargs: Any,
) -> tuple[nn.Module, str]:
    """Load pretrained weights, resolving the model name from the registry.

    Args:
        weights: Name of registered pretrained weights, or a path
        model_name: Registered model name; inferred from the weight registry
            when the weights are registered
        **model_kwargs: Constructor arguments forwarded to the model

    Returns:
        Tuple of the loaded model and the resolved registry name.

    Raises:
        ValueError: If the model name is neither given nor inferable.
    """
    resolved = model_name or model_name_for_weights(str(weights))
    if resolved is None:
        raise ValueError(
            f"Could not determine which model '{weights}' belongs to. Pass model_name explicitly."
        )
    model = from_pretrained(resolved, weights, **model_kwargs)
    return model, resolved
