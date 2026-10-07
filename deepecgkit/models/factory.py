"""Construction of registered models with shapes resolved from data or checkpoints.

Building a model requires knowing its input channel count and output size. Getting
either wrong produces a state dict that will not load, or worse, one that loads
partially. This module centralises that resolution so every entry point builds
models the same way.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from deepecgkit.models.pretrained import _split_checkpoint
from deepecgkit.registry import get_model, get_model_info, get_model_names
from deepecgkit.utils.checkpoints import load_checkpoint_file

__all__ = [
    "build_model",
    "build_model_from_checkpoint",
    "infer_input_channels",
    "infer_output_size",
    "recorded_model_name",
]


def build_model(name: str, input_channels: int, output_size: int, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model.

    Args:
        name: Registered model name
        input_channels: Number of ECG leads the model consumes
        output_size: Number of output units of the classifier head
        **kwargs: Additional keyword arguments forwarded to the model, overriding
            any default_kwargs recorded in the registry

    Returns:
        The instantiated model.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in get_model_names():
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(get_model_names())}")
    model_class = get_model(name)
    resolved = dict(get_model_info(name).get("default_kwargs") or {})
    resolved.update(kwargs)
    return model_class(input_channels=input_channels, output_size=output_size, **resolved)


def infer_output_size(
    state_dict: dict[str, torch.Tensor], head_attr: str | None = None
) -> int | None:
    """Infer the classifier output size from a saved state dict.

    Every registered architecture terminates in a linear classifier. When the
    head's attribute name is known (recorded in the registry as HEAD_ATTR),
    the last two-dimensional weight under that prefix is exact; otherwise the
    last two-dimensional weight tensor with a matching bias anywhere in the
    state dict is used as a heuristic, which also covers bare and legacy
    checkpoints.

    Args:
        state_dict: Model state dict
        head_attr: Attribute name of the model's head submodule, if known

    Returns:
        The inferred output size, or None if no classifier head was found.
    """
    output_size = None
    prefix = f"{head_attr}." if head_attr else None
    for key, tensor in state_dict.items():
        if not key.endswith(".weight") or tensor.dim() != 2:
            continue
        if prefix is not None and not key.startswith(prefix):
            continue
        bias = state_dict.get(key[: -len(".weight")] + ".bias")
        if bias is not None and bias.dim() == 1 and bias.shape[0] == tensor.shape[0]:
            output_size = int(tensor.shape[0])
    if output_size is None and prefix is not None:
        return infer_output_size(state_dict)
    return output_size


def infer_input_channels(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Infer the input lead count from a saved state dict, where recoverable.

    Architectures whose first layer is a 1D convolution encode the lead count in
    that layer. Recurrent, attention and 2D architectures do not, so callers must
    supply the value from the dataset instead.

    Args:
        state_dict: Model state dict

    Returns:
        The inferred input channel count, or None if it cannot be recovered.
    """
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and tensor.dim() == 3:
            return int(tensor.shape[1])
    return None


def recorded_model_name(checkpoint: Any) -> str | None:
    """Return the registry name a checkpoint says it was trained as, if any.

    Args:
        checkpoint: The object returned by torch.load

    Returns:
        The recorded model name, or None for bare state dicts and checkpoints
        that predate the record.
    """
    if not isinstance(checkpoint, dict):
        return None
    recorded = checkpoint.get("model_config")
    name = recorded.get("name") if isinstance(recorded, dict) else None
    return name if isinstance(name, str) and name else None


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    name: str | None = None,
    input_channels: int | None = None,
    output_size: int | None = None,
    map_location: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Build a model matching a checkpoint and load its weights.

    The architecture and its shapes are taken from explicit arguments when
    given, then from the model_config block recorded at save time, then (for
    shapes) inferred from the weights themselves.

    Checkpoints written by ECGTrainer, bare state dicts, and legacy PyTorch
    Lightning checkpoints are all accepted.

    Args:
        checkpoint_path: Path to a checkpoint written by ECGTrainer.save_checkpoint
        name: Registered model name; optional when the checkpoint records one
        input_channels: Overrides the recorded or inferred lead count
        output_size: Overrides the recorded or inferred output size
        map_location: Device to map tensors onto while loading

    Returns:
        Tuple of the loaded model and the checkpoint dictionary.

    Raises:
        ValueError: If the checkpoint holds no weights, names a different
            architecture than the one requested, does not say which
            architecture it holds, or the lead count cannot be determined.
    """
    checkpoint = load_checkpoint_file(checkpoint_path, map_location)
    try:
        state_dict, recorded = _split_checkpoint(checkpoint)
    except ValueError as error:
        raise ValueError(f"Checkpoint {checkpoint_path} contains no model_state_dict") from error

    recorded_name = recorded_model_name(checkpoint)
    if name is None:
        if recorded_name is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not record which architecture it holds. "
                f"Name it explicitly (CLI: --model)."
            )
        name = recorded_name
    elif recorded_name is not None and recorded_name != name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was trained as '{recorded_name}', not '{name}'. "
            f"Leave the model name out, or pass '{recorded_name}'."
        )

    if output_size is None:
        head_attr = get_model_info(name).get("head_attr") if name in get_model_names() else None
        output_size = recorded.get("output_size") or infer_output_size(state_dict, head_attr)
    if output_size is None:
        raise ValueError(
            f"Could not determine output size for '{name}' from {checkpoint_path}. "
            "Pass output_size explicitly."
        )

    if input_channels is None:
        input_channels = recorded.get("input_channels") or infer_input_channels(state_dict)
    if input_channels is None:
        raise ValueError(
            f"Input channel count is not recoverable from the weights of '{name}'. "
            "Pass input_channels explicitly, for example from the dataset being evaluated."
        )

    model = build_model(name, input_channels=input_channels, output_size=output_size)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        expected = set(model.state_dict())
        missing = len(expected - set(state_dict))
        unexpected = len(set(state_dict) - expected)
        if not missing and not unexpected:
            raise
        raise ValueError(
            f"The weights in {checkpoint_path} do not fit the '{name}' architecture "
            f"({missing} of its tensors are missing and {unexpected} are not part of it). "
            f"The checkpoint was most likely trained with a different model."
        ) from error
    return model, checkpoint
