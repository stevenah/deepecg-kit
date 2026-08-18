"""Construction of registered models with shapes resolved from data or checkpoints.

Building a model requires knowing its input channel count and output size. Getting
either wrong produces a state dict that will not load, or worse, one that loads
partially. This module centralises that resolution so every entry point builds
models the same way.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from deepecgkit.models.pretrained import _split_checkpoint
from deepecgkit.registry import get_model, get_model_info, get_model_names

__all__ = [
    "build_model",
    "build_model_from_checkpoint",
    "infer_input_channels",
    "infer_output_size",
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


def infer_output_size(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer the classifier output size from a saved state dict.

    Every registered architecture terminates in a linear classifier, so the last
    two-dimensional weight tensor with a matching bias gives the output size.

    Args:
        state_dict: Model state dict

    Returns:
        The inferred output size, or None if no classifier head was found.
    """
    output_size = None
    for key, tensor in state_dict.items():
        if not key.endswith(".weight") or tensor.dim() != 2:
            continue
        bias = state_dict.get(key[: -len(".weight")] + ".bias")
        if bias is not None and bias.dim() == 1 and bias.shape[0] == tensor.shape[0]:
            output_size = int(tensor.shape[0])
    return output_size


def infer_input_channels(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
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


def build_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    name: str,
    input_channels: Optional[int] = None,
    output_size: Optional[int] = None,
    map_location: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build a model matching a checkpoint and load its weights.

    Shapes are taken from explicit arguments when given, then from a model_config
    block recorded at save time, then inferred from the weights themselves.

    Checkpoints written by ECGTrainer, bare state dicts, and legacy PyTorch
    Lightning checkpoints are all accepted.

    Args:
        checkpoint_path: Path to a checkpoint written by ECGTrainer.save_checkpoint
        name: Registered model name
        input_channels: Overrides the recorded or inferred lead count
        output_size: Overrides the recorded or inferred output size
        map_location: Device to map tensors onto while loading

    Returns:
        Tuple of the loaded model and the checkpoint dictionary.

    Raises:
        ValueError: If the checkpoint holds no weights, or the lead count cannot
            be determined and was not supplied.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    try:
        state_dict, recorded = _split_checkpoint(checkpoint)
    except ValueError as error:
        raise ValueError(f"Checkpoint {checkpoint_path} contains no model_state_dict") from error

    if output_size is None:
        output_size = recorded.get("output_size") or infer_output_size(state_dict)
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
    model.load_state_dict(state_dict)
    return model, checkpoint
