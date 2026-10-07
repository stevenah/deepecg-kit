"""Loading pretrained weights into registered models.

The same "name or path to state dict" resolution was previously duplicated in
every model that supported pretrained weights, and again in the train command.
This module holds the single implementation. It deliberately imports nothing
from deepecgkit.models so that every model module can import it without
creating a cycle.
"""

from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

from deepecgkit.registry import get_model, get_model_names
from deepecgkit.utils.weights import WEIGHTS_REGISTRY, get_weight_info, load_pretrained_weights

__all__ = [
    "PretrainedMixin",
    "from_pretrained",
    "load_matching_weights",
    "model_name_for_weights",
]

T = TypeVar("T", bound="PretrainedMixin")

_LIGHTNING_PREFIX = "model."
_BUILD_KEYS = ("input_channels", "output_size")


def _strip_lightning_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove the wrapper prefix PyTorch Lightning adds to every weight key.

    Args:
        state_dict: Weights as stored in the checkpoint

    Returns:
        The weights with the prefix removed, or unchanged if it is not present
        on every key.
    """
    if state_dict and all(key.startswith(_LIGHTNING_PREFIX) for key in state_dict):
        return {key[len(_LIGHTNING_PREFIX) :]: value for key, value in state_dict.items()}
    return state_dict


def _split_checkpoint(obj: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate model weights from any recorded build configuration.

    Three layouts are recognised: a bare state dict, a checkpoint written by
    ECGTrainer.save_checkpoint keyed on model_state_dict, and a legacy PyTorch
    Lightning checkpoint keyed on state_dict.

    Args:
        obj: The object returned by torch.load

    Returns:
        Tuple of the weights and the constructor keyword arguments recovered
        from a recorded model_config, which is empty for a bare state dict.

    Raises:
        ValueError: If no weights can be found.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a checkpoint dictionary, got {type(obj).__name__}.")

    for key in ("model_state_dict", "state_dict"):
        inner = obj.get(key)
        if isinstance(inner, dict) and inner:
            recorded = obj.get("model_config") or {}
            build_kwargs = {k: recorded[k] for k in _BUILD_KEYS if recorded.get(k) is not None}
            return _strip_lightning_prefix(inner), build_kwargs

    if any(isinstance(value, torch.Tensor) for value in obj.values()):
        return _strip_lightning_prefix(obj), {}

    raise ValueError(
        "Checkpoint contains no weights. Expected a state dict, or a dictionary with a "
        "'model_state_dict' or 'state_dict' entry."
    )


def load_matching_weights(model: nn.Module, loaded: Any) -> dict[str, list[str]]:
    """Load into a model whatever in a set of weights fits it, and say what did not.

    This is how training starts from existing weights. A head of another size
    is expected there, when the new task has a different number of classes,
    and must not keep the backbone from loading; but a file that matches
    nothing at all must not pass for a successful load either.

    Args:
        model: The model to load into
        loaded: The object torch.load returned: a state dict, an ECGTrainer
            checkpoint, or a legacy PyTorch Lightning checkpoint

    Returns:
        Weight names under four keys: "loaded"; "mismatched", present on both
        sides with different shapes and left as initialised; "missing", in the
        model but not in the file; and "unexpected", in the file but not in
        the model.

    Raises:
        ValueError: If the object holds no weights, or none of them fits the
            model.
    """
    state_dict, _ = _split_checkpoint(loaded)
    own = model.state_dict()

    fitting = {
        name: value
        for name, value in state_dict.items()
        if name in own and own[name].shape == value.shape
    }
    if not fitting:
        raise ValueError(
            f"None of the {len(state_dict)} weight tensors in the file fits "
            f"{type(model).__name__}; they belong to a different architecture."
        )
    model.load_state_dict(fitting, strict=False)

    return {
        "loaded": sorted(fitting),
        "mismatched": sorted(name for name in state_dict if name in own and name not in fitting),
        "missing": sorted(name for name in own if name not in state_dict),
        "unexpected": sorted(name for name in state_dict if name not in own),
    }


def model_name_for_weights(weight_name: str) -> str | None:
    """Return the registry name of the model a set of pretrained weights targets.

    The weight registry records the model's class name while the model registry
    is keyed on its short registry name, so the two need bridging.

    Args:
        weight_name: Key into the pretrained weight registry

    Returns:
        The registered model name, or None if the weights are unknown or their
        class is not registered.
    """
    entry = WEIGHTS_REGISTRY.get(weight_name)
    if entry is None:
        return None
    class_name = entry.get("model_class")
    for name in get_model_names():
        if get_model(name).__name__ == class_name:
            return name
    return None


class PretrainedMixin:
    """Adds pretrained weight loading to a registered model.

    Mix into any model whose constructor follows the shared
    ``(input_channels, output_size, **kwargs)`` convention.

    Example:
        >>> model = KanResWideX.from_pretrained("kanres-af-30s")
        >>> model = KanResWideX.from_pretrained("kanres-af-30s", map_location="cuda")
        >>> model = KanResWideX.from_pretrained("/path/to/weights.pt", output_size=2)
        >>> model = AFModel.from_pretrained("/path/to/trained.ckpt")
    """

    @classmethod
    def from_pretrained(
        cls: type[T],
        weights: str | Path,
        map_location: str | torch.device | None = None,
        force_download: bool = False,
        **kwargs: Any,
    ) -> T:
        """Build the model and load pretrained weights into it.

        Args:
            weights: Name of registered pretrained weights, or a path to a
                weights file, an ECGTrainer checkpoint, or a legacy PyTorch
                Lightning checkpoint
            map_location: Device to map the weights onto while loading
            force_download: Re-download registered weights even if cached
            **kwargs: Constructor arguments, overriding anything recorded in the
                checkpoint or the weight registry

        Returns:
            The model with the weights loaded.

        Raises:
            ValueError: If the weights name is unknown, or the file holds no
                recognisable weights.
        """
        weight_path = Path(weights)
        if weight_path.exists():
            loaded = torch.load(weight_path, map_location=map_location, weights_only=True)
            state_dict, recorded = _split_checkpoint(loaded)
            model = cls(**{**recorded, **kwargs})
        else:
            info = get_weight_info(str(weights))
            model = cls(**{**info["model_kwargs"], **kwargs})
            loaded = load_pretrained_weights(str(weights), map_location, force_download)
            state_dict, _ = _split_checkpoint(loaded)

        model.load_state_dict(state_dict)
        return model


def from_pretrained(
    model_name: str,
    weights: str | Path,
    map_location: str | torch.device | None = None,
    force_download: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Build a registered model by name and load pretrained weights into it.

    The callers that hold a model name rather than a class, such as the CLI and
    the feature extractor, use this instead of the classmethod.

    Args:
        model_name: Registered model name
        weights: Name of registered pretrained weights, or a path to a weights
            file or checkpoint
        map_location: Device to map the weights onto while loading
        force_download: Re-download registered weights even if cached
        **kwargs: Constructor arguments forwarded to the model

    Returns:
        The model with the weights loaded.

    Raises:
        KeyError: If the model name is not registered.
        TypeError: If the model does not support pretrained weight loading.
    """
    model_class = get_model(model_name)
    if not hasattr(model_class, "from_pretrained"):
        raise TypeError(f"Model '{model_name}' does not support loading pretrained weights.")
    return model_class.from_pretrained(
        weights, map_location=map_location, force_download=force_download, **kwargs
    )
