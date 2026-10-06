import os
from pathlib import Path

import torch

from deepecgkit.utils.download import download_file
from deepecgkit.utils.paths import DATA_HOME_ENV, get_data_home

_PLACEHOLDER_URL_MARKERS = ("your-org",)

WEIGHTS_REGISTRY: dict[str, dict] = {
    "kanres-af-30s": {
        "url": "https://github.com/your-org/deepecg-kit/releases/download/v1.0.0/kanres-af-30s.pt",
        "model_class": "KanResWideX",
        "model_kwargs": {"input_channels": 1, "output_size": 4, "base_channels": 64},
        "description": "KanResWideX trained on AF classification (30s recordings)",
    },
    "afmodel-30s": {
        "url": "https://github.com/your-org/deepecg-kit/releases/download/v1.0.0/afmodel-30s.pt",
        "model_class": "AFModel",
        "model_kwargs": {"recording_length": 30},
        "description": "AFModel trained on PhysioNet 2017 Challenge (30s recordings)",
    },
}


def get_weights_dir() -> Path:
    """Return (and create) the local cache directory for downloaded weights.

    Returns:
        ``$DEEPECGKIT_HOME/weights`` when that variable is set, otherwise
        ``~/.cache/deepecgkit/weights``.
    """
    if os.environ.get(DATA_HOME_ENV):
        cache_dir = get_data_home() / "weights"
    else:
        cache_dir = Path.home() / ".cache" / "deepecgkit" / "weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_published(weight_name: str) -> bool:
    """Report whether registered weights can actually be downloaded.

    Args:
        weight_name: Key into the pretrained weight registry

    Returns:
        False for unknown names and for entries whose URL is still a
        placeholder awaiting a published release.
    """
    info = WEIGHTS_REGISTRY.get(weight_name)
    if info is None:
        return False
    return not any(marker in info["url"] for marker in _PLACEHOLDER_URL_MARKERS)


def list_pretrained_weights(include_unpublished: bool = False) -> dict[str, str]:
    """List the pretrained weights that can be downloaded, with descriptions.

    Args:
        include_unpublished: Also list registry entries that are reserved for
            a future release and have nothing to download yet
    """
    return {
        name: info["description"]
        for name, info in WEIGHTS_REGISTRY.items()
        if include_unpublished or is_published(name)
    }


def get_weight_info(weight_name: str) -> dict:
    """Get information about a specific pretrained weight."""
    if weight_name not in WEIGHTS_REGISTRY:
        available = ", ".join(WEIGHTS_REGISTRY.keys())
        raise ValueError(f"Unknown weight '{weight_name}'. Available: {available}")
    return WEIGHTS_REGISTRY[weight_name]


def download_weights(weight_name: str, force: bool = False) -> Path:
    """Download pretrained weights by name.

    Args:
        weight_name: Name of the pretrained weights (e.g., "kanres-af-30s")
        force: If True, re-download even if weights exist locally

    Returns:
        Path to the downloaded weights file

    Raises:
        RuntimeError: If the registry entry is a placeholder rather than a
            published download URL.
    """
    info = get_weight_info(weight_name)
    url = info["url"]
    if any(marker in url for marker in _PLACEHOLDER_URL_MARKERS):
        raise RuntimeError(
            f"No public weights have been published for '{weight_name}'; its registry entry "
            f"is still a placeholder ({url}). Pass a path to a local weights file or an "
            f"ECGTrainer checkpoint instead, or call register_weights() to point at your own "
            f"hosted release asset."
        )

    weights_dir = get_weights_dir()
    weight_path = weights_dir / f"{weight_name}.pt"

    download_file(
        url=url,
        file_path=weight_path,
        desc=f"Downloading {weight_name} weights",
        force=force,
    )

    return weight_path


def load_pretrained_weights(
    weight_name: str,
    map_location: str | torch.device | None = None,
    force_download: bool = False,
) -> dict:
    """Load pretrained weights by name.

    Args:
        weight_name: Name of the pretrained weights
        map_location: Device to map weights to (e.g., "cpu", "cuda")
        force_download: If True, re-download weights even if cached

    Returns:
        State dict containing model weights
    """
    weight_path = download_weights(weight_name, force=force_download)
    return torch.load(weight_path, map_location=map_location, weights_only=True)


def register_weights(
    name: str,
    url: str,
    model_class: str,
    model_kwargs: dict,
    description: str = "",
) -> None:
    """Register custom pretrained weights.

    Args:
        name: Unique name for the weights
        url: URL to download weights from
        model_class: Name of the model class these weights are for
        model_kwargs: Keyword arguments to instantiate the model
        description: Human-readable description of the weights
    """
    WEIGHTS_REGISTRY[name] = {
        "url": url,
        "model_class": model_class,
        "model_kwargs": model_kwargs,
        "description": description,
    }
