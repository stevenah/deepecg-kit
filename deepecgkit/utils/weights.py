from pathlib import Path
from typing import Dict, Optional, Union

import torch

from deepecgkit.utils.download import download_file

WEIGHTS_REGISTRY: Dict[str, Dict] = {
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
    cache_dir = Path.home() / ".cache" / "deepecgkit" / "weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def list_pretrained_weights() -> Dict[str, str]:
    """List all available pretrained weights with descriptions."""
    return {name: info["description"] for name, info in WEIGHTS_REGISTRY.items()}


def get_weight_info(weight_name: str) -> Dict:
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
    """
    info = get_weight_info(weight_name)
    weights_dir = get_weights_dir()
    weight_path = weights_dir / f"{weight_name}.pt"

    download_file(
        url=info["url"],
        file_path=weight_path,
        desc=f"Downloading {weight_name} weights",
        force=force,
    )

    return weight_path


def load_pretrained_weights(
    weight_name: str,
    map_location: Optional[Union[str, torch.device]] = None,
    force_download: bool = False,
) -> Dict:
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
    model_kwargs: Dict,
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
