"""Single source of truth for device resolution.

Every entry point that accepts a device or accelerator string — the trainer,
the evaluator, feature extraction, explainability, and the CLI commands —
resolves it here, so "auto" and the CLI's accelerator vocabulary mean the same
thing everywhere.
"""

import torch

__all__ = ["resolve_device"]


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device name, including the CLI's accelerator vocabulary.

    Args:
        device: One of auto, cpu, cuda, gpu, mps, or any string torch accepts.
            "auto" prefers CUDA, then MPS, then CPU. "gpu" is an alias for CUDA.

    Returns:
        The resolved torch.device.

    Raises:
        RuntimeError: If CUDA or MPS is requested explicitly but unavailable.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "gpu":
        device = "cuda"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --accelerator auto to "
            "fall back to the best available device."
        )
    if resolved.type == "mps" and not _mps_available():
        raise RuntimeError(
            "MPS was requested but is not available. Use --accelerator auto to "
            "fall back to the best available device."
        )
    return resolved
