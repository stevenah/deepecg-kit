"""Single source of truth for device resolution.

Every entry point that accepts a device or accelerator string — the trainer,
the evaluator, feature extraction, explainability, and the CLI commands —
resolves it here, so "auto" and the CLI's accelerator vocabulary mean the same
thing everywhere.
"""

import platform
import sys

import torch

__all__ = ["resolve_device"]


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _on_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _ready(device: torch.device) -> torch.device:
    """Return the device, having made the CPU usable on Apple Silicon.

    PyTorch sends CPU convolutions through its NNPACK backend from a batch
    size of 16 up, and on Apple Silicon that path is many times slower than
    the one it uses below 16 (one 64-channel layer over a batch of 32
    ten-second segments: 368 ms against 5.5 ms). Switching it off only changes
    which implementation runs.
    """
    if device.type == "cpu" and _on_apple_silicon() and hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.set_flags(False)
    return device


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device name, including the CLI's accelerator vocabulary.

    Args:
        device: One of auto, cpu, cuda, gpu, mps, or any string torch accepts.
            "auto" prefers CUDA, then MPS, then CPU. "gpu" is an alias for CUDA.

    Returns:
        The resolved torch.device. Resolving to the CPU on Apple Silicon also
        switches PyTorch's NNPACK backend off for the process; see _ready.

    Raises:
        RuntimeError: If CUDA or MPS is requested explicitly but unavailable.
    """
    if isinstance(device, torch.device):
        return _ready(device)
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return _ready(torch.device("cpu"))
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
    return _ready(resolved)
