"""Finding and reading checkpoint files without running code they may carry.

A checkpoint is a pickle, and unpickling one from an untrusted source can
execute arbitrary code. Everything ECGTrainer writes consists of tensors and
plain values, which PyTorch can load in its restricted mode, so that mode is
used throughout and a file that needs more is refused unless its source is
declared trusted.

Wherever a checkpoint is expected, a run directory may be given instead and
stands for the best checkpoint inside it.
"""

import os
import pickle
import re
from pathlib import Path
from typing import Any

import torch

__all__ = ["TRUST_ENV", "load_checkpoint_file", "resolve_checkpoint"]

#: Environment variable that, set to "1", allows checkpoints holding arbitrary objects.
TRUST_ENV = "DEEPECGKIT_TRUST_CHECKPOINTS"

CHECKPOINT_DIRNAME = "checkpoints"
_VAL_LOSS_PATTERN = re.compile(r"val_loss=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def resolve_checkpoint(path: str | Path) -> Path:
    """Turn a checkpoint argument into a checkpoint file.

    A file is returned as it is. A directory stands for the best checkpoint
    inside it: a run directory is searched in its ``checkpoints/`` folder,
    and the checkpoint with the lowest validation loss in its file name wins
    (ties and unnamed files fall back to the most recently written).

    Args:
        path: A checkpoint file, a run directory, or a checkpoints directory

    Returns:
        Path of the checkpoint file to load.

    Raises:
        FileNotFoundError: If the path is a directory holding no checkpoints.
    """
    target = Path(path)
    if not target.is_dir():
        return target

    folder = target / CHECKPOINT_DIRNAME if (target / CHECKPOINT_DIRNAME).is_dir() else target
    candidates = sorted(
        candidate for pattern in ("*.pt", "*.ckpt") for candidate in folder.glob(pattern)
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files (*.pt, *.ckpt) found in {folder}")

    def rank(candidate: Path) -> tuple[float, float]:
        match = _VAL_LOSS_PATTERN.search(candidate.name)
        loss = float(match.group(1)) if match else float("inf")
        return loss, -candidate.stat().st_mtime

    return min(candidates, key=rank)


def load_checkpoint_file(path: str | Path, map_location: str | torch.device | None = "cpu") -> Any:
    """Load a checkpoint in PyTorch's restricted mode.

    Args:
        path: Checkpoint file, or a run directory standing for its best checkpoint
        map_location: Device to map its tensors onto

    Returns:
        Whatever the file holds, normally the checkpoint dictionary.

    Raises:
        RuntimeError: If the file holds objects other than tensors and plain
            values and DEEPECGKIT_TRUST_CHECKPOINTS is not set to "1".
        FileNotFoundError: If path is a directory holding no checkpoints.
    """
    path = resolve_checkpoint(path)
    if os.environ.get(TRUST_ENV) == "1":
        return torch.load(path, map_location=map_location, weights_only=False)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as error:
        raise RuntimeError(
            f"{path} holds objects other than tensors and plain values, so loading it could "
            f"run code from the file. If you trust where it came from, set {TRUST_ENV}=1 and "
            f"try again."
        ) from error
