"""Reading checkpoint files without running code they may carry.

A checkpoint is a pickle, and unpickling one from an untrusted source can
execute arbitrary code. Everything ECGTrainer writes consists of tensors and
plain values, which PyTorch can load in its restricted mode, so that mode is
used throughout and a file that needs more is refused unless its source is
declared trusted.
"""

import os
import pickle
from pathlib import Path
from typing import Any

import torch

__all__ = ["TRUST_ENV", "load_checkpoint_file"]

#: Environment variable that, set to "1", allows checkpoints holding arbitrary objects.
TRUST_ENV = "DEEPECGKIT_TRUST_CHECKPOINTS"


def load_checkpoint_file(path: str | Path, map_location: str | torch.device | None = "cpu") -> Any:
    """Load a checkpoint in PyTorch's restricted mode.

    Args:
        path: Checkpoint file
        map_location: Device to map its tensors onto

    Returns:
        Whatever the file holds, normally the checkpoint dictionary.

    Raises:
        RuntimeError: If the file holds objects other than tensors and plain
            values and DEEPECGKIT_TRUST_CHECKPOINTS is not set to "1".
    """
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
