"""Where deepecg-kit keeps the files it downloads.

Datasets and cached weights live under one root so that it can be moved off
a small home quota (an HPC login node, a container) with a single setting.
"""

import os
from pathlib import Path

__all__ = ["DATA_HOME_ENV", "get_data_home"]

#: Environment variable that relocates the download root.
DATA_HOME_ENV = "DEEPECGKIT_HOME"


def get_data_home() -> Path:
    """Return the root directory for downloaded datasets.

    Returns:
        ``$DEEPECGKIT_HOME`` when that variable is set, otherwise
        ``~/.deepecgkit``. Datasets live in its ``datasets/`` subdirectory.
    """
    override = os.environ.get(DATA_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".deepecgkit"
