"""Shared matplotlib finishing helper.

Every plotting function in the package ends the same way: save at 300 dpi
with tight bounding box and close, or show interactively. The block was
copied thirteen times across the evaluation and explainability visualization
modules; it lives here once.
"""

from pathlib import Path

import matplotlib.pyplot as plt

__all__ = ["finalize_figure"]


def finalize_figure(save_path: str | None = None) -> None:
    """Save the current figure and close it, or show it interactively.

    Args:
        save_path: Where to save the figure, its folder being created when
            missing; None shows it instead.
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
