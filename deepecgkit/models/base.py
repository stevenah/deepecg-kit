"""Shared base class for every registered ECG model.

All architectures follow the same convention: a backbone that produces a flat
feature vector (``extract_features``), an optional dropout, and a linear
classification head. This base class carries that shared machinery once —
``feature_dim``, the canonical ``forward``, and uniform head access — so each
model file only describes its architecture.

The head submodule keeps its historical per-class attribute name (``fc``,
``head``, ``output``, ...) because those names are state_dict keys: renaming
them would break every checkpoint users have already saved. ``HEAD_ATTR``
records the name so callers can reach the head generically via ``get_head`` /
``set_head`` without knowing it.
"""

from typing import ClassVar

import torch
from torch import nn

from deepecgkit.models.pretrained import PretrainedMixin

__all__ = ["ECGModelBase"]


class ECGModelBase(PretrainedMixin, nn.Module):
    """Base class for registered ECG models.

    Subclasses must:

    - call ``super().__init__()`` first,
    - set ``self._feature_dim`` to the backbone's output width,
    - create their head module under the attribute named by ``HEAD_ATTR``,
    - implement ``extract_features``.

    A subclass without its own top-level ``self.dropout`` inherits the
    parameter-free ``nn.Identity`` set here, keeping the default ``forward``
    (features -> dropout -> head) bit-identical to a model with no dropout.
    """

    #: Name of the attribute holding the classification head submodule.
    HEAD_ATTR: ClassVar[str] = "classifier"

    def __init__(self) -> None:
        super().__init__()
        self.dropout: nn.Module = nn.Identity()

    @property
    def feature_dim(self) -> int:
        """Width of the flat feature vector produced by ``extract_features``."""
        return self._feature_dim

    def get_head(self) -> nn.Module:
        """Return the classification head, whatever its attribute name."""
        return getattr(self, self.HEAD_ATTR)

    def set_head(self, module: nn.Module) -> None:
        """Replace the classification head, preserving its attribute name."""
        setattr(self, self.HEAD_ATTR, module)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Map an input batch to flat feature vectors of size ``feature_dim``."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.get_head()(x)
