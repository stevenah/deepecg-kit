from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model
from deepecgkit.utils.weights import get_weight_info, load_pretrained_weights


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return func.relu(self.bn(self.conv(x)))


class KanResModule(nn.Module):
    """Residual module for the KanResWideX architecture."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


@register_model(
    name="kanres",
    description="KAN-ResNet architecture with wide layers",
)
class KanResWideX(nn.Module):
    """
    KanRes-Wide-X model for ECG signal classification.

    A convolutional neural network architecture designed for ECG signal analysis
    with residual connections and wide blocks for improved feature extraction.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes or regression targets
        base_channels: Base number of channels for the first layer (default: 64)

    Example:
        >>> model = KanResWideX(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)  # [32, 4]

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 64)
    """

    def __init__(self, input_channels: int = 1, output_size: int = 4, base_channels: int = 64):
        super().__init__()

        self.input_layer = ConvBlock(input_channels, base_channels)
        self.res_modules = nn.Sequential(
            KanResModule(base_channels), KanResModule(base_channels), KanResModule(base_channels)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._feature_dim = base_channels
        self.classifier = nn.Linear(base_channels, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_modules(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        return self.classifier(x)

    @classmethod
    def from_pretrained(
        cls,
        weights: str,
        map_location: Optional[Union[str, torch.device]] = None,
        force_download: bool = False,
        **kwargs,
    ) -> "KanResWideX":
        """Load a pretrained KanResWideX model.

        Args:
            weights: Name of pretrained weights (e.g., "kanres-af-30s") or path to weights file
            map_location: Device to map weights to (e.g., "cpu", "cuda")
            force_download: If True, re-download weights even if cached
            **kwargs: Override default model parameters from the weight registry

        Returns:
            Model with pretrained weights loaded

        Example:
            >>> model = KanResWideX.from_pretrained("kanres-af-30s")
            >>> model = KanResWideX.from_pretrained("kanres-af-30s", map_location="cuda")
            >>> model = KanResWideX.from_pretrained("/path/to/weights.pt", output_size=2)
        """
        weight_path = Path(weights)
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location=map_location, weights_only=True)
            model = cls(**kwargs)
        else:
            info = get_weight_info(weights)
            model_kwargs = {**info["model_kwargs"], **kwargs}
            model = cls(**model_kwargs)
            state_dict = load_pretrained_weights(weights, map_location, force_download)

        model.load_state_dict(state_dict)
        return model
