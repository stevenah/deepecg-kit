import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.pretrained import PretrainedMixin
from deepecgkit.registry import register_model


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
class KanResWideX(PretrainedMixin, nn.Module):
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
