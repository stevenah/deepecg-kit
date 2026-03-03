import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class ResidualBlock1D(nn.Module):
    """
    1D Residual block for ECG signal processing.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 7)
        stride: Convolution stride (default: 1)
        downsample: Downsample layer for skip connection if needed
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        downsample=None,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = func.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = func.relu(out)

        return out


@register_model(
    name="resnet",
    description="1D ResNet architecture adapted for ECG",
)
class ResNet1D(nn.Module):
    """
    1D ResNet model for ECG signal classification.

    A residual network architecture adapted for 1D time-series ECG data,
    providing deep feature extraction with skip connections.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        base_channels: Base number of channels (default: 64)
        num_blocks: List of number of blocks in each layer (default: [2, 2, 2, 2])
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = ResNet1D(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 512) with base_channels=64
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        base_channels: int = 64,
        num_blocks: list | None = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = base_channels

        self.conv1 = nn.Conv1d(
            input_channels,
            base_channels,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = base_channels * 8
        self.fc = nn.Linear(self._feature_dim, output_size)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(
            ResidualBlock1D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
