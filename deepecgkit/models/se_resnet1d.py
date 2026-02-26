import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class SqueezeExcitation1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D signals.

    Learns per-channel attention weights via global pooling followed by a
    bottleneck MLP, allowing the network to emphasize informative channels
    (e.g., specific ECG leads) and suppress less useful ones.

    Args:
        channels: Number of input/output channels
        reduction: Channel reduction ratio for the bottleneck (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.fc2 = nn.Linear(mid, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        scale = x.mean(dim=2)
        scale = func.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.view(b, c, 1)


class SEResidualBlock1D(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 7)
        stride: Convolution stride (default: 1)
        downsample: Downsample layer for skip connection if needed
        se_reduction: SE reduction ratio (default: 16)
        dropout_rate: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        downsample: nn.Module | None = None,
        se_reduction: int = 16,
        dropout_rate: float = 0.0,
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

        self.se = SqueezeExcitation1D(out_channels, reduction=se_reduction)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = func.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = func.relu(out)
        return out


@register_model(
    name="se-resnet",
    description="ResNet with Squeeze-and-Excitation channel attention for ECG",
)
class SEResNet1D(nn.Module):
    """
    SE-ResNet model for ECG signal classification.

    Extends ResNet1D with Squeeze-and-Excitation blocks that learn channel
    attention weights. Particularly effective for multi-lead ECG where the
    network can learn which leads are most informative for each class.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        base_channels: Base number of channels (default: 64)
        num_blocks: List of number of blocks in each layer (default: [2, 2, 2, 2])
        se_reduction: SE reduction ratio (default: 16)
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = SEResNet1D(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 5000)
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
        se_reduction: int = 16,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = base_channels
        self.se_reduction = se_reduction
        self.dropout_rate = dropout_rate

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

        layers = [
            SEResidualBlock1D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                se_reduction=self.se_reduction,
                dropout_rate=self.dropout_rate,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                SEResidualBlock1D(
                    out_channels,
                    out_channels,
                    se_reduction=self.se_reduction,
                    dropout_rate=self.dropout_rate,
                )
            )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
