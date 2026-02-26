import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class BasicBlock1DWang(nn.Module):
    """
    Basic residual block for the Wang ResNet variant.

    Uses two convolutional layers with batch normalization and a skip connection.
    The second convolution uses a smaller kernel (kernel_size // 2 + 1).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Primary kernel size (default: 5)
        stride: Stride for the first convolution (default: 1)
        downsample: Optional downsample module for the skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        ks2 = kernel_size // 2 + 1

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=ks2,
            stride=1,
            padding=(ks2 - 1) // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = func.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return func.relu(out, inplace=True)


@register_model(
    name="resnet-wang",
    description="ResNet (Wang et al.) for time series classification",
)
class ResNetWang(nn.Module):
    """
    ResNet model based on Wang et al.'s architecture for time series classification.

    A shallow 3-block residual network without initial pooling, using larger
    initial channels (128) and asymmetric kernel sizes in residual blocks.
    This is the standard ResNet baseline from the PTB-XL benchmark.

    Reference:
        Wang Z., Yan W., Oates T. "Time Series Classification from Scratch
        with Deep Neural Networks: A Strong Baseline" (2017)
        https://github.com/helme/ecg_ptbxl_benchmarking

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        base_channels: Base number of channels (default: 128)
        kernel_size: Primary kernel size for residual blocks (default: 5)
        kernel_size_stem: Kernel size for the stem convolution (default: 7)
        dropout_rate: Dropout probability before classifier (default: 0.3)

    Example:
        >>> model = ResNetWang(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 1000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 128)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        base_channels: int = 128,
        kernel_size: int = 5,
        kernel_size_stem: int = 7,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.in_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                base_channels,
                kernel_size=kernel_size_stem,
                stride=1,
                padding=(kernel_size_stem - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(base_channels, 1, kernel_size=kernel_size, stride=1)
        self.layer2 = self._make_layer(base_channels, 1, kernel_size=kernel_size, stride=1)
        self.layer3 = self._make_layer(base_channels, 1, kernel_size=kernel_size, stride=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = base_channels
        self.classifier = nn.Linear(self._feature_dim, output_size)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        kernel_size: int = 5,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            BasicBlock1DWang(
                self.in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1DWang(out_channels, out_channels, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.classifier(x)
