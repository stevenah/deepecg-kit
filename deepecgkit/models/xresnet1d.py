import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(func.softplus(x))


class BlurPool1D(nn.Module):
    """
    Anti-aliased downsampling via blur-then-subsample.

    Applies a fixed triangular low-pass filter before strided subsampling
    to reduce aliasing artifacts during spatial reduction.

    Args:
        channels: Number of channels (applied per-channel)
        stride: Downsampling stride (default: 2)
    """

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        kernel = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1)
        self.register_buffer("kernel", kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = func.pad(x, (1, 1), mode="reflect")
        x = func.conv1d(x, self.kernel, stride=self.stride, groups=self.channels)
        return x


class XResBlock1D(nn.Module):
    """
    Improved residual block with Mish activation and optional blur-pool downsampling.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 7)
        stride: Convolution stride (default: 1)
        downsample: Downsample layer for skip connection if needed
        use_blur_pool: Whether to use anti-aliased downsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        downsample: nn.Module | None = None,
        use_blur_pool: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
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

        self.act = Mish()
        self.downsample = downsample

        if stride > 1 and use_blur_pool:
            self.pool = BlurPool1D(out_channels, stride=stride)
        elif stride > 1:
            self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)
        else:
            self.pool = nn.Identity()

        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


@register_model(
    name="xresnet",
    description="Improved ResNet with Mish activation and blur-pool downsampling",
)
class XResNet1D(nn.Module):
    """
    XResNet model for ECG signal classification.

    An improved ResNet incorporating three key enhancements from recent
    research: (1) a multi-layer stem instead of a single large convolution,
    (2) Mish activation for smoother gradients, and (3) anti-aliased
    blur-pool downsampling to reduce aliasing.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        base_channels: Base number of channels (default: 64)
        num_blocks: List of number of blocks in each layer (default: [2, 2, 2, 2])
        dropout_rate: Dropout probability (default: 0.3)
        use_blur_pool: Whether to use anti-aliased downsampling (default: True)

    Example:
        >>> model = XResNet1D(input_channels=1, output_size=4)
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
        use_blur_pool: bool = True,
    ):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = base_channels
        self.use_blur_pool = use_blur_pool
        self.act = Mish()

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                base_channels // 2,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels // 2),
            Mish(),
            nn.Conv1d(
                base_channels // 2,
                base_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels // 2),
            Mish(),
            nn.Conv1d(
                base_channels // 2,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            Mish(),
        )

        if use_blur_pool:
            self.stem_pool = BlurPool1D(base_channels, stride=2)
        else:
            self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

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
            ds_layers: list[nn.Module] = []
            if stride > 1:
                if self.use_blur_pool:
                    ds_layers.append(BlurPool1D(self.in_channels, stride=stride))
                else:
                    ds_layers.append(nn.AvgPool1d(kernel_size=stride, stride=stride))
            ds_layers.append(nn.Conv1d(self.in_channels, out_channels, kernel_size=1, bias=False))
            ds_layers.append(nn.BatchNorm1d(out_channels))
            downsample = nn.Sequential(*ds_layers)

        layers = [
            XResBlock1D(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                use_blur_pool=self.use_blur_pool,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                XResBlock1D(
                    out_channels,
                    out_channels,
                    use_blur_pool=self.use_blur_pool,
                )
            )

        return nn.Sequential(*layers)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_pool(x)

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
