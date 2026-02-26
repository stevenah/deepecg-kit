import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class InceptionBlock1D(nn.Module):
    """
    Single Inception block with multi-scale 1D convolutions.

    Applies parallel convolutions with different kernel sizes to capture
    temporal patterns at multiple scales, plus a max-pool branch.

    Args:
        in_channels: Number of input channels
        n_filters: Number of filters per branch (default: 32)
        kernel_sizes: Tuple of kernel sizes for parallel branches
        bottleneck_channels: Channels in bottleneck layer (default: 32)
        use_bottleneck: Whether to use 1x1 bottleneck before convolutions
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (5, 15, 41),
        bottleneck_channels: int = 32,
        use_bottleneck: bool = True,
    ):
        super().__init__()

        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            conv_in = in_channels

        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_in,
                    n_filters,
                    kernel_size=ks,
                    padding=ks // 2,
                    bias=False,
                )
                for ks in kernel_sizes
            ]
        )

        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )

        total_filters = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(total_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bottleneck:
            bottleneck_out = self.bottleneck(x)
            conv_outputs = [branch(bottleneck_out) for branch in self.conv_branches]
        else:
            conv_outputs = [branch(x) for branch in self.conv_branches]

        pool_output = self.maxpool_branch(x)
        conv_outputs.append(pool_output)

        out = torch.cat(conv_outputs, dim=1)
        out = self.bn(out)
        out = func.relu(out)
        return out


class InceptionResidualBlock1D(nn.Module):
    """
    Inception block with a residual shortcut connection.

    Args:
        in_channels: Number of input channels
        n_filters: Number of filters per Inception branch
        kernel_sizes: Tuple of kernel sizes for parallel branches
        bottleneck_channels: Channels in bottleneck layer
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int = 32,
        kernel_sizes: tuple[int, ...] = (5, 15, 41),
        bottleneck_channels: int = 32,
    ):
        super().__init__()

        self.inception = InceptionBlock1D(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
        )

        out_channels = n_filters * (len(kernel_sizes) + 1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inception(x)
        shortcut = self.shortcut(x)
        return func.relu(out + shortcut)


@register_model(
    name="inception-time",
    description="Multi-scale temporal CNN inspired by InceptionTime",
)
class InceptionTime1D(nn.Module):
    """
    InceptionTime model adapted for 1D ECG signal classification.

    Uses parallel convolutions at multiple temporal scales (short/medium/long
    kernels) with residual connections to capture both rapid arrhythmia spikes
    and slow rhythm patterns simultaneously.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        n_filters: Number of filters per Inception branch (default: 32)
        depth: Number of Inception residual blocks (default: 6)
        kernel_sizes: Tuple of kernel sizes for multi-scale branches
        bottleneck_channels: Channels in bottleneck layers (default: 32)
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = InceptionTime1D(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 128) with n_filters=32
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        n_filters: int = 32,
        depth: int = 6,
        kernel_sizes: tuple[int, ...] = (5, 15, 41),
        bottleneck_channels: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        n_branches = len(kernel_sizes) + 1
        block_out_channels = n_filters * n_branches

        blocks = []
        for i in range(depth):
            in_ch = input_channels if i == 0 else block_out_channels
            blocks.append(
                InceptionResidualBlock1D(
                    in_channels=in_ch,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = block_out_channels

        self.classifier = nn.Linear(block_out_channels, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
