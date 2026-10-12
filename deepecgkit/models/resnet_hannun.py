"""34-layer residual network of Hannun et al. (2019)."""

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


def _same_pad(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Pad so that a stride-1 convolution keeps the length, also for even kernels."""
    total = kernel_size - 1
    return func.pad(x, (total // 2, total - total // 2))


class HannunBlock(nn.Module):
    """
    Pre-activation residual block with two convolutions.

    Each convolution is preceded by batch normalization and ReLU, and the
    second one also by dropout. When the block subsamples, the first
    convolution has stride 2 and the shortcut is max-pooled; when the channel
    count changes, the shortcut is projected by a 1-sample convolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 16)
        stride: Stride of the first convolution (default: 1)
        dropout_rate: Dropout probability before the second convolution (default: 0.2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 16,
        stride: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, bias=False)
        self.shortcut_pool = nn.MaxPool1d(stride, ceil_mode=True) if stride > 1 else None
        self.shortcut_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.shortcut_pool is not None:
            identity = self.shortcut_pool(identity)
        if self.shortcut_proj is not None:
            identity = self.shortcut_proj(identity)

        out = func.relu(self.bn1(x))
        out = self.conv1(_same_pad(out, self.kernel_size))
        out = self.dropout(func.relu(self.bn2(out)))
        out = self.conv2(_same_pad(out, self.kernel_size))
        return out + identity


@register_model(
    name="resnet-hannun",
    description="34-layer residual network (Hannun et al. 2019) with 16 pre-activation blocks",
)
class ResNetHannun(ECGModelBase):
    """
    The 34-layer convolutional network of Hannun et al. for rhythm classification.

    A stem convolution is followed by 16 residual blocks of two convolutions
    each, all with kernel size 16. The number of filters starts at 64 and
    increases by 64 every fourth block; every second block subsamples by 2,
    so the output resolution is 1/256 of the input. The original network
    emits one prediction per output step; this implementation pools the final
    feature map over time and applies a single classifier (an adaptation).

    Reference:
        Hannun A.Y. et al. "Cardiologist-level arrhythmia detection and
        classification in ambulatory electrocardiograms using a deep neural
        network" Nature Medicine (2019)

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        base_channels: Filters of the first block group (default: 64)
        num_blocks: Number of residual blocks (default: 16)
        kernel_size: Convolution kernel size (default: 16)
        block_dropout: Dropout probability inside each block (default: 0.2)
        dropout_rate: Dropout probability before the classifier (default: 0.3)

    Example:
        >>> model = ResNetHannun(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 256)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        base_channels: int = 64,
        num_blocks: int = 16,
        kernel_size: int = 16,
        block_dropout: float = 0.2,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stem_conv = nn.Conv1d(input_channels, base_channels, kernel_size, bias=False)
        self.stem_bn = nn.BatchNorm1d(base_channels)

        blocks = []
        in_channels = base_channels
        for i in range(num_blocks):
            out_channels = base_channels * (i // 4 + 1)
            blocks.append(
                HannunBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2 if i % 2 == 1 else 1,
                    dropout_rate=block_dropout,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.final_bn = nn.BatchNorm1d(in_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = in_channels
        self.classifier = nn.Linear(in_channels, output_size)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_conv(_same_pad(x, self.kernel_size))
        x = func.relu(self.stem_bn(x))
        x = self.blocks(x)
        x = func.relu(self.final_bn(x))
        x = self.global_pool(x)
        return torch.flatten(x, 1)
