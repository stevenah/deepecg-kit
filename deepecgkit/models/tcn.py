import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that only looks at past and present timesteps.

    Uses left-side padding to ensure no information leakage from the future,
    making it suitable for real-time or streaming ECG inference.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation factor for dilated convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolutions and residual connection.

    Each block applies two dilated causal convolutions with weight normalization,
    ReLU activations, and dropout, plus a 1x1 shortcut if channel sizes differ.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 7)
        dilation: Dilation factor (default: 1)
        dropout_rate: Dropout probability (default: 0.2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.causal_conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        nn.utils.parametrizations.weight_norm(self.causal_conv1.conv)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.causal_conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        nn.utils.parametrizations.weight_norm(self.causal_conv2.conv)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.causal_conv1(x)
        out = func.relu(self.bn1(out))
        out = self.dropout1(out)

        out = self.causal_conv2(out)
        out = func.relu(self.bn2(out))
        out = self.dropout2(out)

        return func.relu(out + self.shortcut(x))


@register_model(
    name="tcn",
    description="Temporal Convolutional Network with dilated causal convolutions",
)
class TCN(nn.Module):
    """
    Temporal Convolutional Network for ECG signal classification.

    Uses stacked dilated causal convolutions with exponentially growing
    receptive fields to efficiently model long-range dependencies without
    recurrence. The causal structure makes it suitable for real-time
    ECG monitoring applications.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        num_channels: List of channel sizes per temporal block
            (default: [64, 64, 128, 128, 256, 256])
        kernel_size: Convolution kernel size (default: 7)
        dropout_rate: Dropout probability (default: 0.2)

    Example:
        >>> model = TCN(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 256) with default num_channels
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        num_channels: list | None = None,
        kernel_size: int = 7,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 128, 128, 256, 256]

        blocks = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_channels if i == 0 else num_channels[i - 1]
            dilation = 2**i
            blocks.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout_rate=dropout_rate,
                )
            )

        self.network = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._feature_dim = num_channels[-1]
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self._feature_dim, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
