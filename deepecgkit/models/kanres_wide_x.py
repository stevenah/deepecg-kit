import torch
from torch import nn

from deepecgkit.registry import register_model


class KanResInit(nn.Module):
    """Initial convolutional block for KanResWideX architecture.

    Args:
        in_channels: Number of input channels
        filterno_1: Number of filters in first convolution
        filterno_2: Number of filters in second convolution
        filtersize_1: Kernel size for first convolution
        filtersize_2: Kernel size for second convolution
        stride: Stride for first convolution
    """

    def __init__(
        self,
        in_channels: int,
        filterno_1: int,
        filterno_2: int,
        filtersize_1: int,
        filtersize_2: int,
        stride: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filterno_1, filtersize_1, stride=stride)
        self.bn1 = nn.BatchNorm1d(filterno_1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(filterno_1, filterno_2, filtersize_2)
        self.bn2 = nn.BatchNorm1d(filterno_2)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class KanResModule(nn.Module):
    """Residual module with skip connection for KanResWideX architecture.

    Args:
        in_channels: Number of input channels
        filterno_1: Number of filters in first convolution
        filterno_2: Number of filters in second convolution
        filtersize_1: Kernel size for first convolution
        filtersize_2: Kernel size for second convolution
        stride: Stride for first convolution
    """

    def __init__(
        self,
        in_channels: int,
        filterno_1: int,
        filterno_2: int,
        filtersize_1: int,
        filtersize_2: int,
        stride: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filterno_1, filtersize_1, stride=stride, padding="same")
        self.bn1 = nn.BatchNorm1d(filterno_1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(filterno_1, filterno_2, filtersize_2, padding="same")
        self.bn2 = nn.BatchNorm1d(filterno_2)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = out + identity
        return out


@register_model(
    name="kanres-deep",
    description="Deep KAN-ResNet architecture",
)
class KanResDeepX(nn.Module):
    """KanRes-Deep-X model for ECG signal classification.

    A deep residual convolutional neural network architecture designed for ECG
    signal analysis with 8 residual blocks for improved feature extraction.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes (default: 4)
        base_channels: Base number of channels for the architecture (default: 32)

    Example:
        >>> model = KanResDeepX(input_channels=1, output_size=4, base_channels=32)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 32)
    """

    def __init__(self, input_channels: int = 1, output_size: int = 4, base_channels: int = 32):
        super().__init__()

        self.base_channels = base_channels
        init_channels = base_channels * 2

        self.init_block = KanResInit(input_channels, init_channels, base_channels, 8, 3, 1)
        self.pool = nn.AvgPool1d(kernel_size=2)

        self.res_modules = nn.ModuleList(
            [
                KanResModule(base_channels, base_channels * 2, base_channels, 50, 50, 1)
                for _ in range(8)
            ]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._feature_dim = base_channels
        self.fc = nn.Linear(base_channels, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_block(x)
        x = self.pool(x)

        for res_module in self.res_modules:
            x = res_module(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc(x)
        return x
