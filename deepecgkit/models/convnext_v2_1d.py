import torch
from torch import nn

from deepecgkit.registry import register_model


class GRN(nn.Module):
    """
    Global Response Normalization layer for 1D signals.

    Normalizes feature responses globally across the temporal dimension,
    the key differentiator of ConvNeXtV2 over ConvNeXtV1.

    Args:
        dim: Number of channels
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class LayerNorm1d(nn.Module):
    """
    LayerNorm for (B, C, T) tensors, normalizing over the channel dimension.

    Args:
        dim: Number of channels
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class ConvNeXtV2Block1D(nn.Module):
    """
    Single ConvNeXtV2 block with inverted bottleneck and GRN.

    depthwise conv -> LayerNorm -> pointwise expand -> GELU -> GRN -> pointwise project

    Args:
        dim: Number of input/output channels
        expansion_factor: Expansion ratio for inverted bottleneck (default: 4)
        kernel_size: Kernel size for depthwise convolution (default: 7)
    """

    def __init__(self, dim: int, expansion_factor: int = 4, kernel_size: int = 7):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = LayerNorm1d(dim)
        self.pwconv_expand = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv_project = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv_expand(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv_project(x)
        return x + residual


@register_model(
    name="convnext-v2",
    description="Modern ConvNet with depthwise convolutions and Global Response Normalization for ECG",
)
class ConvNeXtV21D(nn.Module):
    """
    ConvNeXtV2 adapted for 1D ECG signals.

    A modern convolutional architecture using depthwise separable convolutions,
    LayerNorm, GELU activation, and Global Response Normalization (GRN).

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        dims: Channel dimensions for each stage (default: [64, 128, 256, 512])
        depths: Number of blocks per stage (default: [2, 2, 6, 2])
        kernel_size: Kernel size for depthwise convolutions (default: 7)
        expansion_factor: Expansion ratio for inverted bottleneck (default: 4)
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = ConvNeXtV21D(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 512)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        dims: list | None = None,
        depths: list | None = None,
        kernel_size: int = 7,
        expansion_factor: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if dims is None:
            dims = [64, 128, 256, 512]
        if depths is None:
            depths = [2, 2, 6, 2]

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=7, stride=4, padding=3),
            LayerNorm1d(dims[0]),
        )

        stages = []
        for i in range(len(dims)):
            stage_blocks = []
            for _ in range(depths[i]):
                stage_blocks.append(ConvNeXtV2Block1D(dims[i], expansion_factor, kernel_size))
            stages.append(nn.Sequential(*stage_blocks))

        self.stages = nn.ModuleList(stages)

        self.downsamples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.downsamples.append(
                nn.Sequential(
                    LayerNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.final_norm = nn.LayerNorm(dims[-1])
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = dims[-1]

        self.classifier = nn.Linear(dims[-1], output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.final_norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
