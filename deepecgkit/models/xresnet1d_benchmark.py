import torch
from torch import nn

from deepecgkit.registry import register_model


def _init_cnn(m: nn.Module):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for child in m.children():
        _init_cnn(child)


class ConvBnAct(nn.Sequential):
    """Conv1d → BatchNorm1d → Activation."""

    def __init__(
        self,
        ni: int,
        nf: int,
        ks: int = 3,
        stride: int = 1,
        act: bool = True,
        zero_bn: bool = False,
    ):
        layers: list[nn.Module] = [
            nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=(ks - 1) // 2, bias=False),
        ]
        bn = nn.BatchNorm1d(nf)
        if zero_bn:
            nn.init.zeros_(bn.weight)
        layers.append(bn)
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class XResBlock(nn.Module):
    """
    XResNet residual block with expansion support.

    Args:
        expansion: Channel expansion factor (1 for BasicBlock, 4 for Bottleneck)
        ni: Input channels (before expansion)
        nf: Output channels (before expansion)
        stride: Stride for downsampling
        kernel_size: Convolution kernel size
    """

    def __init__(self, expansion: int, ni: int, nf: int, stride: int = 1, kernel_size: int = 5):
        super().__init__()
        nf_out = nf * expansion
        ni_in = ni * expansion

        if expansion == 1:
            self.convs = nn.Sequential(
                ConvBnAct(ni_in, nf, ks=kernel_size, stride=stride),
                ConvBnAct(nf, nf_out, ks=kernel_size, zero_bn=True, act=False),
            )
        else:
            self.convs = nn.Sequential(
                ConvBnAct(ni_in, nf, ks=1),
                ConvBnAct(nf, nf, ks=kernel_size, stride=stride),
                ConvBnAct(nf, nf_out, ks=1, zero_bn=True, act=False),
            )

        id_layers: list[nn.Module] = []
        if ni_in != nf_out:
            id_layers.append(ConvBnAct(ni_in, nf_out, ks=1, act=False))
        if stride != 1:
            id_layers.insert(0, nn.AvgPool1d(2, ceil_mode=True))
        self.idpath = nn.Sequential(*id_layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.idpath(x))


@register_model(
    name="xresnet1d-benchmark",
    description="XResNet1d from PTB-XL benchmark (fixed feature dim, concat pooling)",
)
class XResNet1dBenchmark(nn.Module):
    """
    XResNet1d adapted from the PTB-XL benchmarking repository.

    Key differences from the standard XResNet1D:
    - Fixed feature dimension across all layers (all blocks use same width)
    - Concat pooling head (AdaptiveAvgPool + AdaptiveMaxPool concatenated)
    - Multi-layer stem with configurable kernel size
    - Supports both BasicBlock (expansion=1) and Bottleneck (expansion=4)

    Reference:
        Strodthoff N., Wagner P., Schaeffter T., Samek W.
        "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL" (2021)
        https://github.com/helme/ecg_ptbxl_benchmarking

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        expansion: Block expansion factor (1=BasicBlock, 4=Bottleneck)
        layers: List of block counts per layer (default: [3, 4, 6, 3] for ResNet-50)
        base_channels: Fixed channel width for all layers (default: 64)
        kernel_size: Convolution kernel size (default: 5)
        kernel_size_stem: Stem convolution kernel size (default: 5)
        stem_channels: Tuple of stem layer channels
        dropout_rate: Dropout probability in head (default: 0.5)
        concat_pooling: Use avg+max concat pooling (default: True)

    Example:
        >>> model = XResNet1dBenchmark(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 1000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 128) with concat_pooling
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        expansion: int = 1,
        layers: list[int] | None = None,
        base_channels: int = 64,
        kernel_size: int = 5,
        kernel_size_stem: int = 5,
        stem_channels: tuple[int, ...] = (32, 32, 64),
        dropout_rate: float = 0.5,
        concat_pooling: bool = True,
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]

        self.expansion = expansion
        self.concat_pooling = concat_pooling

        stem_szs = [input_channels, *stem_channels]
        self.stem = nn.Sequential(
            *[
                ConvBnAct(
                    stem_szs[i], stem_szs[i + 1], ks=kernel_size_stem, stride=2 if i == 0 else 1
                )
                for i in range(len(stem_channels))
            ]
        )
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        block_szs = [64 // expansion, *([base_channels] * len(layers))]

        self.res_layers = nn.Sequential(
            *[
                self._make_layer(
                    ni=block_szs[i],
                    nf=block_szs[i + 1],
                    blocks=n_blocks,
                    stride=1 if i == 0 else 2,
                    kernel_size=kernel_size,
                )
                for i, n_blocks in enumerate(layers)
            ]
        )

        final_nf = block_szs[-1] * expansion
        self._feature_dim = final_nf * 2 if concat_pooling else final_nf

        if concat_pooling:
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)])
        else:
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool1d(1)])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(self._feature_dim),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self._feature_dim, output_size),
        )

        _init_cnn(self)

    def _make_layer(
        self, ni: int, nf: int, blocks: int, stride: int, kernel_size: int
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                XResBlock(
                    self.expansion,
                    ni if i == 0 else nf,
                    nf,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                )
                for i in range(blocks)
            ]
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_pool(x)
        x = self.res_layers(x)
        pooled = torch.cat([p(x) for p in self.pool], dim=1)
        return torch.flatten(pooled, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.head(x)


def xresnet1d18_benchmark(**kwargs) -> XResNet1dBenchmark:
    return XResNet1dBenchmark(expansion=1, layers=[2, 2, 2, 2], **kwargs)


def xresnet1d50_benchmark(**kwargs) -> XResNet1dBenchmark:
    return XResNet1dBenchmark(expansion=4, layers=[3, 4, 6, 3], **kwargs)


def xresnet1d101_benchmark(**kwargs) -> XResNet1dBenchmark:
    return XResNet1dBenchmark(expansion=4, layers=[3, 4, 23, 3], **kwargs)
