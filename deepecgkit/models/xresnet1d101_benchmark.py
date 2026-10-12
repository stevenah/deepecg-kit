"""XResNet1d-101 as configured in the PTB-XL benchmark."""

from deepecgkit.models.xresnet1d_benchmark import XResNet1dBenchmark
from deepecgkit.registry import register_model


@register_model(
    name="xresnet1d101",
    description="XResNet1d-101 (bottleneck, [3, 4, 23, 3]) as in the PTB-XL benchmark",
)
class XResNet1d101Benchmark(XResNet1dBenchmark):
    """
    The deep XResNet1d of the PTB-XL benchmark, ``xresnet1d101``.

    Bottleneck blocks (expansion 4), ``[3, 4, 23, 3]`` blocks per stage with
    stage widths 64, 128, 256 and 512 before expansion, kernel size 5, a
    three-convolution stem and a concatenated average-and-maximum pooling head,
    following the ``xresnet1d101`` constructor of the PTB-XL benchmark code.
    The benchmark's fixed-width ``xresnet1d-benchmark`` is the shallow
    counterpart of this model.

    Reference:
        Strodthoff N., Wagner P., Schaeffter T., Samek W. "Deep Learning for
        ECG Analysis: Benchmarks and Insights from PTB-XL" (2021)
        https://github.com/helme/ecg_ptbxl_benchmarking

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        kernel_size: Convolution kernel size (default: 5)
        kernel_size_stem: Stem convolution kernel size (default: 5)
        dropout_rate: Dropout probability in head (default: 0.5)

    Example:
        >>> model = XResNet1d101Benchmark(input_channels=12, output_size=5)
        >>> x = torch.randn(4, 12, 1000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (4, 4096)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        kernel_size: int = 5,
        kernel_size_stem: int = 5,
        dropout_rate: float = 0.5,
    ):
        super().__init__(
            input_channels=input_channels,
            output_size=output_size,
            expansion=4,
            layers=[3, 4, 23, 3],
            kernel_size=kernel_size,
            kernel_size_stem=kernel_size_stem,
            dropout_rate=dropout_rate,
            concat_pooling=True,
            block_channels=(64, 128, 256, 512),
        )
