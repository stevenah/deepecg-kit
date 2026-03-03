import torch
from torch import nn

from deepecgkit.registry import register_model


@register_model(
    name="fcn-wang",
    description="Fully Convolutional Network (Wang et al.) for time series classification",
)
class FCNWang(nn.Module):
    """
    Fully Convolutional Network for ECG signal classification.

    Based on Wang et al.'s FCN architecture for time series classification,
    using three convolutional blocks with batch normalization and global
    average pooling. No dense layers except the final classifier.

    Reference:
        Wang Z., Yan W., Oates T. "Time Series Classification from Scratch
        with Deep Neural Networks: A Strong Baseline" (2017)
        https://github.com/helme/ecg_ptbxl_benchmarking

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        filters: List of filter counts for each conv block (default: [128, 256, 128])
        kernel_sizes: List of kernel sizes for each conv block (default: [8, 5, 3])
        dropout_rate: Dropout probability before classifier (default: 0.3)

    Example:
        >>> model = FCNWang(input_channels=12, output_size=5)
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
        filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if filters is None:
            filters = [128, 256, 128]
        if kernel_sizes is None:
            kernel_sizes = [8, 5, 3]

        assert len(filters) == len(kernel_sizes)

        blocks = []
        in_ch = input_channels
        for nf, ks in zip(filters, kernel_sizes):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, nf, kernel_size=ks, padding=(ks - 1) // 2, bias=False),
                    nn.BatchNorm1d(nf),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = nf

        self.conv_blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = filters[-1]
        self.classifier = nn.Linear(self._feature_dim, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.classifier(x)
