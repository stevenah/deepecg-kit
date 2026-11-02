"""Residual network of Ribeiro et al. (2020) for 12-lead ECG diagnosis."""

import itertools

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model

# Keras batch normalization defaults, which the original network was trained with.
_BN_EPS = 1e-3
_BN_MOMENTUM = 0.01


def _same_pad(x: torch.Tensor, kernel_size: int, stride: int, value: float = 0.0) -> torch.Tensor:
    """Pad as TensorFlow's "same" does: ceil(length / stride) outputs, the odd sample on the right."""
    length = x.shape[-1]
    total = max((-(-length // stride) - 1) * stride + kernel_size - length, 0)
    return func.pad(x, (total // 2, total - total // 2), value=value)


class RibeiroBlock(nn.Module):
    """
    Residual unit with two paths, as in the network of Ribeiro et al.

    The block carries two tensors: ``x``, the activated input of the main
    path, and ``y``, the un-activated sum that feeds the skip path. The skip
    path is max-pooled by the downsampling factor and projected by a 1x1
    convolution when the width changes. The main path is a convolution, batch
    normalization, ReLU and dropout, then a strided convolution; the sum of
    both paths is the next ``y``, and its batch normalization, ReLU and
    dropout the next ``x``.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        downsample: Temporal downsampling factor
        dropout_rate: Dropout probability after each activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        downsample: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=downsample, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels, eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.dropout = nn.Dropout(dropout_rate)
        self.skip_conv = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.downsample > 1:
            y = func.max_pool1d(
                _same_pad(y, self.downsample, self.downsample, float("-inf")), self.downsample
            )
        if self.skip_conv is not None:
            y = self.skip_conv(y)

        x = self.conv1(_same_pad(x, self.kernel_size, 1))
        x = self.dropout(func.relu(self.bn1(x)))
        x = self.conv2(_same_pad(x, self.kernel_size, self.downsample))

        y = x + y
        x = self.dropout(func.relu(self.bn2(y)))
        return x, y


@register_model(
    name="resnet-ribeiro",
    description="Residual network (Ribeiro et al. 2020) for 12-lead ECG diagnosis",
)
class ResNetRibeiro(ECGModelBase):
    """
    The network of Ribeiro et al. for automatic diagnosis of the 12-lead ECG.

    A convolution with 64 filters, batch normalization and ReLU is followed
    by four residual units with 128, 196, 256 and 320 filters, each
    downsampling by 4, with kernel size 16, "same" padding as in the original
    Keras model, Keras batch-normalization settings, and dropout of 0.8, the
    rate of the released model. Convolutions use He-normal initialization.
    The weights on Zenodo store a Keras Dropout rate of 0.8, as the
    tensorflow-v1 code gives; the repository's later master branch reads the
    same 0.8 as a keep probability, which would be a rate of 0.2.

    The original flattens a 16-step feature map, which a 4096-sample input
    produces; here the final feature map is averaged over time to 16 steps
    before flattening, so that any input length gives the same 5120-wide
    vector. For a 4096-sample input the pooling changes nothing.

    Reference:
        Ribeiro A.H. et al. "Automatic diagnosis of the 12-lead ECG using a
        deep neural network" Nature Communications (2020);
        github.com/antonior92/automatic-ecg-diagnosis (MIT licence)

    Args:
        input_channels: Number of input leads (default: 12)
        output_size: Number of output classes
        kernel_size: Convolution kernel size (default: 16)
        block_dropout: Dropout probability inside the residual units (default: 0.8)
        time_steps: Temporal steps kept before flattening (default: 16)

    Example:
        >>> model = ResNetRibeiro(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 5120)
    """

    #: Filters of the stem, then of each residual unit.
    FILTERS = (64, 128, 196, 256, 320)
    DOWNSAMPLE = 4

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        kernel_size: int = 16,
        block_dropout: float = 0.8,
        time_steps: int = 16,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stem_conv = nn.Conv1d(input_channels, self.FILTERS[0], kernel_size, bias=False)
        self.stem_bn = nn.BatchNorm1d(self.FILTERS[0], eps=_BN_EPS, momentum=_BN_MOMENTUM)
        self.blocks = nn.ModuleList(
            RibeiroBlock(c_in, c_out, kernel_size, self.DOWNSAMPLE, block_dropout)
            for c_in, c_out in itertools.pairwise(self.FILTERS)
        )
        self.time_pool = nn.AdaptiveAvgPool1d(time_steps)

        self._feature_dim = self.FILTERS[-1] * time_steps
        self.classifier = nn.Linear(self._feature_dim, output_size)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = func.relu(self.stem_bn(self.stem_conv(_same_pad(x, self.kernel_size, 1))))
        y = x
        for block in self.blocks:
            x, y = block(x, y)
        return torch.flatten(self.time_pool(x), 1)
