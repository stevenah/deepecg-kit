"""ROCKET: random convolutional kernels with a linear classifier."""

import math

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


@register_model(
    name="rocket",
    description="ROCKET: frozen random convolution kernels, PPV and max pooling, linear head",
)
class ROCKET(ECGModelBase):
    """
    ROCKET features with a linear classifier, as a non-deep baseline.

    At construction, ``num_kernels`` convolution kernels of length
    ``kernel_length`` are drawn with standard-normal weights (mean-centred per
    kernel), a bias from U(-1, 1) and a dilation 2^e, with the exponents
    spread evenly over 0 .. floor(log2((reference_length - 1) /
    (kernel_length - 1))). The kernels are stored as buffers, never trained,
    and are saved with the checkpoint so that a run is reproducible. Each
    kernel yields two features: the proportion of positive values (PPV) and
    the maximum of its output. The 2 x ``num_kernels`` features are
    standardized by a parameter-free batch-normalization layer and classified
    by one linear layer, the only trainable part of the model.

    Differences from the original: every kernel sees all input channels,
    every kernel uses "same" padding (the original draws padding at random),
    and the linear classifier is fitted by gradient descent with the
    framework's loss instead of a ridge regression.

    Reference:
        Dempster A., Petitjean F., Webb G.I. "ROCKET: exceptionally fast and
        accurate time series classification using random convolutional
        kernels" Data Mining and Knowledge Discovery (2020)

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        num_kernels: Number of random kernels (default: 1000)
        kernel_length: Taps per kernel (default: 9)
        reference_length: Signal length used to bound the dilations (default: 5000)

    Example:
        >>> model = ROCKET(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 2000)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        num_kernels: int = 1000,
        kernel_length: int = 9,
        reference_length: int = 5000,
    ):
        super().__init__()

        self.kernel_length = kernel_length
        max_exponent = math.floor(math.log2((reference_length - 1) / (kernel_length - 1)))
        self.dilations = [2**e for e in range(max_exponent + 1)]
        per_group, remainder = divmod(num_kernels, len(self.dilations))
        for i, _dilation in enumerate(self.dilations):
            count = per_group + (1 if i < remainder else 0)
            weight = torch.randn(count, input_channels, kernel_length)
            weight = weight - weight.mean(dim=-1, keepdim=True)
            bias = torch.empty(count).uniform_(-1.0, 1.0)
            self.register_buffer(f"weight_{i}", weight)
            self.register_buffer(f"bias_{i}", bias)

        self._feature_dim = 2 * num_kernels
        self.norm = nn.BatchNorm1d(self._feature_dim, affine=False)
        self.classifier = nn.Linear(self._feature_dim, output_size)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for i, dilation in enumerate(self.dilations):
            weight = getattr(self, f"weight_{i}")
            bias = getattr(self, f"bias_{i}")
            total = (self.kernel_length - 1) * dilation
            padded = func.pad(x, (total // 2, total - total // 2))
            out = func.conv1d(padded, weight, bias, dilation=dilation)
            features.append((out > 0).to(out.dtype).mean(dim=-1))
            features.append(out.amax(dim=-1))
        return self.norm(torch.cat(features, dim=1))
