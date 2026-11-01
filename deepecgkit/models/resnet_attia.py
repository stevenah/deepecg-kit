"""Residual AI-ECG network of Attia et al. (2019) for atrial fibrillation."""

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


class AttiaBlock(nn.Module):
    """
    Pre-activation residual block that halves the temporal axis.

    The main arm is batch normalization, ReLU and a convolution along time,
    twice; the second convolution has stride 2. The skip arm is a 1x1
    convolution followed by max pooling by 2. Every lead is processed by the
    same filters, so the lead axis is carried through unchanged.

    Args:
        in_channels: Number of input feature maps
        out_channels: Number of output feature maps
        kernel_size: Temporal kernel size (odd)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size // 2, 0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, (kernel_size, 1), stride=(2, 1), padding=padding
        )
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.skip_pool = nn.MaxPool2d((2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(func.relu(self.bn1(x)))
        out = self.conv2(func.relu(self.bn2(out)))
        return out + self.skip_pool(self.skip_conv(x))


@register_model(
    name="resnet-attia",
    description="Residual AI-ECG network (Attia et al. 2019): 9 temporal blocks, then a lead convolution",
)
class ResNetAttia(ECGModelBase):
    """
    The atrial fibrillation network of Attia et al. (Lancet 2019).

    Nine pre-activation residual blocks convolve each lead along time with
    shared filters (kernel sizes 7, 5 and 3 for 16, 32 and 64 feature maps,
    three blocks each), each block halving the temporal axis, with dropout of
    0.2 after every third block. A convolution spanning all leads then
    combines them into 128 feature maps, followed by batch normalization,
    ReLU, dropout and the classifier. The input is zero-padded at the end to
    a multiple of 512 samples, as the original pads 5000 to 5120.

    The original combines 8 leads; the lead convolution here spans however
    many leads the input has. Its feature map is averaged over time to the
    original's 10 steps before flattening, so that any input length gives the
    same 1280-wide vector; for 10 s at 500 Hz the pooling changes nothing.

    Reference:
        Attia Z.I. et al. "An artificial intelligence-enabled ECG algorithm
        for the identification of patients with atrial fibrillation during
        sinus rhythm: a retrospective analysis of outcome prediction"
        The Lancet (2019), supplementary appendix

    Args:
        input_channels: Number of input leads (default: 12)
        output_size: Number of output classes
        lead_filters: Feature maps of the lead convolution (default: 128)
        time_steps: Temporal steps kept before flattening (default: 10)
        block_dropout: Dropout probability after every third block (default: 0.2)
        dropout_rate: Dropout probability before the classifier (default: 0.2)

    Example:
        >>> model = ResNetAttia(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 1280)
    """

    #: Kernel size and feature maps of the three groups of three blocks.
    STAGES = ((7, 16), (5, 32), (3, 64))
    BLOCKS_PER_STAGE = 3

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        lead_filters: int = 128,
        time_steps: int = 10,
        block_dropout: float = 0.2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = 1
        for kernel_size, channels in self.STAGES:
            for _ in range(self.BLOCKS_PER_STAGE):
                layers.append(AttiaBlock(in_channels, channels, kernel_size))
                in_channels = channels
            layers.append(nn.Dropout(block_dropout))
        self.blocks = nn.Sequential(*layers)
        self.multiple = 2 ** (len(self.STAGES) * self.BLOCKS_PER_STAGE)

        self.lead_conv = nn.Conv2d(in_channels, lead_filters, (1, input_channels))
        self.lead_bn = nn.BatchNorm2d(lead_filters)
        self.time_pool = nn.AdaptiveAvgPool1d(time_steps)

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = lead_filters * time_steps
        self.classifier = nn.Linear(self._feature_dim, output_size)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = func.pad(x, (0, -x.shape[-1] % self.multiple))
        x = x.transpose(1, 2).unsqueeze(1)  # (batch, 1, time, leads)
        x = self.blocks(x)
        x = func.relu(self.lead_bn(self.lead_conv(x))).squeeze(-1)  # (batch, maps, time)
        return torch.flatten(self.time_pool(x), 1)
