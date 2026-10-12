"""2-D convolutional network on per-lead log-magnitude spectrograms."""

import torch
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, ceil_mode=True),
    )


@register_model(
    name="spectrogram-cnn",
    description="2-D CNN on per-lead log-magnitude STFT spectrograms",
    min_input_length=128,
)
class SpectrogramCNN(ECGModelBase):
    """
    Time-frequency baseline: a 2-D CNN applied to spectrograms of each lead.

    Each lead is transformed by a short-time Fourier transform with a Hann
    window of ``n_fft`` samples and hop ``hop_length`` (centred, so the number
    of frames is about ``length / hop_length``), and the log of one plus the
    magnitude forms one image channel per lead. Four blocks of two 3 x 3
    convolutions with batch normalization, ReLU and 2 x 2 max pooling are
    followed by global average pooling over frequency and time. This is the
    spectrogram-input design common among entries to the PhysioNet/CinC 2017
    challenge, without a published reference implementation.

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        n_fft: Window length of the STFT in samples (default: 128)
        hop_length: Hop between frames in samples (default: 32)
        channels: Output channels of the four convolution blocks
        dropout_rate: Dropout probability before the classifier (default: 0.3)

    Example:
        >>> model = SpectrogramCNN(input_channels=12, output_size=5)
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
        n_fft: int = 128,
        hop_length: int = 32,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        blocks = []
        in_channels = input_channels
        for out_channels in channels:
            blocks.append(_conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = in_channels
        self.classifier = nn.Linear(in_channels, output_size)

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Log-magnitude spectrogram of every lead: (batch, leads, freq, frames)."""
        batch, leads, length = x.shape
        spec = torch.stft(
            x.reshape(batch * leads, length),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.log1p(spec.abs())
        return magnitude.reshape(batch, leads, magnitude.shape[-2], magnitude.shape[-1])

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)
