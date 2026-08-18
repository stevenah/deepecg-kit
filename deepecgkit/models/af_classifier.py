import torch
from torch import nn

from deepecgkit.models.pretrained import PretrainedMixin
from deepecgkit.registry import register_model


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_dropout: bool = False,
        use_pooling: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        self.use_pooling = use_pooling

        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        if use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.batch_norm(x))

        if self.use_dropout:
            x = self.dropout(x)
        if self.use_pooling:
            x = self.pool(x)

        return x


@register_model(
    name="afmodel",
    description="Atrial Fibrillation model optimized for 30s ECG segments",
    default_kwargs={"recording_length": 30},
)
class AFModel(PretrainedMixin, nn.Module):
    """
    Atrial Fibrillation classification model.

    A convolutional neural network specifically designed for AF detection
    in ECG signals with configurable recording length support.

    Args:
        input_channels: Number of input ECG leads (default: 1)
        output_size: Number of output classes (default: 4)
        recording_length: Recording length in seconds (must be 6, 10, or 30)

    Example:
        >>> model = AFModel(recording_length=30)
        >>> x = torch.randn(32, 1, 9000)
        >>> output = model(x)
        >>> print(output.shape)  # (32, 4)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 196)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        recording_length: int = 30,
    ):
        super().__init__()
        assert recording_length in [6, 10, 30], (
            f"Recording length must be 6, 10, or 30, got {recording_length}"
        )

        self.recording_length = recording_length
        kernel_size = 6 if recording_length in [6, 10] else 32

        layer_configs = [
            (input_channels, 64, True, True),
            (64, 64, False, False),
            (64, 64, True, True),
            (64, 64, False, False),
            (64, 64, True, True),
            (64, 64, False, False),
            (64, 128, True, True),
            (128, 128, False, False),
            (128, 128, True, True),
            (128, 128, False, False),
            (128, 196, True, True),
            (196, 196, False, False),
            (196, 196, True, True),
        ]

        self.conv_layers = nn.ModuleList(
            [
                ConvBlock(in_ch, out_ch, kernel_size, dropout, pooling)
                for in_ch, out_ch, dropout, pooling in layer_configs
            ]
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self._feature_dim = 196
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(in_features=self._feature_dim, out_features=output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def min_input_length(self) -> int:
        """Smallest signal length the convolutional stack accepts.

        Each block narrows the signal by ``kernel_size - 1`` and optionally
        halves it, so the bound is obtained by walking the stack backwards from
        a single remaining sample.

        Returns:
            Minimum number of samples per lead.
        """
        required = 1
        for block in reversed(self.conv_layers):
            if block.use_pooling:
                required *= 2
            required += block.conv.kernel_size[0] - 1
        return required

    def extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """Return the pooled representation preceding dropout and the head.

        Args:
            data: Input of shape (batch, leads, time)

        Returns:
            Tensor of shape (batch, feature_dim).

        Raises:
            ValueError: If the signal is shorter than :attr:`min_input_length`.
                ``recording_length`` selects the convolution kernel size but is
                never compared against the data, so without this check a short
                signal fails deep inside the stack with an opaque kernel-size
                error.
        """
        if data.dim() != 3:
            raise ValueError(
                f"{type(self).__name__} expects input of shape (batch, leads, time), "
                f"got a {data.dim()}-D tensor of shape {tuple(data.shape)}."
            )
        if data.shape[-1] < self.min_input_length:
            raise ValueError(
                f"{type(self).__name__}(recording_length={self.recording_length}) needs at "
                f"least {self.min_input_length} samples per lead but received "
                f"{data.shape[-1]}. Pass longer signals, or build the model with a "
                f"recording_length matching the data."
            )

        for conv_layer in self.conv_layers:
            data = conv_layer(data)
        data = self.adaptive_pool(data)
        data = self.flatten(data)
        return data

    def forward(self, data):
        data = self.extract_features(data)
        data = self.dropout(data)
        return self.output(data)

    def get_feature_size(self, input_size: int) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)

            for conv_layer in self.conv_layers:
                dummy_input = conv_layer(dummy_input)

            return dummy_input.shape[-1]
