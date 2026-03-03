import torch
from torch import nn

from deepecgkit.registry import register_model


@register_model(
    name="crnn",
    description="CNN-LSTM hybrid for local feature extraction and temporal aggregation",
)
class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for ECG classification.

    Uses a CNN front-end for local feature extraction followed by a
    bidirectional LSTM for temporal aggregation, combining the strengths
    of both architectures.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        cnn_channels: List of channel sizes for CNN stages (default: [32, 64, 128, 256])
        lstm_hidden_size: Size of LSTM hidden state (default: 128)
        lstm_num_layers: Number of LSTM layers (default: 2)
        bidirectional: Use bidirectional LSTM (default: True)
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = CRNN(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 256)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        cnn_channels: list | None = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        layers = []
        in_ch = input_channels
        kernel_sizes = [7, 5, 3, 3]
        for i, out_ch in enumerate(cnn_channels):
            ks = kernel_sizes[i] if i < len(kernel_sizes) else 3
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = lstm_hidden_size * self.num_directions

        self.classifier = nn.Sequential(
            nn.Linear(self._feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_n = h_n.view(self.lstm_num_layers, 2, batch_size, self.lstm_hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        return h_n

    def forward(self, x):
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.classifier(x)
