import torch
from torch import nn

from deepecgkit.registry import register_model


@register_model(
    name="lstm",
    description="LSTM-based model for sequential ECG analysis",
)
class LSTMECG(nn.Module):
    """
    LSTM-based model for ECG signal classification.

    A recurrent neural network using bidirectional LSTM layers
    for temporal pattern recognition in ECG signals.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        hidden_size: Size of LSTM hidden state (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout_rate: Dropout probability (default: 0.3)
        bidirectional: Use bidirectional LSTM (default: True)

    Example:
        >>> model = LSTMECG(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 256) with bidirectional=True
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = hidden_size * self.num_directions

        self.classifier = nn.Sequential(
            nn.Linear(self._feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _extract_lstm_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]

        return h_n

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_lstm_features(x)

    def forward(self, x):
        x = self.extract_features(x)
        x = self.dropout(x)
        return self.classifier(x)
