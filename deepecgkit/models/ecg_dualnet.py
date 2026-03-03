import math

import torch
from torch import nn

from deepecgkit.registry import register_model


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer branch.

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNLSTMBranch(nn.Module):
    """
    CNN feature extraction followed by LSTM temporal aggregation.

    Args:
        input_channels: Number of input ECG channels
        cnn_channels: Output channels for CNN stages (default: 128)
        lstm_hidden: LSTM hidden size (default: 128)
        lstm_layers: Number of LSTM layers (default: 1)
        dropout_rate: Dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_channels: int,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        self.output_dim = lstm_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.view(-1, 2, batch_size, self.lstm_hidden)
        h_n = h_n[-1]
        return torch.cat([h_n[0], h_n[1]], dim=1)


class TransformerBranch(nn.Module):
    """
    Transformer encoder for global temporal attention.

    Args:
        input_channels: Number of input ECG channels
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 2)
        dim_feedforward: Feedforward dimension (default: 256)
        dropout_rate: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        input_channels: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        return x.squeeze(-1)


@register_model(
    name="dualnet",
    description="Dual-path architecture with CNN-LSTM and Transformer branches for ECG",
)
class ECGDualNet(nn.Module):
    """
    Dual-path ECG classification network.

    Runs a CNN-LSTM branch and a Transformer branch in parallel,
    then fuses their outputs via concatenation and a fusion head.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        cnn_channels: CNN output channels in CNN-LSTM branch (default: 128)
        lstm_hidden: LSTM hidden size in CNN-LSTM branch (default: 128)
        lstm_layers: Number of LSTM layers (default: 1)
        d_model: Transformer model dimension (default: 128)
        nhead: Number of attention heads (default: 8)
        transformer_layers: Number of transformer encoder layers (default: 2)
        dim_feedforward: Transformer feedforward dimension (default: 256)
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = ECGDualNet(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 384)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.cnn_lstm_branch = CNNLSTMBranch(
            input_channels=input_channels,
            cnn_channels=cnn_channels,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate,
        )

        self.transformer_branch = TransformerBranch(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout_rate=dropout_rate,
        )

        self._feature_dim = self.cnn_lstm_branch.output_dim + self.transformer_branch.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(self._feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        cnn_lstm_out = self.cnn_lstm_branch(x)
        transformer_out = self.transformer_branch(x)
        return torch.cat([cnn_lstm_out, transformer_out], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.classifier(x)
