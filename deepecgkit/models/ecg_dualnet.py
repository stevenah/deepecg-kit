import torch
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.models.layers import PositionalEncoding, lstm_final_hidden
from deepecgkit.registry import register_model


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
        return lstm_final_hidden(
            h_n,
            num_layers=self.lstm.num_layers,
            batch_size=batch_size,
            hidden_size=self.lstm_hidden,
        )


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
    min_input_length=8,
    max_input_length=5000,
)
class ECGDualNet(ECGModelBase):
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

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        cnn_lstm_out = self.cnn_lstm_branch(x)
        transformer_out = self.transformer_branch(x)
        return torch.cat([cnn_lstm_out, transformer_out], dim=1)
