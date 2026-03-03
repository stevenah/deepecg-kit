import math

import torch
from torch import nn

from deepecgkit.registry import register_model


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.

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

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@register_model(
    name="transformer",
    description="Transformer-based ECG classifier",
)
class TransformerECG(nn.Module):
    """
    Transformer-based model for ECG signal classification.

    A transformer architecture that uses self-attention mechanisms
    to capture long-range dependencies in ECG signals.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        d_model: Dimension of the model (default: 128)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of transformer encoder layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 512)
        dropout_rate: Dropout probability (default: 0.1)
        max_len: Maximum sequence length (default: 5000)

    Example:
        >>> model = TransformerECG(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 128) with d_model=128
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model

        self.input_projection = nn.Linear(input_channels, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._feature_dim = d_model

        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward // 2, output_size),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        return self.classifier(x)
