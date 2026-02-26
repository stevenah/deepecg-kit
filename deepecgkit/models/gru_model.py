import torch
from torch import nn

from deepecgkit.registry import register_model


@register_model(
    name="gru",
    description="GRU-based model for sequential ECG analysis",
)
class GRUECG(nn.Module):
    """
    GRU-based model for ECG signal classification.

    A recurrent neural network using GRU layers with concat pooling
    (adaptive avg + adaptive max + last hidden state) for ECG classification.
    Based on the RNN architecture from the PTB-XL benchmark.

    Reference:
        https://github.com/helme/ecg_ptbxl_benchmarking

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        hidden_size: Size of GRU hidden state (default: 256)
        num_layers: Number of GRU layers (default: 2)
        dropout_rate: Dropout probability (default: 0.3)
        bidirectional: Use bidirectional GRU (default: False)

    Example:
        >>> model = GRUECG(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 1000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 768) with hidden_size=256, unidirectional
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        rnn_out_dim = hidden_size * self.num_directions
        self._feature_dim = rnn_out_dim * 3

        self.head = nn.Sequential(
            nn.BatchNorm1d(self._feature_dim),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self._feature_dim, output_size),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _concat_pool(self, rnn_output: torch.Tensor) -> torch.Tensor:
        x = rnn_output.transpose(1, 2)
        avg_pool = torch.mean(x, dim=2)
        max_pool, _ = torch.max(x, dim=2)

        if self.bidirectional:
            last = torch.cat(
                [rnn_output[:, -1, : self.hidden_size], rnn_output[:, 0, self.hidden_size :]], dim=1
            )
        else:
            last = rnn_output[:, -1, :]

        return torch.cat([avg_pool, max_pool, last], dim=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        output, _ = self.gru(x)
        return self._concat_pool(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        return self.head(x)
