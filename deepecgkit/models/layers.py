"""Building blocks shared across model architectures.

Modules and helpers here are used by more than one architecture file. They are
deliberately small and parameter-compatible with the code they replaced, so
existing checkpoints keep loading.
"""

import math

import torch
from torch import nn

__all__ = ["PositionalEncoding", "lstm_final_hidden"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-style branches.

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """

    pe: torch.Tensor

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
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"The sequence is {x.size(1)} time steps long but the positional encoding "
                f"covers {self.pe.size(1)}. Pass shorter signals (a shorter segment or a lower "
                f"sampling rate), or build the model with a larger max_len where it takes one."
            )
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def lstm_final_hidden(
    h_n: torch.Tensor,
    *,
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Return the final-layer hidden state of an (optionally bidirectional) LSTM.

    For a bidirectional LSTM the forward and backward hidden states of the last
    layer are concatenated, giving a (batch, 2 * hidden_size) tensor; otherwise
    the last layer's hidden state is returned as (batch, hidden_size).

    Args:
        h_n: Hidden state as returned by nn.LSTM, shaped
            (num_layers * num_directions, batch, hidden_size)
        num_layers: Number of LSTM layers
        batch_size: Batch size of the input
        hidden_size: LSTM hidden size
        bidirectional: Whether the LSTM is bidirectional
    """
    if bidirectional:
        h_n = h_n.view(num_layers, 2, batch_size, hidden_size)[-1]
        return torch.cat([h_n[0], h_n[1]], dim=1)
    return h_n[-1]
