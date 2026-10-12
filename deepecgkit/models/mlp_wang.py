"""Multilayer perceptron baseline of Wang et al. (2017)."""

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


@register_model(
    name="mlp-wang",
    description="MLP (Wang et al.) on the flattened, time-pooled signal",
)
class MLPWang(ECGModelBase):
    """
    Multilayer perceptron baseline for time series classification.

    The input is first averaged over time to a fixed number of samples per
    lead (``pooled_length``), so that the model accepts any signal length,
    then flattened and passed through three fully connected layers of 500
    units with ReLU activations, with dropout of 0.1 on the input and 0.2
    after each hidden layer, as in the MLP of Wang et al. Averaging to a
    fixed length is an adaptation; the original flattens the series as is.

    Reference:
        Wang Z., Yan W., Oates T. "Time Series Classification from Scratch
        with Deep Neural Networks: A Strong Baseline" (2017)

    Args:
        input_channels: Number of input channels (default: 12 for 12-lead ECG)
        output_size: Number of output classes
        pooled_length: Samples per lead after temporal averaging (default: 1000)
        hidden_size: Width of the three hidden layers (default: 500)
        dropout_rate: Dropout probability before the classifier (default: 0.3)

    Example:
        >>> model = MLPWang(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 500)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        pooled_length: int = 1000,
        hidden_size: int = 500,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.pooled_length = pooled_length
        self.pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.input_dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_channels * pooled_length, hidden_size)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.drop3 = nn.Dropout(0.2)

        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = hidden_size
        self.classifier = nn.Linear(hidden_size, output_size)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.input_dropout(x)
        x = self.drop1(func.relu(self.fc1(x)))
        x = self.drop2(func.relu(self.fc2(x)))
        return self.drop3(func.relu(self.fc3(x)))
