import torch
from torch import nn

from deepecgkit.registry import register_model


@register_model(
    name="simple-cnn",
    description="Lightweight CNN for fast inference",
)
class SimpleCNN(nn.Module):
    """
    Simple CNN model for ECG signal classification.

    A straightforward convolutional neural network with pooling and dropout
    layers for basic ECG classification tasks.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        dropout_rate: Dropout probability (default: 0.3)

    Example:
        >>> model = SimpleCNN(input_channels=1, output_size=4)
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
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout_rate),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size),
        )

    @property
    def feature_dim(self) -> int:
        return 256

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
