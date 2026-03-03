import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class StemResBlock2D(nn.Module):
    """
    First residual block following the stem convolution.

    Unlike subsequent blocks, this starts directly with convolution
    (no pre-activation BN/ReLU), matching the Keras reference architecture.
    All convolutions use valid padding (no padding).

    Args:
        in_channels: Number of input filter channels
        out_channels: Number of output filter channels
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=(1, 2))

        self.shortcut_pool = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 2))
        self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = func.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)

        shortcut = self.shortcut_pool(x)
        shortcut = self.shortcut_proj(shortcut)

        return out + shortcut


class PreActResBlock2D(nn.Module):
    """
    Pre-activation 2D residual block for blocks 2-4.

    Follows BN -> ReLU -> Dropout -> Conv -> BN -> ReLU -> Dropout -> Conv(stride=2)
    pattern matching the Keras reference model. All convolutions use valid padding.

    Args:
        in_channels: Number of input filter channels
        out_channels: Number of output filter channels
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=(1, 2))

        self.shortcut_pool = nn.MaxPool2d(kernel_size=(1, 9), stride=(1, 2))
        self.shortcut_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = func.relu(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = func.relu(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        shortcut = self.shortcut_pool(x)
        shortcut = self.shortcut_proj(shortcut)

        return out + shortcut


@register_model(
    name="deep-res-cnn",
    description="Deep Residual 2D CNN (Elyamani et al. 2022) for multi-lead ECG classification",
)
class DeepResCNN(nn.Module):
    """
    Deep Residual 2D CNN for ECG classification.

    Faithful implementation of Elyamani et al. (2022). Uses Conv2d with (1, k)
    kernels and valid padding to process each ECG lead independently along the
    time axis, then fuses across leads with a (leads, 1) convolution at the end.

    Input convention: (batch, leads, time) -- standard deepecg-kit format.
    Internally reshaped to (batch, 1, leads, time) for 2D convolution.

    The classifier head includes L2 regularization matching the original Keras
    model. Call l2_regularization_loss() to obtain the penalty term.

    Reference:
        https://github.com/HaneenElyamani/ECG-classification

    Args:
        input_channels: Number of ECG leads (default: 12)
        output_size: Number of output classes (default: 5)
        dropout_rate: Dropout probability in residual blocks (default: 0.1)

    Example:
        >>> model = DeepResCNN(input_channels=12, output_size=5)
        >>> x = torch.randn(32, 12, 1000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 128)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.stem_conv = nn.Conv2d(1, 32, kernel_size=(1, 7))
        self.stem_bn = nn.BatchNorm2d(32)

        self.block1 = StemResBlock2D(32, 64, dropout_rate=dropout_rate)
        self.block2 = PreActResBlock2D(64, 64, dropout_rate=dropout_rate)
        self.block3 = PreActResBlock2D(64, 128, dropout_rate=dropout_rate)
        self.block4 = PreActResBlock2D(128, 128, dropout_rate=dropout_rate)

        self.lead_fusion = nn.Conv2d(128, 128, kernel_size=(input_channels, 1))
        self.lead_fusion_bn = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self._feature_dim = 128

        self.fc1 = nn.Linear(128, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_drop = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc2_drop = nn.Dropout(0.15)

        self.fc_out = nn.Linear(64, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def l2_regularization_loss(self) -> torch.Tensor:
        """Return the L2 penalty for the classifier head weights.

        Matches the Keras kernel_regularizer=L2(lambda) on the two Dense layers.
        Add this to the training loss for full equivalence with the original model.
        """
        return 0.005 * self.fc1.weight.pow(2).sum() + 0.009 * self.fc2.weight.pow(2).sum()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = func.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.lead_fusion(x)
        x = self.lead_fusion_bn(x)
        x = func.relu(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)

        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = func.relu(x)
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = func.relu(x)
        x = self.fc2_drop(x)

        x = self.fc_out(x)
        return x
