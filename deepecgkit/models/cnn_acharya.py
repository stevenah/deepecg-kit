"""Nine-layer heartbeat CNN of Acharya et al. (2017)."""

import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


@register_model(
    name="cnn-acharya",
    description="Nine-layer heartbeat CNN (Acharya et al. 2017): 3 convolutions, 3 dense layers",
    min_input_length=28,
)
class CNNAcharya(ECGModelBase):
    """
    The nine-layer convolutional network of Acharya et al. for heartbeats.

    Three unpadded convolutions with stride 1 (5, 10 and 20 filters; kernel
    sizes 3, 4 and 4), each followed by max pooling by 2, then fully connected
    layers of 30 and 20 units, with leaky ReLU after every convolution and
    hidden layer. The paper does not give the leaky ReLU slope; PyTorch's
    default of 0.01 is used. The 20-unit layer is the feature vector.

    The original classifies single beats of 260 samples at 360 Hz, which
    leave 30 steps of 20 maps to flatten; here the last feature map is
    averaged over time to 30 steps, so that any input length gives the same
    600 values. For a 260-sample beat the pooling changes nothing. The layer
    table follows the paper as quoted in its author-posted copy and agrees
    with public reimplementations; it has not been checked against the
    publisher's version.

    Reference:
        Acharya U.R. et al. "A deep convolutional neural network model to
        classify heartbeats" Computers in Biology and Medicine (2017)

    Args:
        input_channels: Number of input leads (default: 1)
        output_size: Number of output classes
        negative_slope: Leaky ReLU slope for negative inputs (default: 0.01)
        time_steps: Temporal steps kept before flattening (default: 30)

    Example:
        >>> model = CNNAcharya(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 20)
    """

    #: (filters, kernel size) of the three convolutions.
    CONVOLUTIONS = ((5, 3), (10, 4), (20, 4))
    HIDDEN = (30, 20)

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 5,
        negative_slope: float = 0.01,
        time_steps: int = 30,
    ):
        super().__init__()

        self.negative_slope = negative_slope
        convs = []
        in_channels = input_channels
        for filters, kernel_size in self.CONVOLUTIONS:
            convs.append(nn.Conv1d(in_channels, filters, kernel_size))
            in_channels = filters
        self.convs = nn.ModuleList(convs)
        self.time_pool = nn.AdaptiveAvgPool1d(time_steps)
        self.fc1 = nn.Linear(in_channels * time_steps, self.HIDDEN[0])
        self.fc2 = nn.Linear(self.HIDDEN[0], self.HIDDEN[1])

        self._feature_dim = self.HIDDEN[1]
        self.classifier = nn.Linear(self.HIDDEN[1], output_size)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = func.max_pool1d(func.leaky_relu(conv(x), self.negative_slope), 2)
        x = torch.flatten(self.time_pool(x), 1)
        x = func.leaky_relu(self.fc1(x), self.negative_slope)
        return func.leaky_relu(self.fc2(x), self.negative_slope)
