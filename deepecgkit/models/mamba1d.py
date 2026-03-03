import torch
import torch.nn.functional as func
from torch import nn

from deepecgkit.registry import register_model


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model implemented in pure PyTorch.

    Implements a discretized state space: h_t = A_bar * h_{t-1} + B_bar * x_t,
    y_t = C * h_t, where B, C, and delta are input-dependent (selective).

    Args:
        d_inner: Inner model dimension
        d_state: State space dimension (default: 16)
        d_conv: Local convolution width (default: 4)
    """

    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        a_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        a_log = a_log.unsqueeze(0).expand(d_inner, -1)
        self.a_log = nn.Parameter(a_log)

        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = func.silu(x_conv).transpose(1, 2)

        proj = self.x_proj(x_conv)
        b_sel = proj[:, :, : self.d_state]
        c_sel = proj[:, :, self.d_state : self.d_state * 2]
        delta = proj[:, :, self.d_state * 2 :]

        delta = func.softplus(self.dt_proj(delta))

        a_coeff = -torch.exp(self.a_log)

        outputs = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            dt = delta[:, t].unsqueeze(-1)
            a_bar = torch.exp(dt * a_coeff)
            b_bar = dt * b_sel[:, t].unsqueeze(1)
            x_t = x_conv[:, t].unsqueeze(-1)

            h = a_bar * h + b_bar * x_t
            y_t = (h * c_sel[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y + x * self.D


class MambaBlock(nn.Module):
    """
    Single Mamba block: norm -> linear expand -> SSM -> gated output -> project + residual.

    Args:
        d_model: Model dimension
        d_state: State space dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expansion_factor: Inner dimension expansion (default: 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expansion_factor: int = 2,
    ):
        super().__init__()
        d_inner = d_model * expansion_factor

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.ssm = SelectiveSSM(d_inner, d_state, d_conv)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.ssm(x)
        x = x * func.silu(z)
        x = self.out_proj(x)
        return x + residual


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba: runs forward and backward SSMs and combines outputs.

    Args:
        d_model: Model dimension
        d_state: State space dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expansion_factor: Inner dimension expansion (default: 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expansion_factor: int = 2,
    ):
        super().__init__()
        self.forward_block = MambaBlock(d_model, d_state, d_conv, expansion_factor)
        self.backward_block = MambaBlock(d_model, d_state, d_conv, expansion_factor)
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_out = self.forward_block(x)
        backward_out = self.backward_block(x.flip(1)).flip(1)
        return self.combine(torch.cat([forward_out, backward_out], dim=-1))


@register_model(
    name="mamba",
    description="Bidirectional Mamba state space model with linear complexity for ECG",
)
class Mamba1D(nn.Module):
    """
    Bidirectional Mamba model for ECG classification.

    Uses selective state space models with linear complexity as an
    alternative to Transformer self-attention. The SSM is implemented
    from scratch in pure PyTorch with no external dependencies.

    Args:
        input_channels: Number of input channels (default: 1 for single-lead ECG)
        output_size: Number of output classes
        d_model: Model dimension (default: 128)
        d_state: State space dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expansion_factor: Inner dimension expansion (default: 2)
        num_layers: Number of bidirectional Mamba layers (default: 4)
        patch_size: Patch size for input tokenization (default: 50)
        dropout_rate: Dropout probability (default: 0.1)
        max_patches: Maximum number of patches (default: 500)

    Example:
        >>> model = Mamba1D(input_channels=1, output_size=4)
        >>> x = torch.randn(32, 1, 3000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (32, 128)
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_size: int = 4,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expansion_factor: int = 2,
        num_layers: int = 4,
        patch_size: int = 50,
        dropout_rate: float = 0.1,
        max_patches: int = 500,
    ):
        super().__init__()

        self.patch_embed = nn.Conv1d(
            input_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        self.layers = nn.ModuleList(
            [
                BidirectionalMamba(d_model, d_state, d_conv, expansion_factor)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self._feature_dim = d_model

        self.classifier = nn.Linear(d_model, output_size)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.transpose(1, 2)
        num_patches = x.size(1)
        x = x + self.pos_embedding[:, :num_patches]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
