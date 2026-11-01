"""Structured state space (S4) classifier of Mehari and Strodthoff (2022).

The S4 layer is a self-contained port of the standalone S4 module of
state-spaces/s4 (Gu, Goel and Ré; Apache License 2.0), at the version that
github.com/tmehari/ssm_ecg (MIT licence) vendors, reduced to the
configuration that repository trains on PTB-XL: a HiPPO-LegS state matrix in
normal-plus-low-rank form with a rank-1 correction, bilinear discretization,
a kernel computed from its truncated generating function by a Cauchy sum and
a Woodbury correction, and a bidirectional convolution. The Cauchy sum is
evaluated directly in PyTorch, which needs neither the CUDA extension nor
pykeops of the original; for the small state size used here this is cheap.
"""

import math

import numpy as np
import torch
import torch.nn.functional as func
from torch import nn
from torch.utils.checkpoint import checkpoint

from deepecgkit.models.base import ECGModelBase
from deepecgkit.registry import register_model


def _legs_nplr(state_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HiPPO-LegS in normal-plus-low-rank form, diagonalized.

    ``A + P P^T`` is normal for the LegS matrix ``A`` and ``P_n =
    sqrt(n + 1/2)``. In the eigenbasis ``V`` of that sum, ``A`` becomes
    ``diag(w) - (V^* P)(V^* P)^*``. Eigenvalues come in conjugate pairs; one of
    each pair is kept, as the full system is recovered by conjugate symmetry.

    Returns:
        ``w``, ``P`` and ``B`` in the eigenbasis, each of size
        ``state_size // 2``, as complex128.
    """
    q = np.arange(state_size, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    transition = -(np.where(row >= col, 2 * q + 1, 0) - np.diag(q))
    scale = np.sqrt(np.diag(2 * q + 1))
    a = scale @ transition @ np.linalg.inv(scale)
    b = np.diag(scale)
    p = np.sqrt(0.5 + q)

    w, v = np.linalg.eig(a + np.outer(p, p))
    keep = w.imag > 0
    if keep.sum() != state_size // 2:
        raise ValueError(f"LegS of size {state_size} has unpaired eigenvalues")
    w, v = w[keep], v[:, keep]
    v_inv = v.conj().T
    return (torch.as_tensor(x, dtype=torch.cdouble) for x in (w, v_inv @ p, v_inv @ b))


class S4Kernel(nn.Module):
    """
    Convolution kernel of a diagonal-plus-low-rank state space model.

    ``A``, ``B``, ``P`` and the step sizes are fixed at initialization and
    stored as buffers; only ``C`` is trained, as in the published model. ``C``
    is stored as ``C (I - A_d^L)`` for the reference length ``L``, the form the
    truncated generating function uses. A kernel longer than ``L`` is computed
    from the correspondingly extended ``C`` without changing the parameter;
    the original rewrites the parameter in place to the same effect.

    Args:
        d_model: Number of independent channels (H)
        state_size: State size N, of which N/2 conjugate pairs are stored
        l_max: Reference kernel length L
        channels: Number of output kernels per channel (2 when bidirectional)
        dt_min: Smallest initial step size
        dt_max: Largest initial step size
    """

    def __init__(
        self,
        d_model: int,
        state_size: int,
        l_max: int,
        channels: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        self.l_max = l_max
        w, p, b = _legs_nplr(state_size)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register_buffer("log_dt", log_dt)
        self.register_buffer("w", torch.view_as_real(w.to(torch.cfloat)))
        self.register_buffer("P", torch.view_as_real(p.to(torch.cfloat)))
        self.register_buffer("B", torch.view_as_real(b.to(torch.cfloat)))

        c = torch.randn(channels, d_model, state_size // 2, dtype=torch.cfloat).conj()
        c = self._extend(c.resolve_conj(), l_max, sign=-1.0)
        self.C = nn.Parameter(torch.view_as_real(c.contiguous()))

    def _discrete_a(self) -> torch.Tensor:
        """Bilinear discretization of the full (conjugate-augmented) ``A``, per channel."""
        w = torch.view_as_complex(self.w).to(torch.cdouble)
        p = torch.view_as_complex(self.P).to(torch.cdouble)
        w, p = torch.cat([w, w.conj()]), torch.cat([p, p.conj()])
        a = torch.diag_embed(w) - torch.outer(p, p.conj())
        two_over_dt = (2.0 / torch.exp(self.log_dt.to(torch.double)))[:, None, None]
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        return torch.linalg.solve(two_over_dt * eye - a, two_over_dt * eye + a)

    def _extend(self, c: torch.Tensor, length: int, sign: float) -> torch.Tensor:
        """Return the stored half of ``[c, conj(c)] (I + sign A_d^length)``."""
        half = c.shape[-1]
        power = torch.linalg.matrix_power(self._discrete_a(), length).to(c.dtype)
        full = torch.cat([c, c.conj()], dim=-1)
        full = full + sign * torch.einsum("chm,hmn->chn", full, power)
        return full[..., :half]

    def forward(self, length: int) -> torch.Tensor:
        """Return the kernel of the given length, shape (channels, d_model, length)."""
        c = torch.view_as_complex(self.C)
        size = self.l_max
        while size < length:
            c = self._extend(c, size, sign=1.0)
            size *= 2

        dt = torch.exp(self.log_dt)
        w = torch.view_as_complex(self.w) * dt[:, None]
        p = torch.view_as_complex(self.P).expand_as(w)
        b = torch.view_as_complex(self.B).expand_as(w)

        omega = torch.tensor(np.exp(-2j * np.pi / size), dtype=torch.cfloat, device=w.device)
        omega = omega ** torch.arange(size // 2 + 1, device=w.device)
        z = 2 * (1 - omega) / (1 + omega)

        # Cauchy sums over each conjugate pair for every (input, output) pairing of
        # [B, P] with [C, Q], Q = conj(P); then the rank-1 Woodbury correction.
        v = torch.stack([b, p])[:, None] * torch.cat([c, p.conj()[None]])[None]
        resolvent = 1 / (z[None, :, None] - w[:, None, :])
        resolvent_conj = 1 / (z[None, :, None] - w.conj()[:, None, :])
        r = torch.einsum("abhn,hln->abhl", v, resolvent)
        r = (r + torch.einsum("abhn,hln->abhl", v.conj(), resolvent_conj)) * dt[:, None]
        k_f = r[:-1, :-1] - r[:-1, -1:] * r[-1:, :-1] / (1 + r[-1:, -1:])
        k_f = k_f * 2 / (1 + omega)
        return torch.fft.irfft(k_f, n=size)[0, ..., :length]


class S4Layer(nn.Module):
    """
    S4 layer: a state space convolution with skip term ``D``, GELU, channel
    dropout and a position-wise linear layer with a gated linear unit.

    When bidirectional, a second kernel is applied to the reversed sequence
    and both are summed into one two-sided convolution.

    Args:
        d_model: Number of channels (H)
        state_size: State size N
        l_max: Reference kernel length
        dropout: Channel dropout probability after the activation
        bidirectional: Whether to also convolve the reversed sequence
    """

    def __init__(
        self,
        d_model: int,
        state_size: int,
        l_max: int,
        dropout: float,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.D = nn.Parameter(torch.randn(1, d_model))
        self.kernel = S4Kernel(d_model, state_size, l_max, channels=2 if bidirectional else 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        # A position-wise linear layer, initialized as nn.Linear is.
        self.output_linear = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=-2)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        length = u.shape[-1]
        k = self.kernel(length)
        if self.bidirectional:
            k = func.pad(k[0], (0, length)) + func.pad(k[1].flip(-1), (length, 0))
        else:
            k = k[0]
        n = 2 * length
        y = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :length]
        y = y + u * self.D[0, :, None]
        y = self.dropout(self.activation(y))
        return self.glu(self.output_linear(y))


@register_model(
    name="s4",
    description="Bidirectional S4 state space model (Mehari & Strodthoff 2022): 4 layers, mean readout",
)
class S4ECG(ECGModelBase):
    """
    The S4 classifier of Mehari and Strodthoff for 12-lead ECG analysis.

    A pointwise convolution maps the leads to 512 channels, followed by four
    bidirectional S4 layers with state size 8, each in a post-norm residual
    block (channel dropout of 0.2 on the S4 output, addition, layer
    normalization). The sequence is averaged over time and passed to a
    linear classifier. The state matrix, input matrix and step sizes stay at
    their initial values; ``C``, ``D`` and the linear layers are trained.

    The reference kernel length defaults to 5000 samples, 10 s at 500 Hz;
    the original ties it to its 250-sample input. Kernels for other lengths
    are exact truncations or extensions, so it only fixes how ``C`` is stored.

    Reference:
        Mehari T., Strodthoff N. "Advancing the State-of-the-Art for ECG
        Analysis through Structured State Space Models" ML4H (2022);
        github.com/tmehari/ssm_ecg (MIT licence). S4: Gu A., Goel K., Ré C.
        "Efficiently Modeling Long Sequences with Structured State Spaces"
        ICLR (2022); github.com/state-spaces/s4 (Apache License 2.0)

    Args:
        input_channels: Number of input leads (default: 12)
        output_size: Number of output classes
        d_model: Number of channels (default: 512)
        state_size: State size N (default: 8)
        num_layers: Number of S4 layers (default: 4)
        dropout: Channel dropout inside and after each S4 layer (default: 0.2)
        l_max: Reference kernel length (default: 5000)
        bidirectional: Whether the S4 layers are bidirectional (default: True)

    Example:
        >>> model = S4ECG(input_channels=12, output_size=5)
        >>> x = torch.randn(4, 12, 5000)
        >>> output = model(x)
        >>> print(output.shape)

        >>> features = model.extract_features(x)
        >>> print(features.shape)  # (4, 512)
    """

    def __init__(
        self,
        input_channels: int = 12,
        output_size: int = 5,
        d_model: int = 512,
        state_size: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        l_max: int = 5000,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.encoder = nn.Conv1d(input_channels, d_model, 1)
        self.layers = nn.ModuleList(
            S4Layer(d_model, state_size, l_max, dropout, bidirectional) for _ in range(num_layers)
        )
        self.layer_dropouts = nn.ModuleList(nn.Dropout1d(dropout) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(num_layers))

        self._feature_dim = d_model
        self.classifier = nn.Linear(d_model, output_size)

    def _residual_block(self, x: torch.Tensor, index: int) -> torch.Tensor:
        x = x + self.layer_dropouts[index](self.layers[index](x))
        return self.norms[index](x.transpose(-1, -2)).transpose(-1, -2)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for index in range(len(self.layers)):
            if self.training and torch.is_grad_enabled():
                # Recompute each block in the backward pass instead of keeping its FFT
                # intermediates, which for 32 ten-second 12-lead inputs at 500 Hz would
                # take about 25 GB. Dropout masks are replayed, so gradients are unchanged.
                x = checkpoint(self._residual_block, x, index, use_reentrant=False)
            else:
                x = self._residual_block(x, index)
        return x.mean(dim=-1)
