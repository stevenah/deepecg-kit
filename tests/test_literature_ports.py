"""Ports of published ECG networks reproduce their sources.

Each check pins a port to something stated by its source: the parameter count
of released weights, the feature-map sizes of a paper's layer table, or, for
S4, the defining property of its kernel (the impulse response of the
discretized state space model).
"""

import pytest
import torch
import torch.nn.functional as func

from deepecgkit.models.cnn_acharya import CNNAcharya
from deepecgkit.models.resnet_attia import ResNetAttia
from deepecgkit.models.resnet_ribeiro import ResNetRibeiro
from deepecgkit.models.s4 import S4ECG, S4Kernel


def _trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestResNetRibeiro:
    def test_parameter_count_matches_the_released_model(self):
        """The released Keras weights (Zenodo 3625017) hold 6,421,910 trainable values."""
        assert _trainable(ResNetRibeiro(input_channels=12, output_size=6)) == 6_421_910

    def test_each_unit_divides_the_length_by_four(self):
        model = ResNetRibeiro(input_channels=12, output_size=6).eval()
        x = func.relu(model.stem_bn(model.stem_conv(func.pad(torch.randn(1, 12, 4096), (7, 8)))))
        y, lengths = x, []
        for block in model.blocks:
            x, y = block(x, y)
            lengths.append(x.shape[-1])
        assert lengths == [1024, 256, 64, 16]

    @pytest.mark.parametrize("length", [4096, 5000, 300])
    def test_any_length_gives_the_original_feature_width(self, length):
        model = ResNetRibeiro(input_channels=1, output_size=4).eval()
        assert model.extract_features(torch.randn(2, 1, length)).shape == (2, 5120)


class TestResNetAttia:
    def test_ten_seconds_at_500_hz_leave_the_original_ten_steps(self):
        model = ResNetAttia(input_channels=12, output_size=2).eval()
        x = func.pad(torch.randn(1, 12, 5000), (0, 120)).transpose(1, 2).unsqueeze(1)
        assert model.blocks(x).shape == (1, 64, 10, 12)

    @pytest.mark.parametrize("leads,length", [(12, 5000), (1, 5000), (1, 300)])
    def test_any_input_gives_the_original_feature_width(self, leads, length):
        model = ResNetAttia(input_channels=leads, output_size=4).eval()
        assert model.extract_features(torch.randn(2, leads, length)).shape == (2, 1280)


class TestCNNAcharya:
    def test_a_beat_follows_the_paper_layer_table(self):
        model = CNNAcharya(input_channels=1, output_size=5).eval()
        x, shapes = torch.randn(1, 1, 260), []
        for conv in model.convs:
            x = func.max_pool1d(func.leaky_relu(conv(x)), 2)
            shapes.append(tuple(x.shape[1:]))
        assert shapes == [(5, 129), (10, 63), (20, 30)]

    def test_features_are_the_twenty_unit_layer(self):
        model = CNNAcharya(input_channels=1, output_size=4).eval()
        assert model.extract_features(torch.randn(2, 1, 5000)).shape == (2, 20)


def _impulse_response(kernel: S4Kernel, length: int) -> torch.Tensor:
    """C A_d^l B_d of the full discretized system, stepped one sample at a time."""
    with torch.no_grad():
        a_d = kernel._discrete_a()
        channels, size = a_d.shape[0], a_d.shape[-1]
        w, p, b = (torch.view_as_complex(t).cdouble() for t in (kernel.w, kernel.P, kernel.B))
        w, p, b = (torch.cat([t, t.conj()]) for t in (w, p, b))
        a = torch.diag_embed(w) - torch.outer(p, p.conj())
        two_over_dt = (2 / torch.exp(kernel.log_dt.double()))[:, None, None]
        eye = torch.eye(size, dtype=torch.cdouble)
        state = torch.linalg.solve(
            two_over_dt * eye - a, (2 * b)[None, :, None].expand(channels, size, 1)
        )
        state = state.squeeze(-1)
        # The parameter stores C (I - A_d^L); undo that to get C.
        stored = torch.view_as_complex(kernel.C).cdouble()
        stored = torch.cat([stored, stored.conj()], dim=-1)
        correction = eye - torch.linalg.matrix_power(a_d, kernel.l_max)
        c = torch.einsum("chm,hmn->chn", stored, torch.linalg.inv(correction))
        taps = []
        for _ in range(length):
            taps.append(torch.einsum("chn,hn->ch", c, state).real)
            state = torch.einsum("hmn,hn->hm", a_d, state)
        return torch.stack(taps, dim=-1)


class TestS4:
    @pytest.mark.parametrize("length", [40, 64, 200])
    def test_kernel_is_the_impulse_response_of_the_discretized_system(self, length):
        """Shorter, equal and longer than the reference length, in both directions."""
        torch.manual_seed(0)
        kernel = S4Kernel(d_model=6, state_size=8, l_max=64, channels=2)
        with torch.no_grad():
            fast = kernel(length).double()
        reference = _impulse_response(kernel, length)
        assert (fast - reference).abs().max() <= 1e-5 * reference.abs().max()

    def test_a_longer_kernel_leaves_the_parameter_unchanged(self):
        kernel = S4Kernel(d_model=4, state_size=8, l_max=32, channels=1)
        before = kernel.C.detach().clone()
        kernel(100)
        assert torch.equal(kernel.C.detach(), before)

    def test_parameter_count_matches_the_published_configuration(self):
        """4 layers of 512 channels and state size 8 on 12 leads with 71 outputs."""
        model = S4ECG(input_channels=12, output_size=71, l_max=250)
        assert _trainable(model) == 2_183_239

    def test_only_c_d_and_the_linear_layers_are_trained(self):
        model = S4ECG(input_channels=1, output_size=4, d_model=8, l_max=64)
        frozen = {name.split(".")[-1] for name, _ in model.named_buffers()}
        assert frozen == {"w", "P", "B", "log_dt"}
        trained = {name.split(".")[-1] for name, _ in model.named_parameters()}
        assert {"C", "D"} <= trained and not trained & frozen

    def test_the_reference_length_kernel_is_finite(self):
        kernel = S4Kernel(d_model=8, state_size=8, l_max=5000, channels=2)
        with torch.no_grad():
            assert torch.isfinite(kernel(5000)).all()

    def test_recomputing_blocks_in_training_leaves_gradients_unchanged(self):
        torch.manual_seed(0)
        model = S4ECG(input_channels=3, output_size=2, d_model=16, l_max=64).train()
        x = torch.randn(4, 3, 64)
        gradients = []
        for recompute in (True, False):
            model.zero_grad()
            torch.manual_seed(1)
            if recompute:
                out = model(x)
            else:
                features = model.encoder(x)
                for index in range(len(model.layers)):
                    features = model._residual_block(features, index)
                out = model.classifier(features.mean(dim=-1))
            out.sum().backward()
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        assert torch.equal(gradients[0], gradients[1])
