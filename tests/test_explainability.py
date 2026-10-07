"""
Tests for deepecgkit.explainability module.
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from deepecgkit.explainability import (
    BEAT_WINDOW_MS,
    PQRST_WINDOWS_MS,
    Attribution,
    ECGExplainer,
    beat_aligned_attribution,
    detect_r_peaks,
    explain,
    fit_to_constraints,
    integrated_gradients,
    lead_importance,
    length_bounds,
    occlusion,
    plot_attribution,
    plot_beat_profile,
    plot_lead_importance,
    pqrst_enrichment,
    pqrst_fractions,
    region_duration_fractions,
    saliency,
)
from deepecgkit.models.factory import build_model
from deepecgkit.registry import get_model_names

SIGNAL_FIXTURES = {"afmodel": "sample_batch_ecg_30s"}

LENGTH_LIMITED_MODELS = ["transformer", "dualnet", "medformer"]

LENGTH_CEILINGS = {"transformer": 5000, "dualnet": 5000, "medformer": 5009}

BLIND_CUT = 1500
OCCLUSION_WINDOW = 100
OCCLUSION_STRIDE = 50
CHEAP_IG_STEPS = 2
IG_STEPS = 64
IG_COMPLETENESS_RTOL = 1e-3
IG_COMPLETENESS_ATOL = 1e-4
LEAD_SUM_ATOL = 1e-5
BATCH_INVARIANCE_ATOL = 1e-6
SAMPLING_RATE = 500.0
REGION_PURITY = 0.95
QRS_HALF_WIDTH_MS = 40


class HalfBlindModel(nn.Module):
    """A model that architecturally cannot see the tail of its input.

    The forward pass slices before the only layer, so every sample from ``cut``
    onward is absent from the graph. Its true attribution is not merely small,
    it is zero, which is what lets the masked-region tests assert exactness.
    """

    def __init__(self, cut: int = BLIND_CUT, leads: int = 1, output_size: int = 4):
        super().__init__()
        self.cut = cut
        self.classifier = nn.Linear(leads * cut, output_size)

    def forward(self, x):
        return self.classifier(x[..., : self.cut].flatten(1))


class LeadSelectorModel(nn.Module):
    """A model that reads one lead and drops every other before its only layer."""

    def __init__(self, lead: int = 3, leads: int = 12, length: int = 3000, output_size: int = 4):
        super().__init__()
        self.lead = lead
        self.input_channels = leads
        self.classifier = nn.Linear(length, output_size)

    def forward(self, x):
        return self.classifier(x[:, self.lead, :])


class RegionOnlyModel(nn.Module):
    """A model that sees only the samples a fixed mask keeps."""

    def __init__(self, mask: np.ndarray, output_size: int = 4):
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        self.classifier = nn.Linear(int(mask.shape[0]), output_size)

    def forward(self, x):
        return self.classifier((x * self.mask).squeeze(1))


def _synthetic_ecg(length=5000, sampling_rate=SAMPLING_RATE, bpm=60.0, seed=0):
    """Build a single-lead ECG with P, Q, R, S and T waves at known R positions."""
    time = np.arange(length) / sampling_rate
    trace = np.zeros(length)
    peaks = []

    def wave(centre, amplitude, width):
        return amplitude * np.exp(-((time - centre) ** 2) / (2 * width**2))

    interval = 60.0 / bpm
    index = 0
    while index * interval + 0.5 < length / sampling_rate:
        centre = index * interval + 0.5
        trace += wave(centre - 0.20, 0.12, 0.020)
        trace += wave(centre - 0.03, -0.10, 0.008)
        trace += wave(centre, 1.00, 0.008)
        trace += wave(centre + 0.03, -0.20, 0.008)
        trace += wave(centre + 0.22, 0.25, 0.040)
        peaks.append(round(centre * sampling_rate))
        index += 1

    generator = np.random.default_rng(seed)
    trace += 0.01 * generator.standard_normal(length)
    return trace, np.asarray(peaks)


@pytest.fixture(autouse=True)
def headless_matplotlib():
    """Render without a display and close every figure a test leaves open."""
    matplotlib.use("Agg", force=True)
    yield
    plt.close("all")


@pytest.fixture
def realistic_ecg():
    """A synthetic ECG with its true R-peak positions."""
    return _synthetic_ecg()


@pytest.fixture
def twelve_lead_ecg():
    """A batch of two 12-lead recordings built from the synthetic ECG."""
    trace, _ = _synthetic_ecg()
    generator = np.random.default_rng(7)
    leads = np.stack([trace * (1.0 + 0.08 * i) for i in range(12)])
    batch = np.stack([leads, leads + 0.01 * generator.standard_normal(leads.shape)])
    return torch.tensor(batch, dtype=torch.float32)


@pytest.fixture
def half_blind_batch():
    """Random input long enough to straddle the half-blind model's cut."""
    generator = torch.Generator().manual_seed(11)
    return torch.randn(2, 1, 3000, generator=generator)


class TestAttributionContract:
    """Every registered model produces attribution shaped like its input."""

    @pytest.mark.parametrize("model_name", get_model_names())
    @pytest.mark.parametrize("method", ["saliency", "integrated-gradients", "occlusion"])
    def test_attribution_preserves_the_input_shape(self, model_name, method, request):
        """Attribution is per lead and per timestep, never pooled."""
        signals = request.getfixturevalue(SIGNAL_FIXTURES.get(model_name, "sample_batch_ecg"))
        model = build_model(model_name, input_channels=1, output_size=4)
        explainer = ECGExplainer(
            model,
            model_name=model_name,
            method=method,
            device="cpu",
            batch_size=2,
            align_beats=False,
            ig_steps=CHEAP_IG_STEPS,
            occlusion_window=signals.shape[-1] // 3,
            occlusion_stride=signals.shape[-1] // 3,
        )

        result = explainer.explain(signals)

        assert result.values.shape == tuple(signals.shape)
        assert result.values.dtype == np.float32
        assert np.isfinite(result.values).all()

    @pytest.mark.parametrize("model_name", get_model_names())
    def test_lead_importance_is_a_distribution(self, model_name, request):
        """Each row of the lead distribution is non-negative and sums to one."""
        signals = request.getfixturevalue(SIGNAL_FIXTURES.get(model_name, "sample_batch_ecg"))
        model = build_model(model_name, input_channels=1, output_size=4)
        result = ECGExplainer(
            model, model_name=model_name, device="cpu", batch_size=2, align_beats=False
        ).explain(signals)

        assert result.lead_importance.shape == (signals.shape[0], signals.shape[1])
        assert (result.lead_importance >= 0).all()
        assert np.allclose(result.lead_importance.sum(axis=-1), 1.0, atol=LEAD_SUM_ATOL)

    def test_the_model_is_left_as_it_was_found(self, sample_batch_ecg):
        """Training mode is restored and no parameter gradient is populated."""
        model = build_model("kanres", input_channels=1, output_size=4)
        model.train()

        ECGExplainer(model, device="cpu", align_beats=False).explain(sample_batch_ecg)

        assert model.training
        assert all(parameter.grad is None for parameter in model.parameters())

    def test_it_runs_inside_a_caller_s_no_grad_block(self, sample_batch_ecg):
        """Attribution enables gradients itself rather than relying on the caller."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with torch.no_grad():
            result = ECGExplainer(model, device="cpu", align_beats=False).explain(sample_batch_ecg)

        assert np.abs(result.values).sum() > 0


class TestMaskedRegionIsNotAttributed:
    """A region the model cannot read must receive no attribution."""

    def test_saliency_is_exactly_zero_in_the_ignored_region(self, half_blind_batch):
        """Samples absent from the graph get a literal zero, not a small number."""
        values = saliency(HalfBlindModel(), half_blind_batch, target=0)

        assert torch.count_nonzero(values[..., BLIND_CUT:]) == 0
        assert values[..., :BLIND_CUT].abs().max() > 0

    def test_integrated_gradients_is_exactly_zero_in_the_ignored_region(self, half_blind_batch):
        """A mean of exact zeros scaled by a finite factor is still exactly zero."""
        values = integrated_gradients(
            HalfBlindModel(), half_blind_batch, target=0, steps=CHEAP_IG_STEPS
        )

        assert torch.count_nonzero(values[..., BLIND_CUT:]) == 0
        assert values[..., :BLIND_CUT].abs().max() > 0

    def test_occlusion_is_exactly_zero_beyond_one_window(self, half_blind_batch):
        """Past the window that straddles the cut, blanking changes nothing."""
        values = occlusion(
            HalfBlindModel(),
            half_blind_batch,
            target=0,
            window=OCCLUSION_WINDOW,
            stride=OCCLUSION_STRIDE,
        )

        assert torch.count_nonzero(values[..., BLIND_CUT + OCCLUSION_WINDOW :]) == 0
        assert values[..., :BLIND_CUT].abs().max() > 0

    def test_occlusion_bleeds_no_further_than_one_window(self, half_blind_batch):
        """The smear is bounded by the window width, and is documented, not hidden."""
        values = occlusion(
            HalfBlindModel(),
            half_blind_batch,
            target=0,
            window=OCCLUSION_WINDOW,
            stride=OCCLUSION_STRIDE,
        )
        beyond = values[..., BLIND_CUT:]
        nonzero = torch.nonzero(beyond)

        assert nonzero.numel() == 0 or int(nonzero[:, -1].max()) < OCCLUSION_WINDOW

    def test_changing_the_ignored_region_changes_no_attribution(self, half_blind_batch):
        """Attribution over the readable region ignores what the model cannot see."""
        model = HalfBlindModel()
        disturbed = half_blind_batch.clone()
        disturbed[..., BLIND_CUT:] += 25.0

        original = saliency(model, half_blind_batch, target=0)
        altered = saliency(model, disturbed, target=0)

        assert torch.equal(original[..., :BLIND_CUT], altered[..., :BLIND_CUT])


class TestIntegratedGradientsCompleteness:
    """Integrated gradients sums to the logit gap it is meant to explain."""

    def test_attributions_sum_to_the_logit_difference(self, sample_batch_ecg):
        """The completeness axiom holds on a registered model."""
        model = build_model("kanres", input_channels=1, output_size=4)
        model.eval()
        signals = sample_batch_ecg
        baseline = torch.zeros_like(signals)

        values = integrated_gradients(model, signals, target=0, steps=IG_STEPS)

        with torch.no_grad():
            gap = model(signals)[:, 0] - model(baseline)[:, 0]

        assert np.allclose(
            values.sum(dim=(1, 2)).numpy(),
            gap.numpy(),
            rtol=IG_COMPLETENESS_RTOL,
            atol=IG_COMPLETENESS_ATOL,
        )

    def test_an_input_equal_to_the_baseline_gets_no_attribution(self, sample_batch_ecg):
        """With nothing to explain, every attribution is exactly zero."""
        model = build_model("kanres", input_channels=1, output_size=4)
        baseline = torch.zeros_like(sample_batch_ecg)

        values = integrated_gradients(model, baseline, target=0, steps=CHEAP_IG_STEPS)

        assert torch.count_nonzero(values) == 0

    def test_more_steps_do_not_make_completeness_worse(self, sample_batch_ecg):
        """Refining the Riemann sum tightens the residual."""
        model = build_model("kanres", input_channels=1, output_size=4)
        model.eval()
        with torch.no_grad():
            gap = float(
                (model(sample_batch_ecg)[:, 0] - model(torch.zeros_like(sample_batch_ecg))[:, 0])
                .abs()
                .sum()
            )

        coarse = integrated_gradients(model, sample_batch_ecg, target=0, steps=4)
        fine = integrated_gradients(model, sample_batch_ecg, target=0, steps=64)
        coarse_error = abs(float(coarse.sum()) - gap)
        fine_error = abs(float(fine.sum()) - gap)

        assert fine_error <= max(coarse_error, 1e-5)

    def test_steps_must_be_positive(self, sample_batch_ecg):
        """A zero-step path is rejected rather than dividing by zero."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="steps must be at least 1"):
            integrated_gradients(model, sample_batch_ecg, target=0, steps=0)


class TestLeadImportance:
    """Lead importance is a distribution that finds the lead the model reads."""

    def test_the_only_readable_lead_takes_all_the_importance(self, twelve_lead_ecg):
        """A model reading one lead attributes nothing to the other eleven."""
        model = LeadSelectorModel(lead=3, length=twelve_lead_ecg.shape[-1])
        values = saliency(model, twelve_lead_ecg, target=0).numpy()

        importance = lead_importance(values)

        assert importance[:, 3] == pytest.approx(1.0, abs=LEAD_SUM_ATOL)
        assert importance[:, [i for i in range(12) if i != 3]] == pytest.approx(0.0, abs=1e-7)

    def test_a_single_lead_model_reports_one(self):
        """With one lead there is nowhere else for the evidence to be."""
        importance = lead_importance(np.random.default_rng(3).random((2, 1, 100)))

        assert importance == pytest.approx(1.0)

    def test_uniform_attribution_spreads_importance_evenly(self):
        """Equal attribution across twelve leads gives each a twelfth."""
        importance = lead_importance(np.ones((1, 12, 100), dtype=np.float32))

        assert importance[0] == pytest.approx(1.0 / 12.0)

    def test_all_zero_attribution_falls_back_to_uniform(self):
        """A model that reacted to nothing gives no reason to prefer any lead."""
        importance = lead_importance(np.zeros((2, 4, 50), dtype=np.float32))

        assert importance == pytest.approx(0.25)
        assert np.allclose(importance.sum(axis=-1), 1.0)

    def test_sign_does_not_change_importance(self):
        """Evidence against the target class is still evidence the lead was read."""
        values = np.random.default_rng(5).standard_normal((2, 6, 80))

        assert lead_importance(values) == pytest.approx(lead_importance(-values))

    def test_a_two_dimensional_input_is_rejected(self):
        """The batch axis is required so the contract cannot be read ambiguously."""
        with pytest.raises(ValueError, match="shape \\(batch, leads, time\\)"):
            lead_importance(np.zeros((12, 100)))


class TestRPeakDetection:
    """R-peaks are recovered at the positions they were planted."""

    def test_every_planted_peak_is_found_exactly(self, realistic_ecg):
        """Detection on clean synthetic ECG is sample-accurate."""
        trace, planted = realistic_ecg

        detected = detect_r_peaks(trace, SAMPLING_RATE)

        assert len(detected) == len(planted)
        assert np.abs(detected - planted).max() == 0

    def test_peaks_are_sorted_and_unique(self, realistic_ecg):
        """Indices come back strictly increasing."""
        detected = detect_r_peaks(realistic_ecg[0], SAMPLING_RATE)

        assert (np.diff(detected) > 0).all()

    def test_the_recovered_rate_matches_the_planted_rate(self, realistic_ecg):
        """A 60 bpm trace reads back as 60 bpm."""
        detected = detect_r_peaks(realistic_ecg[0], SAMPLING_RATE)
        rate = 60.0 / (float(np.median(np.diff(detected))) / SAMPLING_RATE)

        assert rate == pytest.approx(60.0, abs=1.0)

    def test_a_flat_signal_yields_no_peaks(self):
        """Nothing to detect returns an empty array rather than raising."""
        assert detect_r_peaks(np.zeros(5000), SAMPLING_RATE).size == 0

    def test_a_signal_shorter_than_the_window_yields_no_peaks(self):
        """Too little data to integrate over is reported as no beats."""
        assert detect_r_peaks(np.zeros(10), SAMPLING_RATE).size == 0

    def test_the_band_follows_the_sampling_rate(self, realistic_ecg):
        """Detection parameters are expressed in seconds, not samples."""
        trace, planted = realistic_ecg
        decimated = trace[::2]

        detected = detect_r_peaks(decimated, SAMPLING_RATE / 2.0)

        assert len(detected) == len(planted)

    def test_a_two_dimensional_signal_is_rejected(self):
        """Detection runs on one lead, so the lead must already be chosen."""
        with pytest.raises(ValueError, match="one-dimensional"):
            detect_r_peaks(np.zeros((2, 5000)), SAMPLING_RATE)

    def test_a_band_above_nyquist_is_rejected(self, realistic_ecg):
        """An impossible band is named rather than passed to the filter."""
        with pytest.raises(ValueError, match="Nyquist"):
            detect_r_peaks(realistic_ecg[0], 100.0, bandpass=(60.0, 90.0))


class TestBeatAlignedAttribution:
    """Attribution is averaged over beats and split by cardiac region."""

    @pytest.mark.parametrize("region", ["P", "QRS", "T"])
    def test_a_region_only_model_is_reported_as_that_region(self, realistic_ecg, region):
        """The P, QRS and T windows each recover a model built to see just them."""
        trace, planted = realistic_ecg
        start_ms, end_ms = PQRST_WINDOWS_MS[region]
        mask = np.zeros(trace.size, dtype=np.float32)
        for peak in planted:
            lo = max(0, peak + int(start_ms / 1000 * SAMPLING_RATE))
            mask[lo : peak + int(end_ms / 1000 * SAMPLING_RATE)] = 1.0

        model = RegionOnlyModel(mask)
        signals = torch.tensor(trace, dtype=torch.float32).view(1, 1, -1)
        values = saliency(model, signals, target=0).numpy()[0]

        peaks = detect_r_peaks(trace, SAMPLING_RATE)
        profile, times, count = beat_aligned_attribution(values, peaks, SAMPLING_RATE)
        fractions = pqrst_fractions(profile, times)

        assert count == len(planted)
        assert fractions[region] > REGION_PURITY
        for other in PQRST_WINDOWS_MS:
            if other != region:
                assert fractions[other] < 1.0 - REGION_PURITY

    def test_the_profile_spans_the_requested_window(self, realistic_ecg):
        """The window is placed in milliseconds relative to the R-peak."""
        trace, planted = realistic_ecg
        values = np.ones((3, trace.size), dtype=np.float32)

        profile, times, _ = beat_aligned_attribution(values, planted, SAMPLING_RATE)

        expected = round(800 / 1000 * SAMPLING_RATE)
        assert profile.shape == (3, expected)
        assert times[0] == pytest.approx(BEAT_WINDOW_MS[0])
        assert times.size == expected

    def test_the_profile_is_the_mean_of_the_windows(self):
        """Averaging is a plain mean over the surviving beats."""
        values = np.arange(2 * 4000, dtype=np.float64).reshape(2, 4000)
        peaks = np.array([1000, 2000, 3000])

        profile, _, count = beat_aligned_attribution(values, peaks, SAMPLING_RATE)

        pre = round(300 / 1000 * SAMPLING_RATE)
        post = round(500 / 1000 * SAMPLING_RATE)
        expected = np.mean([values[:, p - pre : p + post] for p in peaks], axis=0)
        assert count == 3
        assert np.allclose(profile, expected)

    def test_beats_running_off_the_end_are_dropped(self):
        """A truncated beat would bias the profile toward the recording's edge."""
        values = np.ones((1, 2000), dtype=np.float32)

        _, _, count = beat_aligned_attribution(values, np.array([10, 1000, 1990]), SAMPLING_RATE)

        assert count == 1

    def test_no_beats_gives_a_zero_profile(self):
        """An empty peak list is reported, not guessed at."""
        profile, _, count = beat_aligned_attribution(
            np.ones((2, 3000), dtype=np.float32), np.array([]), SAMPLING_RATE
        )

        assert count == 0
        assert not profile.any()

    def test_fractions_leave_the_baseline_visible(self, realistic_ecg):
        """Shares are taken against the whole beat, so they need not sum to one."""
        trace, planted = realistic_ecg
        values = np.ones((1, trace.size), dtype=np.float32)
        profile, times, _ = beat_aligned_attribution(values, planted, SAMPLING_RATE)

        fractions = pqrst_fractions(profile, times)

        assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-6)
        assert fractions["baseline"] > 0

    def test_uniform_attribution_has_enrichment_of_one(self, realistic_ecg):
        """Chance is the reference the P-wave claim is measured against."""
        trace, planted = realistic_ecg
        values = np.ones((1, trace.size), dtype=np.float32)
        profile, times, _ = beat_aligned_attribution(values, planted, SAMPLING_RATE)

        enrichment = pqrst_enrichment(pqrst_fractions(profile, times), times)

        for region in PQRST_WINDOWS_MS:
            assert enrichment[region] == pytest.approx(1.0, abs=0.02)

    def test_duration_shares_sum_to_one(self, realistic_ecg):
        """The regions plus the baseline tile the beat window exactly."""
        trace, planted = realistic_ecg
        _, times, _ = beat_aligned_attribution(
            np.ones((1, trace.size), dtype=np.float32), planted, SAMPLING_RATE
        )

        shares = region_duration_fractions(times)

        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-9)

    def test_a_mismatched_time_axis_is_rejected(self):
        """A profile and its axis must describe the same window."""
        with pytest.raises(ValueError, match="does not match the profile"):
            pqrst_fractions(np.ones((2, 100)), np.arange(50))


class TestConstraints:
    """Signals outside a model's bounds are cropped or refused, never mangled."""

    @pytest.mark.parametrize("model_name", LENGTH_LIMITED_MODELS)
    def test_an_over_long_signal_is_centre_cropped(self, model_name, sample_batch_ecg_30s):
        """The ceiling produces a warning and a centred crop, not a shape error."""
        model = build_model(model_name, input_channels=1, output_size=4)
        ceiling = LENGTH_CEILINGS[model_name]

        with pytest.warns(RuntimeWarning, match="centre-cropping"):
            cropped, offset = fit_to_constraints(sample_batch_ecg_30s, model, model_name)

        assert cropped.shape[-1] == ceiling
        assert offset == (sample_batch_ecg_30s.shape[-1] - ceiling) // 2
        assert cropped.shape[:2] == sample_batch_ecg_30s.shape[:2]

    @pytest.mark.parametrize("model_name", LENGTH_LIMITED_MODELS)
    def test_a_short_signal_is_left_alone(self, model_name, sample_batch_ecg):
        """Nothing is cropped and nothing is warned about below the ceiling."""
        model = build_model(model_name, input_channels=1, output_size=4)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            kept, offset = fit_to_constraints(sample_batch_ecg, model, model_name)

        assert offset == 0
        assert kept.shape == sample_batch_ecg.shape

    def test_afmodel_refuses_a_signal_that_is_too_short(self, sample_batch_ecg):
        """Cropping cannot lengthen a recording, so this raises rather than pads."""
        model = build_model("afmodel", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="needs at least 7971"):
            fit_to_constraints(sample_batch_ecg, model, "afmodel")

    def test_the_declared_minimum_beats_the_table(self):
        """AFModel's floor depends on recording_length, which no table can know."""
        model = build_model("afmodel", input_channels=1, output_size=4, recording_length=6)

        assert length_bounds(model, "afmodel") == (1393, None)

    def test_a_lead_mismatch_is_named(self):
        """The lead count is checked before the forward pass, for every model."""
        model = build_model("deep-res-cnn", input_channels=12, output_size=4)

        with pytest.raises(ValueError, match="built for 12 leads but received 13"):
            fit_to_constraints(torch.randn(1, 13, 1000), model, "deep-res-cnn")

    def test_a_two_dimensional_batch_is_rejected(self):
        """A missing lead axis is reported clearly."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="shape \\(batch, leads, time\\)"):
            fit_to_constraints(torch.randn(12, 1000), model, "kanres")

    def test_an_unbounded_model_reports_no_ceiling(self):
        """Seventeen architectures have no maximum, and say so."""
        model = build_model("kanres", input_channels=1, output_size=4)

        assert length_bounds(model, "kanres")[1] is None

    def test_the_explainer_records_the_crop_offset(self, sample_batch_ecg_30s):
        """A cropped attribution can still be placed on the original time axis.

        Medformer stands for the length-limited models here: it attends over
        patches, where the transformer attends over all 5000 samples and takes
        well over a gigabyte to attribute.
        """
        model = build_model("medformer", input_channels=1, output_size=4)

        with pytest.warns(RuntimeWarning, match="centre-cropping"):
            result = ECGExplainer(
                model, model_name="medformer", device="cpu", batch_size=2, align_beats=False
            ).explain(sample_batch_ecg_30s)

        assert result.values.shape[-1] == LENGTH_CEILINGS["medformer"]
        assert result.crop_offset == (15000 - LENGTH_CEILINGS["medformer"]) // 2


class TestExplainerBehaviour:
    """Batching and repetition do not change the answer."""

    @pytest.mark.parametrize("batch_size", [1, 3, 4])
    def test_batch_size_does_not_change_results(self, batch_size, sample_batch_ecg):
        """Rows are attributed independently of how they were grouped."""
        model = build_model("kanres", input_channels=1, output_size=4)
        reference = ECGExplainer(model, device="cpu", batch_size=4, align_beats=False).explain(
            sample_batch_ecg
        )
        chunked = ECGExplainer(
            model, device="cpu", batch_size=batch_size, align_beats=False
        ).explain(sample_batch_ecg)

        assert np.allclose(reference.values, chunked.values, atol=BATCH_INVARIANCE_ATOL)

    def test_repeated_calls_agree_exactly(self, sample_batch_ecg):
        """Attribution is deterministic in evaluation mode."""
        model = build_model("kanres", input_channels=1, output_size=4)
        explainer = ECGExplainer(model, device="cpu", align_beats=False)

        assert np.array_equal(
            explainer.explain(sample_batch_ecg).values,
            explainer.explain(sample_batch_ecg).values,
        )

    def test_the_default_target_is_the_predicted_class(self, sample_batch_ecg):
        """Attribution explains what the model actually said."""
        model = build_model("kanres", input_channels=1, output_size=4)
        model.eval()
        with torch.no_grad():
            predicted = model(sample_batch_ecg).argmax(dim=-1).numpy()

        result = ECGExplainer(model, device="cpu", align_beats=False).explain(sample_batch_ecg)

        assert np.array_equal(result.target, predicted)

    def test_an_explicit_target_overrides_the_prediction(self, sample_batch_ecg):
        """A named class is explained even when the model prefers another."""
        model = build_model("kanres", input_channels=1, output_size=4)

        result = ECGExplainer(model, device="cpu", align_beats=False).explain(
            sample_batch_ecg, target=2
        )

        assert (result.target == 2).all()

    def test_a_per_sample_target_is_honoured(self, sample_batch_ecg):
        """One index per row selects a different class for each recording."""
        model = build_model("kanres", input_channels=1, output_size=4)
        targets = np.array([0, 1, 2, 3])

        result = ECGExplainer(model, device="cpu", batch_size=2, align_beats=False).explain(
            sample_batch_ecg, target=targets
        )

        assert np.array_equal(result.target, targets)

    def test_a_mismatched_target_length_is_rejected(self, sample_batch_ecg):
        """A target list that does not match the batch is an error, not a guess."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="indices but the batch has"):
            ECGExplainer(model, device="cpu", batch_size=4, align_beats=False).explain(
                sample_batch_ecg, target=np.array([0, 1])
            )

    def test_max_samples_stops_early(self, sample_batch_ecg):
        """The cap bounds the work, and is reflected in the result."""
        model = build_model("kanres", input_channels=1, output_size=4)

        result = ECGExplainer(
            model, device="cpu", batch_size=1, max_samples=2, align_beats=False
        ).explain(sample_batch_ecg)

        assert len(result) == 2

    def test_an_unknown_method_is_rejected(self, sample_batch_ecg):
        """The available methods are named rather than guessed at."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="Unknown method"):
            ECGExplainer(model, method="gradcam", device="cpu").explain(sample_batch_ecg)

    def test_an_unknown_split_is_rejected(self, sample_batch_ecg):
        """Split names are validated before any work starts."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="Unknown split"):
            ECGExplainer(model, device="cpu").explain(sample_batch_ecg, split="holdout")

    def test_an_unsupported_input_type_is_rejected(self):
        """The accepted input kinds are listed in the error."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(TypeError, match="Unsupported input type"):
            ECGExplainer(model, device="cpu").explain(42)

    def test_a_numpy_array_is_accepted(self):
        """Arrays are normalised to (batch, leads, time) like everywhere else."""
        model = build_model("kanres", input_channels=1, output_size=4)

        result = ECGExplainer(model, device="cpu", align_beats=False).explain(
            np.random.default_rng(1).standard_normal((2, 1, 3000)).astype(np.float32)
        )

        assert result.values.shape == (2, 1, 3000)


class TestExplainFunction:
    """The one-call helper is the class with defaults."""

    def test_it_matches_the_class(self, sample_batch_ecg):
        """Both routes run the same computation."""
        model = build_model("kanres", input_channels=1, output_size=4)

        direct = ECGExplainer(model, device="cpu", align_beats=False).explain(sample_batch_ecg)
        composed = explain(sample_batch_ecg, model=model, device="cpu", align_beats=False)

        assert np.array_equal(direct.values, composed.values)

    def test_it_requires_exactly_one_weight_source(self, sample_batch_ecg):
        """Neither zero nor two sources is enough to identify the model."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="exactly one"):
            explain(sample_batch_ecg)

        with pytest.raises(ValueError, match="exactly one"):
            explain(sample_batch_ecg, model=model, checkpoint="best.ckpt")

    def test_bare_weights_require_a_model_name(self, sample_batch_ecg, temp_dir):
        """Weights alone do not say which architecture they belong to."""
        path = temp_dir / "bare.pt"
        torch.save(build_model("kanres", input_channels=1, output_size=4).state_dict(), str(path))
        with pytest.raises(ValueError, match="does not record which architecture"):
            explain(sample_batch_ecg, checkpoint=str(path))

    def test_it_loads_from_a_checkpoint(self, sample_batch_ecg, temp_dir):
        """A saved model is attributed the same as a live one."""
        model = build_model("kanres", input_channels=1, output_size=4)
        path = temp_dir / "kanres.ckpt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {"name": "kanres", "input_channels": 1, "output_size": 4},
                "train_config": {},
            },
            str(path),
        )

        result = explain(
            sample_batch_ecg,
            checkpoint=str(path),
            model_name="kanres",
            device="cpu",
            align_beats=False,
        )

        assert result.model_name == "kanres"
        assert result.values.shape == tuple(sample_batch_ecg.shape)


class TestRawMethods:
    """The method functions agree with the explainer that wraps them."""

    def test_saliency_matches_the_explainer(self, sample_batch_ecg):
        """The explainer adds bookkeeping, not a different computation."""
        model = build_model("kanres", input_channels=1, output_size=4)
        model.eval()

        direct = saliency(model, sample_batch_ecg, target=1).numpy()
        wrapped = ECGExplainer(
            model, method="saliency", device="cpu", batch_size=4, align_beats=False
        ).explain(sample_batch_ecg, target=1)

        assert np.allclose(direct, wrapped.values, atol=BATCH_INVARIANCE_ATOL)

    def test_saliency_returns_magnitude_by_default(self, sample_batch_ecg):
        """The default answers how much, not which way."""
        model = build_model("kanres", input_channels=1, output_size=4)

        assert (saliency(model, sample_batch_ecg, target=0) >= 0).all()

    def test_saliency_can_return_a_signed_gradient(self, sample_batch_ecg):
        """The sign is available for callers who want direction."""
        model = build_model("kanres", input_channels=1, output_size=4)

        signed = saliency(model, sample_batch_ecg, target=0, absolute=False)

        assert torch.allclose(signed.abs(), saliency(model, sample_batch_ecg, target=0))

    def test_occlusion_covers_every_timestep(self, sample_batch_ecg):
        """A stride that does not divide the length still reaches the tail."""
        model = build_model("kanres", input_channels=1, output_size=4)

        values = occlusion(model, sample_batch_ecg, target=0, window=700, stride=700)

        assert values.shape == sample_batch_ecg.shape

    def test_occlusion_rejects_a_non_positive_window(self, sample_batch_ecg):
        """A zero-width window would attribute nothing anywhere."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="window must be at least 1"):
            occlusion(model, sample_batch_ecg, target=0, window=0)

    def test_occlusion_rejects_a_non_positive_stride(self, sample_batch_ecg):
        """A zero stride would never advance."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="stride must be at least 1"):
            occlusion(model, sample_batch_ecg, target=0, stride=0)

    def test_a_mismatched_baseline_is_rejected(self, sample_batch_ecg):
        """The path has to start somewhere the model can consume."""
        model = build_model("kanres", input_channels=1, output_size=4)

        with pytest.raises(ValueError, match="does not match the signals"):
            integrated_gradients(
                model, sample_batch_ecg, target=0, steps=2, baseline=torch.zeros(1, 1, 10)
            )


class TestAttributionResult:
    """The result object describes itself and round-trips to disk."""

    def _result(self):
        values = np.random.default_rng(2).standard_normal((2, 12, 400)).astype(np.float32)
        return Attribution(
            values=values,
            signal=values.copy(),
            lead_importance=lead_importance(values),
            target=np.array([0, 1]),
            method="saliency",
            model_name="kanres",
            sampling_rate=SAMPLING_RATE,
            lead_names=[f"L{i}" for i in range(12)],
        )

    def test_it_reports_its_own_dimensions(self):
        """Length and lead count come from the arrays, not from a field."""
        result = self._result()

        assert len(result) == 2
        assert result.num_leads == 12

    def test_the_mean_lead_distribution_still_sums_to_one(self):
        """An average of distributions is itself a distribution."""
        assert self._result().mean_lead_importance().sum() == pytest.approx(1.0, abs=LEAD_SUM_ATOL)

    def test_the_summary_is_json_serialisable(self):
        """summary.json is written straight from this dictionary."""
        import json

        summary = self._result().summary()

        assert json.loads(json.dumps(summary))["model_name"] == "kanres"
        assert set(summary["lead_importance"]) == {f"L{i}" for i in range(12)}

    def test_it_saves_and_reloads_without_pickle(self, temp_dir):
        """Every stored array is a plain array, so npz needs no pickle."""
        result = self._result()
        path = temp_dir / "attribution.npz"

        result.save(path)
        loaded = np.load(str(path), allow_pickle=False)

        assert np.array_equal(loaded["values"], result.values)
        assert np.array_equal(loaded["signal"], result.signal)
        assert str(loaded["model_name"]) == "kanres"

    def test_an_unsupported_suffix_is_rejected(self, temp_dir):
        """Flat formats would lose the axis that makes attribution useful."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            self._result().save(temp_dir / "attribution.csv")


class TestVisualisation:
    """Every plot writes a file and closes its figure."""

    def _result_with_beats(self, realistic_ecg):
        trace, planted = realistic_ecg
        signal = np.stack([trace * (1 + 0.1 * i) for i in range(3)]).astype(np.float32)
        values = np.abs(signal)
        profile, times, _ = beat_aligned_attribution(values, planted, SAMPLING_RATE)
        return signal, values, profile, times

    def test_plot_attribution_writes_a_file(self, realistic_ecg, temp_dir):
        """The overlay renders headlessly and lands on disk."""
        signal, values, _, _ = self._result_with_beats(realistic_ecg)
        path = temp_dir / "signal.png"

        plot_attribution(signal, values, sampling_rate=SAMPLING_RATE, save_path=str(path))

        assert path.stat().st_size > 0
        assert plt.get_fignums() == []

    def test_plot_attribution_handles_a_single_lead(self, realistic_ecg, temp_dir):
        """The one-lead subplot case is the known trap in the existing plots."""
        signal, values, _, _ = self._result_with_beats(realistic_ecg)
        path = temp_dir / "one.png"

        plot_attribution(signal[:1], values[:1], sampling_rate=SAMPLING_RATE, save_path=str(path))

        assert path.stat().st_size > 0

    def test_plot_attribution_rejects_mismatched_shapes(self, realistic_ecg):
        """Signal and attribution describe the same samples or neither is drawn."""
        signal, values, _, _ = self._result_with_beats(realistic_ecg)

        with pytest.raises(ValueError, match="must match"):
            plot_attribution(signal, values[:, :100])

    def test_plot_attribution_rejects_a_one_dimensional_signal(self):
        """The lead axis is required, matching the rest of the package."""
        with pytest.raises(ValueError, match="shape \\(leads, time\\)"):
            plot_attribution(np.zeros(100), np.zeros(100))

    def test_plot_lead_importance_writes_a_file(self, temp_dir):
        """The bar chart renders from a bare distribution."""
        path = temp_dir / "leads.png"

        plot_lead_importance(np.full(12, 1 / 12), save_path=str(path))

        assert path.stat().st_size > 0
        assert plt.get_fignums() == []

    def test_plot_lead_importance_averages_a_batch(self, temp_dir):
        """A per-sample distribution is reduced before drawing."""
        path = temp_dir / "leads_batch.png"

        plot_lead_importance(np.full((4, 12), 1 / 12), save_path=str(path))

        assert path.stat().st_size > 0

    def test_plot_beat_profile_writes_a_file(self, realistic_ecg, temp_dir):
        """The beat profile renders with its regions shaded."""
        _, _, profile, times = self._result_with_beats(realistic_ecg)
        path = temp_dir / "beat.png"

        plot_beat_profile(profile, times, save_path=str(path))

        assert path.stat().st_size > 0
        assert plt.get_fignums() == []

    def test_plot_beat_profile_rejects_a_mismatched_axis(self, realistic_ecg):
        """A profile and its time axis must agree on the window length."""
        _, _, profile, _ = self._result_with_beats(realistic_ecg)

        with pytest.raises(ValueError, match="does not match a time axis"):
            plot_beat_profile(profile, np.arange(10))
