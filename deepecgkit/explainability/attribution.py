"""Attributing an ECG prediction back to the raw signal.

Every registered architecture consumes a (batch, leads, time) tensor, so the
honest place to ask what a model used is that same tensor. The three methods
here all return attribution in the input's own shape, per lead and per
timestep, which is what makes the lead and beat summaries downstream possible.

Nothing here reduces to a picture. A heatmap over five thousand samples and
twelve leads is not a claim anyone can check; a normalised distribution over
leads, and a share of attribution falling in the P-wave region, are.
"""

import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

from deepecgkit.datasets.io import as_model_input, read_ecg_recording
from deepecgkit.datasets.modules import ECGDataModule
from deepecgkit.datasets.ptbxl import PTBXLDataset
from deepecgkit.explainability.beats import (
    BEAT_WINDOW_MS,
    PQRST_WINDOWS_MS,
    beat_aligned_attribution,
    detect_r_peaks,
    pqrst_enrichment,
    pqrst_fractions,
)
from deepecgkit.explainability.constraints import fit_to_constraints, resolve_model_name
from deepecgkit.features import _resolve_device
from deepecgkit.models.factory import build_model_from_checkpoint, infer_input_channels
from deepecgkit.models.pretrained import from_pretrained as _load_pretrained
from deepecgkit.models.pretrained import model_name_for_weights

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = [
    "METHODS",
    "Attribution",
    "ECGExplainer",
    "explain",
    "integrated_gradients",
    "lead_importance",
    "occlusion",
    "saliency",
]

METHODS: Tuple[str, ...] = ("saliency", "integrated-gradients", "occlusion")

SPLITS: Tuple[str, ...] = ("train", "val", "test", "all")

ExplainInput = Union[np.ndarray, torch.Tensor, str, Path, Dataset, DataLoader, ECGDataModule]

TargetSpec = Union[None, int, str, np.ndarray, torch.Tensor]


@contextmanager
def _gradient_ready(model: nn.Module, device: torch.device) -> Iterator[None]:
    """Put a model into a state where input gradients can be taken safely.

    Feature extraction can wrap its whole loop in ``torch.no_grad()``.
    Attribution cannot, because two of the three methods need the graph that
    suppresses. What replaces it is narrower and has to be undone afterwards.

    Evaluation mode is required rather than merely preferred: dropout left
    active would make each integrated-gradients step sample a different
    function, and three architectures raise on a batch of one in training mode
    because their heads normalise over the batch.

    cuDNN is disabled for the four recurrent architectures on CUDA, where its
    fused kernel refuses to run a backward pass outside training mode. Calling
    ``train()`` would satisfy cuDNN and reintroduce dropout, so the fused path
    is turned off instead and the slower generic one is used.

    Args:
        model: The model attribution will run against
        device: The device the attribution will run on

    Yields:
        None, with the model prepared and its previous state restored on exit.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    needs_generic_rnn = device.type == "cuda" and any(
        isinstance(module, nn.RNNBase) for module in model.modules()
    )

    try:
        if needs_generic_rnn:
            with torch.backends.cudnn.flags(enabled=False):
                yield
        else:
            yield
    finally:
        if was_training:
            model.train()


def _forward(model: nn.Module, signals: torch.Tensor) -> torch.Tensor:
    """Run a model and return plain logits.

    The call is positional because AFModel names its forward argument ``data``
    rather than ``x``, so a keyword call fails on exactly one of the registered
    architectures.

    Args:
        model: Any registered model
        signals: Batch of shape (batch, leads, time)

    Returns:
        Logits of shape (batch, output_size).
    """
    outputs = model(signals)
    return outputs.logits if hasattr(outputs, "logits") else outputs


def _resolve_target(logits: torch.Tensor, target: TargetSpec) -> torch.Tensor:
    """Turn a target specification into one class index per sample.

    Args:
        logits: Model output of shape (batch, output_size)
        target: None for the model's own prediction, an integer for one class
            across the batch, or a per-sample sequence of indices

    Returns:
        Long tensor of shape (batch,) on the logits' device.

    Raises:
        ValueError: If a per-sample sequence does not match the batch size.
    """
    batch = logits.shape[0]
    if target is None:
        return logits.argmax(dim=-1).detach()

    if isinstance(target, int):
        return torch.full((batch,), target, dtype=torch.long, device=logits.device)

    indices = torch.as_tensor(target, device=logits.device).reshape(-1).long()
    if indices.numel() == 1:
        return indices.expand(batch).clone()
    if indices.numel() != batch:
        raise ValueError(
            f"target holds {indices.numel()} indices but the batch has {batch} samples. "
            f"Pass one index, or one per sample."
        )
    return indices


def _target_scores(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Select the logit of the target class for each sample.

    Args:
        logits: Model output of shape (batch, output_size)
        index: Long tensor of shape (batch,)

    Returns:
        Tensor of shape (batch,).
    """
    return logits.gather(1, index.view(-1, 1)).squeeze(1)


def saliency(
    model: nn.Module,
    signals: torch.Tensor,
    target: TargetSpec = None,
    absolute: bool = True,
) -> torch.Tensor:
    """Attribute by the gradient of the target logit with respect to the input.

    The cheapest of the three methods, at one forward and one backward pass, and
    the noisiest: the gradient measures sensitivity at a single point, so it
    reacts to sharp local structure such as the QRS complex whether or not the
    model's decision depends on it.

    Args:
        model: Any registered model, already on the right device
        signals: Batch of shape (batch, leads, time)
        target: None for the model's own prediction, an integer, or one index
            per sample
        absolute: Return gradient magnitude rather than signed gradient

    Returns:
        Attribution of shape (batch, leads, time), matching the input.
    """
    inputs = signals.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        logits = _forward(model, inputs)
        index = _resolve_target(logits, target)
        gradients = torch.autograd.grad(_target_scores(logits, index).sum(), inputs)[0]

    return gradients.abs() if absolute else gradients


def integrated_gradients(
    model: nn.Module,
    signals: torch.Tensor,
    target: TargetSpec = None,
    steps: int = 64,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attribute by integrating gradients along a path from a baseline.

    Gradients are sampled at step midpoints, which makes the Riemann sum
    symmetric and lets the result satisfy the completeness axiom closely: the
    attribution sums to the difference between the model's score on the signal
    and on the baseline. Attribution is signed, because that identity is the
    method's value and taking magnitudes would destroy it.

    The default baseline of zeros is the sensible one for z-scored ECG, where
    zero is the per-lead mean, and a poor one for unnormalised millivolts.

    Args:
        model: Any registered model, already on the right device
        signals: Batch of shape (batch, leads, time)
        target: None for the model's own prediction, an integer, or one index
            per sample
        steps: Number of points along the path
        baseline: Reference input of the same shape, defaulting to zeros

    Returns:
        Signed attribution of shape (batch, leads, time), matching the input.

    Raises:
        ValueError: If steps is not positive, or the baseline shape differs.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}.")

    reference = torch.zeros_like(signals) if baseline is None else baseline.to(signals.device)
    if reference.shape != signals.shape:
        raise ValueError(
            f"baseline of shape {tuple(reference.shape)} does not match the signals' shape "
            f"{tuple(signals.shape)}."
        )

    with torch.no_grad():
        index = _resolve_target(_forward(model, signals), target)

    difference = signals - reference
    total = torch.zeros_like(signals)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (reference + alpha * difference).detach().requires_grad_(True)
        with torch.enable_grad():
            scores = _target_scores(_forward(model, point), index)
            total = total + torch.autograd.grad(scores.sum(), point)[0]

    return difference * total / steps


def occlusion(
    model: nn.Module,
    signals: torch.Tensor,
    target: TargetSpec = None,
    window: int = 100,
    stride: int = 50,
    baseline: float = 0.0,
) -> torch.Tensor:
    """Attribute by how much the score drops when a span is blanked out.

    One lead-and-span at a time is replaced by the baseline and the drop in the
    target logit is spread over that span, averaging where windows overlap. The
    result is signed: a positive value means blanking the span lowered the
    score, so the span supported the prediction.

    Attribution bleeds one window width past any sharp boundary in the model's
    receptive field, because a window straddling the boundary perturbs samples
    on both sides and the single resulting score change is smeared across all of
    it. Narrower windows sharpen the result and cost proportionally more
    forward passes.

    Args:
        model: Any registered model, already on the right device
        signals: Batch of shape (batch, leads, time)
        target: None for the model's own prediction, an integer, or one index
            per sample
        window: Width in samples of the occluded span
        stride: Step in samples between successive spans
        baseline: Value written into the occluded span

    Returns:
        Signed attribution of shape (batch, leads, time), matching the input.

    Raises:
        ValueError: If window or stride is not positive.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1 sample, got {window}.")
    if stride < 1:
        raise ValueError(f"stride must be at least 1 sample, got {stride}.")

    _, leads, length = signals.shape
    span = min(window, length)

    starts = list(range(0, max(length - span, 0) + 1, stride))
    if starts[-1] + span < length:
        starts.append(length - span)

    total = torch.zeros_like(signals)
    counts = torch.zeros_like(signals)

    with torch.no_grad():
        logits = _forward(model, signals)
        index = _resolve_target(logits, target)
        reference = _target_scores(logits, index)

        for lead in range(leads):
            for start in starts:
                stop = start + span
                perturbed = signals.clone()
                perturbed[:, lead, start:stop] = baseline
                scores = _target_scores(_forward(model, perturbed), index)
                total[:, lead, start:stop] += (reference - scores).view(-1, 1)
                counts[:, lead, start:stop] += 1.0

    return total / counts.clamp(min=1.0)


_METHODS: Dict[str, Callable[..., torch.Tensor]] = {
    "saliency": saliency,
    "integrated-gradients": integrated_gradients,
    "occlusion": occlusion,
}


def lead_importance(values: np.ndarray) -> np.ndarray:
    """Reduce per-timestep attribution to a distribution over leads.

    Each lead is scored by its mean absolute attribution, then the scores are
    normalised to sum to one. Magnitude rather than sign, because the question
    is how much a lead mattered rather than which way it pushed; mean rather
    than sum, so the scores stay comparable if leads ever carry different
    lengths.

    A single-lead model yields ``[1.0]``. That is correct rather than
    degenerate: with one lead there is nowhere else for the evidence to be.

    Args:
        values: Attribution of shape (batch, leads, time)

    Returns:
        Array of shape (batch, leads) whose rows sum to one. A row whose
        attribution is entirely zero becomes uniform, since a model that reacted
        to nothing gives no reason to prefer any lead.

    Raises:
        ValueError: If the attribution is not three-dimensional.
    """
    attribution = np.asarray(values, dtype=np.float64)
    if attribution.ndim != 3:
        raise ValueError(
            f"lead_importance expects attribution of shape (batch, leads, time), got shape "
            f"{attribution.shape}."
        )

    magnitude = np.abs(attribution).mean(axis=-1)
    total = magnitude.sum(axis=-1, keepdims=True)
    uniform = np.full_like(magnitude, 1.0 / magnitude.shape[-1])
    return np.where(total > 0.0, magnitude / np.where(total > 0.0, total, 1.0), uniform).astype(
        np.float32
    )


def default_lead_names(leads: int) -> List[str]:
    """Name the leads, using the standard twelve-lead order where it applies.

    Args:
        leads: Number of leads

    Returns:
        A list of names, taken from PTB-XL's ordering for twelve leads and
        generated otherwise.
    """
    if leads == len(PTBXLDataset.LEADS):
        return list(PTBXLDataset.LEADS)
    return [f"Lead {i + 1}" for i in range(leads)]


@dataclass
class Attribution:
    """Attribution for a batch of recordings, with its ECG-native summaries.

    Attributes:
        values: Per-lead per-timestep attribution of shape (batch, leads, time)
        signal: The signal the model actually saw, of the same shape, kept so
            attribution and trace can never fall out of alignment
        lead_importance: Distribution over leads of shape (batch, leads), rows
            summing to one
        target: Class index explained for each sample, of shape (batch,)
        method: Which of METHODS produced the values
        model_name: Registry name of the model, for provenance
        sampling_rate: Samples per second the time axis assumes
        crop_offset: Index in the original recording where these values begin,
            non-zero when the signal was centre-cropped to fit the model
        lead_names: Display names aligned with the lead axis
        beat_profile: Mean attribution over R-peak-aligned beats, of shape
            (leads, window), or None when beat alignment was not run
        beat_times_ms: Time axis of the beat window in milliseconds relative to
            the R-peak, or None
        pqrst_fractions: Share of beat-profile mass in each cardiac region, plus
            the baseline remainder, or None
        pqrst_enrichment: Each region's share divided by its share of the beat
            window's duration, so one means chance, or None
        beat_count: Number of beats averaged into the profile
    """

    values: np.ndarray
    signal: np.ndarray
    lead_importance: np.ndarray
    target: np.ndarray
    method: str
    model_name: Optional[str] = None
    sampling_rate: float = 500.0
    crop_offset: int = 0
    lead_names: Optional[List[str]] = None
    beat_profile: Optional[np.ndarray] = None
    beat_times_ms: Optional[np.ndarray] = None
    pqrst_fractions: Optional[Dict[str, float]] = field(default=None)
    pqrst_enrichment: Optional[Dict[str, float]] = field(default=None)
    beat_count: int = 0

    @property
    def num_leads(self) -> int:
        """Number of leads the attribution covers."""
        return int(self.values.shape[1])

    def __len__(self) -> int:
        """Number of recordings explained."""
        return int(self.values.shape[0])

    def mean_lead_importance(self) -> np.ndarray:
        """Average the per-sample lead distribution over the batch.

        Returns:
            Array of shape (leads,) summing to one.
        """
        return self.lead_importance.mean(axis=0).astype(np.float32)

    def summary(self) -> Dict[str, Any]:
        """Describe the attribution in JSON-serialisable form.

        Returns:
            A dictionary holding the provenance, the mean lead distribution
            keyed by lead name, and the cardiac-region breakdown.
        """
        names = self.lead_names or default_lead_names(self.num_leads)
        mean_importance = self.mean_lead_importance()
        return {
            "method": self.method,
            "model_name": self.model_name,
            "samples": len(self),
            "leads": self.num_leads,
            "timesteps": int(self.values.shape[2]),
            "sampling_rate": float(self.sampling_rate),
            "crop_offset": int(self.crop_offset),
            "targets": [int(value) for value in self.target],
            "lead_importance": {name: float(value) for name, value in zip(names, mean_importance)},
            "beat_count": int(self.beat_count),
            "pqrst_fractions": self.pqrst_fractions,
            "pqrst_enrichment": self.pqrst_enrichment,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Write the attribution arrays to disk.

        Args:
            path: Destination ending in .npz

        Raises:
            ValueError: If the suffix is not .npz. The arrays are per-lead and
                per-timestep, so the flat formats FeatureSet supports would lose
                the axis that makes them worth keeping.
        """
        out_path = Path(path)
        if out_path.suffix != ".npz":
            raise ValueError(
                f"Unsupported output format: {out_path.suffix}. Attribution is saved as .npz."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "values": self.values,
            "signal": self.signal,
            "lead_importance": self.lead_importance,
            "target": self.target,
            "method": np.asarray(self.method),
            "sampling_rate": np.asarray(self.sampling_rate),
            "crop_offset": np.asarray(self.crop_offset),
            "beat_count": np.asarray(self.beat_count),
        }
        if self.model_name is not None:
            payload["model_name"] = np.asarray(self.model_name)
        if self.lead_names is not None:
            payload["lead_names"] = np.asarray(self.lead_names)
        if self.beat_profile is not None:
            payload["beat_profile"] = self.beat_profile
        if self.beat_times_ms is not None:
            payload["beat_times_ms"] = self.beat_times_ms
        if self.pqrst_fractions is not None:
            payload["pqrst_fractions"] = np.asarray(json.dumps(self.pqrst_fractions))
        if self.pqrst_enrichment is not None:
            payload["pqrst_enrichment"] = np.asarray(json.dumps(self.pqrst_enrichment))
        np.savez(str(out_path), **payload)


class ECGExplainer:
    """Runs an attribution method over data and summarises it for ECG.

    The explainer holds the same responsibilities as FeatureExtractor, with one
    inversion: it must run with gradients enabled, including when a caller
    invokes it from inside an evaluation loop that has already entered
    ``torch.no_grad()``.

    Args:
        model: Any registered model
        model_name: Registry name recorded on the results, inferred from the
            model when omitted
        method: One of METHODS
        batch_size: Recordings per attribution pass
        device: Device name, or auto to prefer CUDA then MPS then CPU
        sampling_rate: Samples per second, used for beat alignment and plots
        lead_names: Display names for the leads, defaulting to the standard
            twelve-lead order when twelve leads are present
        max_samples: Stop after this many recordings, or None for all of them
        ig_steps: Path length for integrated gradients
        occlusion_window: Span width in samples for occlusion
        occlusion_stride: Step between spans for occlusion
        align_beats: Detect R-peaks and build the beat profile
        detect_lead: Lead index R-peaks are detected on, defaulting to the
            second lead when there is more than one
        progress: Show a progress bar when tqdm is installed

    Example:
        >>> explainer = ECGExplainer.from_checkpoint("best.ckpt", "kanres")
        >>> result = explainer.explain(signals)
        >>> result.lead_importance.shape
        (100, 12)
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: Optional[str] = None,
        method: str = "saliency",
        batch_size: int = 8,
        device: str = "auto",
        sampling_rate: float = 500.0,
        lead_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        ig_steps: int = 64,
        occlusion_window: int = 100,
        occlusion_stride: int = 50,
        align_beats: bool = True,
        detect_lead: Optional[int] = None,
        num_workers: int = 0,
        progress: bool = False,
    ) -> None:
        self.model = model
        self.model_name = model_name or resolve_model_name(model)
        self.method = method
        self.batch_size = batch_size
        self.device = device
        self.sampling_rate = sampling_rate
        self.lead_names = lead_names
        self.max_samples = max_samples
        self.ig_steps = ig_steps
        self.occlusion_window = occlusion_window
        self.occlusion_stride = occlusion_stride
        self.align_beats = align_beats
        self.detect_lead = detect_lead
        self.num_workers = num_workers
        self.progress = progress

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        model_name: str,
        input_channels: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs: Any,
    ) -> "ECGExplainer":
        """Build an explainer from a saved checkpoint.

        Args:
            checkpoint: Path to an ECGTrainer checkpoint or a bare state dict
            model_name: Registered model name matching the weights
            input_channels: Lead count, required for architectures whose weights
                do not encode it
            output_size: Classifier width, normally recovered from the weights
            **kwargs: Forwarded to the constructor

        Returns:
            An explainer wrapping the loaded model.
        """
        model, _ = build_model_from_checkpoint(
            checkpoint,
            name=model_name,
            input_channels=input_channels,
            output_size=output_size,
        )
        return cls(model, model_name=model_name, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        weights: Union[str, Path],
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "ECGExplainer":
        """Build an explainer from registered pretrained weights or a file.

        Args:
            weights: Name of registered pretrained weights, or a path
            model_name: Registered model name, inferred from the weight registry
                when the weights are registered
            **kwargs: Forwarded to the constructor, except that unrecognised
                arguments are passed to the model instead

        Returns:
            An explainer wrapping the loaded model.

        Raises:
            ValueError: If the model name is neither given nor inferable.
        """
        resolved = model_name or model_name_for_weights(str(weights))
        if resolved is None:
            raise ValueError(
                f"Could not determine which model '{weights}' belongs to. Pass model_name "
                f"explicitly."
            )
        explainer_keys = {
            "align_beats",
            "batch_size",
            "detect_lead",
            "device",
            "ig_steps",
            "lead_names",
            "max_samples",
            "method",
            "num_workers",
            "occlusion_stride",
            "occlusion_window",
            "progress",
            "sampling_rate",
        }
        explainer_kwargs = {k: v for k, v in kwargs.items() if k in explainer_keys}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in explainer_keys}
        model = _load_pretrained(resolved, weights, **model_kwargs)
        return cls(model, model_name=resolved, **explainer_kwargs)

    @property
    def input_channels(self) -> Optional[int]:
        """Lead count recovered from the model, or None when unrecoverable."""
        declared = getattr(self.model, "input_channels", None)
        if isinstance(declared, int):
            return declared
        return infer_input_channels(self.model.state_dict())

    def explain(
        self, data: ExplainInput, target: TargetSpec = None, split: str = "test"
    ) -> Attribution:
        """Attribute a model's predictions over the data.

        Args:
            data: An array, tensor, file path, Dataset, DataLoader, or
                ECGDataModule
            target: None for each sample's own predicted class, an integer for
                one class throughout, or one index per sample
            split: Which split to use when data is an ECGDataModule

        Returns:
            The attribution, with the lead distribution always present and the
            beat profile present when align_beats is set and beats were found.

        Raises:
            ValueError: If the split or method is unknown.
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}. Choose one of {', '.join(SPLITS)}.")
        if self.method not in _METHODS:
            raise ValueError(f"Unknown method: {self.method}. Choose one of {', '.join(METHODS)}.")

        device = _resolve_device(self.device)
        loader = self._as_dataloader(data, split)

        value_chunks: List[np.ndarray] = []
        signal_chunks: List[np.ndarray] = []
        target_chunks: List[np.ndarray] = []
        crop_offset = 0
        seen = 0

        with _gradient_ready(self.model, device):
            iterator: Any = loader
            if self.progress and tqdm is not None:
                iterator = tqdm(loader, desc=f"Attributing ({self.method})")

            for batch in iterator:
                signals = batch[0] if isinstance(batch, (list, tuple)) else batch
                signals = signals.to(device=device, dtype=torch.float32)

                if self.max_samples is not None:
                    remaining = self.max_samples - seen
                    if remaining <= 0:
                        break
                    signals = signals[:remaining]

                signals, crop_offset = fit_to_constraints(signals, self.model, self.model_name)
                batch_target = self._batch_target(target, seen, signals.shape[0])

                with torch.no_grad():
                    index = _resolve_target(_forward(self.model, signals), batch_target)

                values = self._run_method(signals, index)
                value_chunks.append(values.detach().cpu().numpy().astype(np.float32))
                signal_chunks.append(signals.detach().cpu().numpy().astype(np.float32))
                target_chunks.append(index.detach().cpu().numpy().astype(np.int64))
                seen += signals.shape[0]

        if not value_chunks:
            raise ValueError("The supplied data yielded no recordings to explain.")

        values = np.concatenate(value_chunks, axis=0)
        signal = np.concatenate(signal_chunks, axis=0)
        targets = np.concatenate(target_chunks, axis=0)
        names = self.lead_names or default_lead_names(values.shape[1])

        result = Attribution(
            values=values,
            signal=signal,
            lead_importance=lead_importance(values),
            target=targets,
            method=self.method,
            model_name=self.model_name,
            sampling_rate=self.sampling_rate,
            crop_offset=crop_offset,
            lead_names=names,
        )

        if self.align_beats:
            self._attach_beat_profile(result, values, signal)

        return result

    def _run_method(self, signals: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Dispatch to the configured attribution method.

        Args:
            signals: Batch of shape (batch, leads, time) on the target device
            index: Resolved class index per sample

        Returns:
            Attribution of shape (batch, leads, time).
        """
        if self.method == "integrated-gradients":
            return integrated_gradients(self.model, signals, index, steps=self.ig_steps)
        if self.method == "occlusion":
            return occlusion(
                self.model,
                signals,
                index,
                window=self.occlusion_window,
                stride=self.occlusion_stride,
            )
        return saliency(self.model, signals, index)

    def _batch_target(self, target: TargetSpec, seen: int, size: int) -> TargetSpec:
        """Slice a per-sample target specification down to the current batch.

        Args:
            target: The caller's target specification
            seen: How many recordings have already been explained
            size: Size of the current batch

        Returns:
            The target specification that applies to this batch.
        """
        if target is None or isinstance(target, int):
            return target
        indices = np.asarray(target).reshape(-1)
        if indices.size == 1:
            return int(indices[0])
        return indices[seen : seen + size]

    def _attach_beat_profile(
        self, result: Attribution, values: np.ndarray, signal: np.ndarray
    ) -> None:
        """Detect beats, average attribution over them, and record the shares.

        R-peaks are detected on the ECG, never on the attribution. The cardiac
        cycle is a property of the recording, so anchoring to the attribution
        would make the window positions depend on the method under test and the
        beat counts would differ between saliency and occlusion on the same
        recording.

        Profiles are pooled across the batch weighted by how many beats each
        recording contributed, so the result answers a question about the model
        rather than about one recording.

        Args:
            result: The attribution to annotate in place
            values: Attribution of shape (batch, leads, time)
            signal: The signal the model saw, of the same shape
        """
        lead = self.detect_lead
        if lead is None:
            lead = 1 if values.shape[1] > 1 else 0

        profiles: List[np.ndarray] = []
        weights: List[int] = []
        times: Optional[np.ndarray] = None

        for attribution, trace in zip(values, signal):
            peaks = detect_r_peaks(trace[lead], self.sampling_rate)
            profile, axis, count = beat_aligned_attribution(
                attribution, peaks, self.sampling_rate, BEAT_WINDOW_MS
            )
            times = axis
            if count:
                profiles.append(profile)
                weights.append(count)

        if not profiles or times is None:
            warnings.warn(
                "No complete beats were found, so no beat profile was produced. Check the "
                "sampling rate, or widen the detector's band with detect_r_peaks.",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        total = float(sum(weights))
        pooled = np.tensordot(np.asarray(weights, dtype=np.float64) / total, profiles, axes=(0, 0))

        result.beat_profile = pooled.astype(np.float32)
        result.beat_times_ms = times
        result.beat_count = int(total)
        result.pqrst_fractions = pqrst_fractions(pooled, times, PQRST_WINDOWS_MS)
        result.pqrst_enrichment = pqrst_enrichment(result.pqrst_fractions, times, PQRST_WINDOWS_MS)

    def _as_dataloader(self, data: ExplainInput, split: str) -> DataLoader:
        """Normalise any supported input to a non-shuffling DataLoader.

        Args:
            data: Any supported input
            split: Split name, used only for an ECGDataModule

        Returns:
            A loader yielding either signals alone or (signals, labels).

        Raises:
            TypeError: If the input type is not supported.
            RuntimeError: If a data module split is unavailable.
        """
        if isinstance(data, ECGDataModule):
            if data.dataset is None or data.train_dataset is None:
                data.setup()
            subset = data.dataset if split == "all" else getattr(data, f"{split}_dataset")
            if subset is None:
                raise RuntimeError(f"Data module has no '{split}' split available.")
            return self._build_loader(subset)

        if isinstance(data, DataLoader):
            if isinstance(data.sampler, RandomSampler):
                warnings.warn(
                    "The supplied DataLoader shuffles, so explained rows will not follow "
                    "dataset order and cannot be joined to external metadata. Pass "
                    "shuffle=False for a reproducible ordering.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return data

        if isinstance(data, Dataset):
            return self._build_loader(data)

        if isinstance(data, (str, Path)):
            array = as_model_input(read_ecg_recording(data), self.input_channels)
            return self._build_loader(TensorDataset(torch.as_tensor(array)))

        if isinstance(data, torch.Tensor):
            array = as_model_input(data.detach().cpu().numpy(), self.input_channels)
            return self._build_loader(TensorDataset(torch.as_tensor(array)))

        if isinstance(data, np.ndarray):
            array = as_model_input(data, self.input_channels)
            return self._build_loader(TensorDataset(torch.as_tensor(array)))

        raise TypeError(
            f"Unsupported input type: {type(data).__name__}. Pass an array, tensor, file path, "
            f"Dataset, DataLoader, or ECGDataModule."
        )

    def _build_loader(self, dataset: Dataset) -> DataLoader:
        """Wrap a dataset in a loader that preserves ordering.

        Args:
            dataset: Any map-style dataset

        Returns:
            A DataLoader with shuffling disabled.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def explain(
    data: ExplainInput,
    model: Optional[nn.Module] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    weights: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    input_channels: Optional[int] = None,
    output_size: Optional[int] = None,
    target: TargetSpec = None,
    split: str = "test",
    **kwargs: Any,
) -> Attribution:
    """Attribute a model's predictions in one call.

    Exactly one weight source must be given: an already-built model, a
    checkpoint path, or pretrained weights.

    Args:
        data: An array, tensor, file path, Dataset, DataLoader, or ECGDataModule
        model: An already-built model
        checkpoint: Path to a checkpoint, requiring model_name
        weights: Registered pretrained weight name, or a path to a weights file
        model_name: Registered model name
        input_channels: Lead count, when it cannot be recovered from the weights
        output_size: Classifier width, when it cannot be recovered
        target: None for each sample's own predicted class, an integer, or one
            index per sample
        split: Which split to use when data is an ECGDataModule
        **kwargs: Forwarded to ECGExplainer

    Returns:
        The attribution, with its lead distribution and beat profile.

    Raises:
        ValueError: If not exactly one weight source is given, or a checkpoint
            is given without a model name.

    Example:
        >>> result = explain(signals, checkpoint="best.ckpt", model_name="kanres")
        >>> result.pqrst_fractions["P"]
        0.31
    """
    supplied = [
        name
        for name, value in (("model", model), ("checkpoint", checkpoint), ("weights", weights))
        if value is not None
    ]
    if len(supplied) != 1:
        raise ValueError(
            f"Pass exactly one of model, checkpoint, or weights; got "
            f"{', '.join(supplied) or 'none'}."
        )

    if model is not None:
        explainer = ECGExplainer(model, model_name=model_name, **kwargs)
    elif checkpoint is not None:
        if model_name is None:
            raise ValueError("model_name is required when loading from a checkpoint.")
        explainer = ECGExplainer.from_checkpoint(
            checkpoint,
            model_name,
            input_channels=input_channels,
            output_size=output_size,
            **kwargs,
        )
    else:
        explainer = ECGExplainer.from_pretrained(weights, model_name=model_name, **kwargs)

    return explainer.explain(data, target=target, split=split)
