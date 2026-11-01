"""Predictions on raw recordings from a trained checkpoint.

A checkpoint records the preprocessing its training data went through and the
classes its outputs stand for, so a raw recording can be prepared the same way
and its outputs read correctly: a softmax over the classes, a sigmoid for a
single-logit binary head, or a sigmoid per class for a multi-label one. Each
recording is cut into training-length segments and gets one prediction, the
average over its segments. The predict command goes through this module too.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from deepecgkit.datasets.io import as_model_input, load_recordings, prepare_recording
from deepecgkit.models.factory import build_model_from_checkpoint, infer_input_channels
from deepecgkit.utils.checkpoints import load_checkpoint_file, resolve_checkpoint
from deepecgkit.utils.device import resolve_device

__all__ = ["ECGPredictor", "Prediction", "predict"]

#: Probability above which a multi-label model counts a class as present.
THRESHOLD = 0.5

#: What predict and ECGPredictor accept as a recording.
RecordingInput = str | Path | np.ndarray | torch.Tensor


@dataclass
class Prediction:
    """One recording's prediction, averaged over its segments.

    Attributes:
        probabilities: Each class name with its probability, averaged over
            the recording's segments
        segment_probabilities: Array of shape (num_segments, num_classes)
        segment_starts: Start of each segment within the recording, in seconds
        multi_label: Whether the classes are independent, so that any number
            of them, including none, can be present
        notes: What the preprocessing did or assumed, for reporting
    """

    probabilities: dict[str, float]
    segment_probabilities: np.ndarray
    segment_starts: np.ndarray
    multi_label: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        """The classes predicted present; exactly one unless the model is multi-label."""
        if self.multi_label:
            return [name for name, p in self.probabilities.items() if p > THRESHOLD]
        return [max(self.probabilities, key=self.probabilities.__getitem__)]

    @property
    def label(self) -> str:
        """The predicted class, or the present classes joined by commas ("none" if none are)."""
        return ", ".join(self.labels) or "none"

    @property
    def confidence(self) -> float:
        """The highest class probability."""
        return max(self.probabilities.values())

    @property
    def num_segments(self) -> int:
        """How many segments the prediction averages over."""
        return len(self.segment_probabilities)


class ECGPredictor:
    """A trained model ready to predict on raw recordings.

    Loading the checkpoint once and reusing the predictor is faster than
    calling predict for every recording.

    Args:
        model: The trained model
        preprocessing: The checkpoint's preprocessing record (sampling rate,
            segment length, normalisation); None feeds recordings to the
            model unchanged
        class_names: Names of the model's outputs, in order
        multi_label: Whether the outputs are independent sigmoids
        input_channels: Leads the model consumes, used to arrange 2-D input
        device: Device to run on ('auto', 'cpu', 'cuda', 'mps')
        batch_size: Segments per forward pass

    Example:
        >>> predictor = ECGPredictor.from_checkpoint("runs/my-run")
        >>> predictor.predict("ecg.npy", sampling_rate=250).label
        'A'
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessing: dict[str, Any] | None = None,
        class_names: list[str] | None = None,
        multi_label: bool = False,
        input_channels: int | None = None,
        device: str | torch.device = "auto",
        batch_size: int = 32,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.preprocessing = preprocessing
        self.class_names = list(class_names) if class_names else None
        self.multi_label = multi_label
        self.input_channels = input_channels
        self.batch_size = max(int(batch_size), 1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        model_name: str | None = None,
        input_channels: int | None = None,
        preprocess: bool = True,
        **kwargs: Any,
    ) -> "ECGPredictor":
        """Load a predictor from a checkpoint or a run directory.

        Args:
            checkpoint: Checkpoint file, or a run directory standing for its
                best checkpoint
            model_name: Registered model name; optional when the checkpoint
                records it
            input_channels: Lead count, for checkpoints that record none and
                architectures whose weights do not show it
            preprocess: Apply the checkpoint's preprocessing; turn off for
                input that has already been prepared
            **kwargs: Forwarded to the constructor (device, batch_size)

        Returns:
            A predictor wrapping the loaded model.
        """
        path = resolve_checkpoint(checkpoint)
        stored = load_checkpoint_file(path)
        stored = stored if isinstance(stored, dict) else {}
        recorded = stored.get("model_config") or {}
        input_channels = (
            input_channels
            or recorded.get("input_channels")
            or infer_input_channels(stored.get("model_state_dict") or stored)
        )
        model, _ = build_model_from_checkpoint(path, name=model_name, input_channels=input_channels)

        data_config = stored.get("data_config") or {}
        train_config = stored.get("train_config") or {}
        return cls(
            model,
            preprocessing=data_config.get("preprocessing") if preprocess else None,
            class_names=data_config.get("class_names"),
            multi_label=bool(data_config.get("multi_label") or train_config.get("multi_label")),
            input_channels=input_channels,
            **kwargs,
        )

    def predict(self, recording: RecordingInput, sampling_rate: float | None = None) -> Prediction:
        """Predict one recording.

        Args:
            recording: A .npy, .csv or WFDB file, or an array shaped (samples,)
                or (leads, samples)
            sampling_rate: Rate the recording was sampled at, in Hz, so it can
                be resampled to the training rate. WFDB files carry their
                own; otherwise None assumes it already matches.

        Returns:
            The recording's prediction.

        Raises:
            ValueError: If the input holds more than one recording; use
                predict_all for those.
        """
        predictions = self.predict_all(recording, sampling_rate)
        if len(predictions) != 1:
            raise ValueError(
                f"The input holds {len(predictions)} recordings; predict_all gives one "
                f"prediction for each."
            )
        return predictions[0]

    def predict_all(
        self, recordings: RecordingInput, sampling_rate: float | None = None
    ) -> list[Prediction]:
        """Predict every recording in a file or array.

        Args:
            recordings: A file, or an array of one or more recordings. With a
                single-lead model, each row of a 2-D array is a recording.
            sampling_rate: As for predict

        Returns:
            One prediction per recording, in order.
        """
        if isinstance(recordings, str | Path):
            batch, file_rate, notes = load_recordings(recordings, self.input_channels)
            sampling_rate = sampling_rate if sampling_rate is not None else file_rate
        else:
            if isinstance(recordings, torch.Tensor):
                recordings = recordings.detach().cpu().numpy()
            batch, notes = as_model_input(np.asarray(recordings), self.input_channels), []

        predictions = []
        for recording in batch:
            prepared = prepare_recording(recording, self.preprocessing, sampling_rate)
            per_segment = self._probabilities(prepared.segments)
            names = self.class_names or [f"class {i}" for i in range(per_segment.shape[1])]
            mean = per_segment.mean(axis=0)
            predictions.append(
                Prediction(
                    probabilities={name: float(p) for name, p in zip(names, mean, strict=True)},
                    segment_probabilities=per_segment,
                    segment_starts=prepared.start_seconds,
                    multi_label=self.multi_label,
                    notes=[*notes, *prepared.notes],
                )
            )
        return predictions

    def _probabilities(self, segments: np.ndarray) -> np.ndarray:
        """Run the model over one recording's segments.

        Args:
            segments: Array of shape (num_segments, leads, samples)

        Returns:
            Probabilities of shape (num_segments, num_classes). A single-logit
            head yields the two columns [1 - p, p].
        """
        chunks = []
        with torch.no_grad():
            for start in range(0, len(segments), self.batch_size):
                batch = torch.as_tensor(segments[start : start + self.batch_size])
                outputs = self.model(batch.float().to(self.device))
                outputs = getattr(outputs, "logits", outputs)
                if self.multi_label:
                    probabilities = torch.sigmoid(outputs)
                elif outputs.shape[-1] == 1:
                    positive = torch.sigmoid(outputs.squeeze(-1))
                    probabilities = torch.stack([1 - positive, positive], dim=-1)
                else:
                    probabilities = torch.softmax(outputs, dim=-1)
                chunks.append(probabilities.cpu())
        return torch.cat(chunks).numpy()


def predict(
    recording: RecordingInput,
    checkpoint: str | Path,
    sampling_rate: float | None = None,
    **kwargs: Any,
) -> Prediction:
    """Predict one recording with a trained checkpoint, in one call.

    The recording is resampled, segmented and normalised the way the
    checkpoint's training data was, so a raw file can be passed as it is.

    Args:
        recording: A .npy, .csv or WFDB file, or an array shaped (samples,)
            or (leads, samples)
        checkpoint: Checkpoint file, or a run directory standing for its best
            checkpoint
        sampling_rate: Rate the recording was sampled at, in Hz. WFDB files
            carry their own; otherwise None assumes it matches training.
        **kwargs: Forwarded to ECGPredictor.from_checkpoint (model_name,
            device, batch_size, preprocess)

    Returns:
        The prediction, averaged over the recording's segments.

    Example:
        >>> result = predict("ecg.npy", checkpoint="runs/my-run", sampling_rate=250)
        >>> result.label, round(result.confidence, 2)
        ('A', 0.87)
    """
    return ECGPredictor.from_checkpoint(checkpoint, **kwargs).predict(recording, sampling_rate)
