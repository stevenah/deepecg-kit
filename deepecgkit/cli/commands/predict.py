"""Predict command."""

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    report_shape_error,
    require_checkpoint,
    require_file,
    require_model,
    resolve_model_name,
)
from deepecgkit.cli.commands.extract import UNPREPROCESSED_WARNING
from deepecgkit.cli.logger import CLILogger
from deepecgkit.datasets.io import SUPPORTED_ECG_SUFFIXES, load_recordings, prepare_recording
from deepecgkit.models.factory import build_model_from_checkpoint, infer_input_channels
from deepecgkit.registry import get_model_names
from deepecgkit.utils.device import resolve_device


def _segment_probabilities(
    model: torch.nn.Module,
    segments: np.ndarray,
    device: torch.device,
    batch_size: int,
    multi_label: bool,
    binary_classification: bool,
) -> np.ndarray:
    """Run the model over one recording's segments.

    Args:
        model: The loaded model, already on device and in eval mode
        segments: Array of shape (num_segments, leads, samples)
        device: Device the model lives on
        batch_size: Segments per forward pass
        multi_label: Whether the head is a set of independent sigmoids
        binary_classification: Whether the head is a single AF-vs-rest logit

    Returns:
        Probabilities of shape (num_segments, num_classes).
    """
    chunks: list[torch.Tensor] = []
    step = max(int(batch_size), 1)
    with torch.no_grad():
        for start in range(0, len(segments), step):
            batch = torch.as_tensor(segments[start : start + step], dtype=torch.float32)
            outputs = model(batch.to(device))
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            if multi_label:
                probabilities = torch.sigmoid(outputs)
            elif binary_classification and outputs.shape[-1] == 1:
                positive = torch.sigmoid(outputs.squeeze(-1))
                probabilities = torch.stack([1 - positive, positive], dim=-1)
            else:
                probabilities = torch.softmax(outputs, dim=-1)
            chunks.append(probabilities.cpu())
    return torch.cat(chunks, dim=0).numpy()


def _label_for(prediction: Any, class_names: list[str] | None) -> str:
    """Render a prediction as class names where they are known."""
    if isinstance(prediction, list):
        if class_names:
            chosen = [class_names[i] for i, flag in enumerate(prediction) if flag]
            return ", ".join(chosen) if chosen else "none"
        return str(prediction)
    if class_names and 0 <= prediction < len(class_names):
        return str(class_names[prediction])
    return f"class {prediction}"


def predict(
    checkpoint: str,
    input_path: str,
    output_path: str | None = None,
    batch_size: int = defaults.BATCH_SIZE,
    accelerator: str = "auto",
    model_name: str | None = None,
    sampling_rate: float | None = None,
    preprocess: bool = True,
    logger: CLILogger | None = None,
) -> int:
    """Run inference on ECG data.

    The recording is first put through the preprocessing the checkpoint was
    trained with (resampling, segmenting, normalising), so a raw file can be
    passed as it is. Each recording gets one prediction, averaged over its
    segments.

    Args:
        checkpoint: Checkpoint file, or a run directory whose best checkpoint
            is used
        input_path: Path to a .npy, .csv, or WFDB recording
        output_path: Optional JSON file to write the results to
        batch_size: Segments per forward pass
        accelerator: Device to run on
        model_name: Registered model name; optional when the checkpoint
            records it
        sampling_rate: Sampling rate of the recording in Hz; read from the
            header for WFDB files, otherwise assumed to match training
        preprocess: Apply the checkpoint's preprocessing; turn off for input
            that has already been prepared
        logger: CLI logger

    Returns:
        Zero on success, one on failure.
    """
    logger = logger or CLILogger()

    resolved = require_checkpoint(checkpoint, logger)
    input_code = require_file(input_path, "Input file", logger)
    if resolved is None or input_code is not None:
        return 1
    checkpoint = resolved

    input_file = Path(input_path)

    try:
        if model_name is not None and (code := require_model(model_name, logger)) is not None:
            return code

        if input_file.suffix not in SUPPORTED_ECG_SUFFIXES:
            logger.error(f"Unsupported input format: {input_file.suffix}")
            logger.info("Supported formats: .npy, .csv, .dat/.hea (WFDB)")
            return 1

        logger.info(f"Loading model from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ckpt = ckpt if isinstance(ckpt, dict) else {}

        model_name = resolve_model_name(ckpt, model_name, logger)
        if model_name is None:
            return 1

        train_config = ckpt.get("train_config") or {}
        multi_label = bool(train_config.get("multi_label", False))
        binary_classification = bool(train_config.get("binary_classification", False))

        data_config = ckpt.get("data_config") or {}
        preprocessing = data_config.get("preprocessing") if preprocess else None
        class_names = data_config.get("class_names")

        recorded = ckpt.get("model_config") or {}
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        input_channels = recorded.get("input_channels") or infer_input_channels(state_dict)
        model, _ = build_model_from_checkpoint(
            checkpoint, name=model_name, input_channels=input_channels
        )
        model.eval()

        logger.info(f"Loading input data from {input_path}...")
        try:
            recordings, file_rate, load_notes = load_recordings(input_file, input_channels)
        except ValueError as shape_error:
            logger.error(str(shape_error))
            return 1
        source_rate = sampling_rate if sampling_rate is not None else file_rate

        if preprocess and not preprocessing:
            logger.warning(UNPREPROCESSED_WARNING)

        logger.info("Running inference...")
        device = resolve_device(accelerator)
        model = model.to(device)

        predictions: list[Any] = []
        probabilities: list[list[float]] = []
        segment_rows: list[dict[str, Any]] = []
        notes: list[str] = []
        segment_counts: list[int] = []
        for index, recording in enumerate(recordings):
            prepared = prepare_recording(recording, preprocessing, source_rate)
            if index == 0:
                notes = [*load_notes, *prepared.notes]
            per_segment = _segment_probabilities(
                model,
                prepared.segments,
                device,
                batch_size,
                multi_label,
                binary_classification,
            )
            mean = per_segment.mean(axis=0)
            if multi_label:
                prediction: Any = (mean > 0.5).astype(int).tolist()
                segment_predictions: list[Any] = (per_segment > 0.5).astype(int).tolist()
            else:
                prediction = int(mean.argmax())
                segment_predictions = per_segment.argmax(axis=1).astype(int).tolist()

            predictions.append(prediction)
            probabilities.append(mean.tolist())
            segment_counts.append(len(per_segment))
            for row, start in enumerate(prepared.start_seconds):
                segment_rows.append(
                    {
                        "recording": index,
                        "start_seconds": float(start),
                        "prediction": segment_predictions[row],
                        "probabilities": per_segment[row].tolist(),
                    }
                )

        if notes:
            logger.info(f"Preprocessing: {'; '.join(notes)}.")

        labels = [_label_for(prediction, class_names) for prediction in predictions]
        results = {
            "model": model_name,
            "class_names": class_names,
            "predictions": predictions,
            "labels": labels,
            "probabilities": probabilities,
            "segments": segment_rows,
        }

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.info("\nPrediction Results:")
            logger.info("-" * 40)
            for i, (label, prob, count) in enumerate(
                zip(labels, probabilities, segment_counts, strict=False)
            ):
                detail = f"confidence={max(prob):.4f}"
                if count > 1:
                    detail += f", averaged over {count} segments"
                logger.info(f"  Recording {i}: {label} ({detail})")
            if class_names and not multi_label:
                logger.info(f"  Classes: {', '.join(class_names)}")

        return 0
    except RuntimeError as e:
        return report_shape_error(e, model_name or "the model", "Prediction", logger)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


def add_parser(subparsers: Any) -> Any:
    """Register the predict subcommand's arguments."""
    parser = subparsers.add_parser("predict", help="Run inference on ECG data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file, or a run directory whose best checkpoint is used",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        choices=get_model_names(),
        metavar="MODEL",
        help="Model architecture (default: the one recorded in the checkpoint)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input ECG file (.npy, .csv, or WFDB .dat/.hea)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save predictions (JSON format)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        metavar="HZ",
        help="Sampling rate of the input recording, so it can be resampled to the rate the "
        "model was trained at (default: the WFDB header's, otherwise assumed to match)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Pass the input to the model unchanged, for recordings that are already "
        "resampled, segmented and normalised like the training data",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=defaults.BATCH_SIZE,
        help=f"Segments per forward pass (default: {defaults.BATCH_SIZE})",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=defaults.ACCELERATOR,
        choices=defaults.ACCELERATOR_CHOICES,
        help="Accelerator to use (default: auto)",
    )
    parser.set_defaults(handler="predict", to_kwargs=_to_kwargs)
    return parser


def _to_kwargs(args: Any) -> dict[str, Any]:
    return {
        "checkpoint": args.checkpoint,
        "input_path": args.input,
        "output_path": args.output,
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
        "model_name": args.model,
        "sampling_rate": args.sampling_rate,
        "preprocess": not args.no_preprocess,
    }
