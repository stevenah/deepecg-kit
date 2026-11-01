"""Predict command."""

import json
import traceback
from pathlib import Path
from typing import Any

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    read_checkpoint,
    report_shape_error,
    require_checkpoint,
    require_file,
    require_model,
    resolve_model_name,
)
from deepecgkit.cli.commands.extract import UNPREPROCESSED_WARNING
from deepecgkit.cli.logger import CLILogger
from deepecgkit.datasets.io import SUPPORTED_ECG_SUFFIXES
from deepecgkit.prediction import THRESHOLD, ECGPredictor, Prediction
from deepecgkit.registry import get_model_names


def _as_record(index: int, result: Prediction) -> dict[str, Any]:
    """Lay one recording's prediction out the way the JSON output stores it.

    Args:
        index: Position of the recording in the input
        result: Its prediction

    Returns:
        The class index (single-label) or multi-hot list (multi-label), the
        label, the mean probabilities, and one row per segment.
    """
    per_segment = result.segment_probabilities
    if result.multi_label:
        prediction: Any = [int(p > THRESHOLD) for p in result.probabilities.values()]
        segment_predictions: list[Any] = (per_segment > THRESHOLD).astype(int).tolist()
    else:
        prediction = int(per_segment.mean(axis=0).argmax())
        segment_predictions = per_segment.argmax(axis=1).astype(int).tolist()
    return {
        "prediction": prediction,
        "label": result.label,
        "probabilities": list(result.probabilities.values()),
        "segments": [
            {
                "recording": index,
                "start_seconds": float(start),
                "prediction": segment_predictions[row],
                "probabilities": per_segment[row].tolist(),
            }
            for row, start in enumerate(result.segment_starts)
        ],
    }


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
        model_name = resolve_model_name(read_checkpoint(checkpoint), model_name, logger)
        if model_name is None:
            return 1
        predictor = ECGPredictor.from_checkpoint(
            checkpoint,
            model_name=model_name,
            preprocess=preprocess,
            device=accelerator,
            batch_size=batch_size,
        )
        if preprocess and not predictor.preprocessing:
            logger.warning(UNPREPROCESSED_WARNING)

        logger.info(f"Running inference on {input_path}...")
        try:
            results = predictor.predict_all(input_file, sampling_rate)
        except ValueError as shape_error:
            logger.error(str(shape_error))
            return 1

        if results and results[0].notes:
            logger.info(f"Preprocessing: {'; '.join(results[0].notes)}.")

        records = [_as_record(index, result) for index, result in enumerate(results)]
        payload = {
            "model": model_name,
            "class_names": predictor.class_names,
            "predictions": [record["prediction"] for record in records],
            "labels": [record["label"] for record in records],
            "probabilities": [record["probabilities"] for record in records],
            "segments": [row for record in records for row in record["segments"]],
        }

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.info("\nPrediction Results:")
            logger.info("-" * 40)
            for index, result in enumerate(results):
                detail = f"confidence={result.confidence:.4f}"
                if result.num_segments > 1:
                    detail += f", averaged over {result.num_segments} segments"
                logger.info(f"  Recording {index}: {result.label} ({detail})")
            if predictor.class_names and not predictor.multi_label:
                logger.info(f"  Classes: {', '.join(predictor.class_names)}")

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
