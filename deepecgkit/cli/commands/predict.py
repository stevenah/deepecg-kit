"""Predict command."""

import json
import traceback
from pathlib import Path
from typing import Any

import torch

from deepecgkit.cli import defaults
from deepecgkit.cli._common import require_file, require_model
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import MODEL_NAMES
from deepecgkit.datasets.io import SUPPORTED_ECG_SUFFIXES, as_model_input, read_ecg_recording
from deepecgkit.models.factory import build_model_from_checkpoint, infer_input_channels
from deepecgkit.utils.device import resolve_device


def predict(
    checkpoint: str,
    input_path: str,
    output_path: str | None = None,
    batch_size: int = 1,
    accelerator: str = "auto",
    model_name: str | None = None,
    logger: CLILogger | None = None,
) -> int:
    """Run inference on ECG data."""
    logger = logger or CLILogger()

    for code in (
        require_file(checkpoint, "Checkpoint", logger),
        require_file(input_path, "Input file", logger),
    ):
        if code is not None:
            return code

    input_file = Path(input_path)

    try:
        if model_name is None:
            logger.error("--model is required for predict command")
            return 1

        if (code := require_model(model_name, logger)) is not None:
            return code

        if input_file.suffix not in SUPPORTED_ECG_SUFFIXES:
            logger.error(f"Unsupported input format: {input_file.suffix}")
            logger.info("Supported formats: .npy, .csv, .dat/.hea (WFDB)")
            return 1

        logger.info(f"Loading model from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        train_config = ckpt.get("train_config") or {}
        multi_label = bool(train_config.get("multi_label", False))
        binary_classification = bool(train_config.get("binary_classification", False))

        recorded = ckpt.get("model_config") or {}
        input_channels = recorded.get("input_channels") or infer_input_channels(
            ckpt["model_state_dict"]
        )
        model, _ = build_model_from_checkpoint(
            checkpoint, name=model_name, input_channels=input_channels
        )
        model.eval()

        logger.info(f"Loading input data from {input_path}...")
        data = read_ecg_recording(input_file)
        try:
            data = as_model_input(data, input_channels)
        except ValueError as shape_error:
            logger.error(str(shape_error))
            return 1

        tensor = torch.tensor(data, dtype=torch.float32)

        logger.info("Running inference...")
        device = resolve_device(accelerator)
        model = model.to(device)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            if multi_label:
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).int()
            elif binary_classification and outputs.shape[-1] == 1:
                positive = torch.sigmoid(outputs.squeeze(-1))
                probabilities = torch.stack([1 - positive, positive], dim=-1)
                predictions = (positive > 0.5).int()
            else:
                probabilities = torch.softmax(outputs, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

        results = {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probabilities.cpu().numpy().tolist(),
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
            for i, (pred, prob) in enumerate(
                zip(results["predictions"], results["probabilities"], strict=False)
            ):
                logger.info(f"  Sample {i}: class={pred}, confidence={max(prob):.4f}")

        return 0
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
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
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
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)",
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
    }
