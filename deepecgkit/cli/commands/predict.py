"""Predict command."""

import json
from pathlib import Path
from typing import Optional

import torch

from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import _get_model_registry
from deepecgkit.datasets.io import SUPPORTED_ECG_SUFFIXES, as_model_input, read_ecg_recording
from deepecgkit.models.factory import build_model_from_checkpoint, infer_input_channels


def predict(
    checkpoint: str,
    input_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 1,
    accelerator: str = "auto",
    model_name: Optional[str] = None,
    logger: Optional[CLILogger] = None,
) -> int:
    """Run inference on ECG data."""
    logger = logger or CLILogger()

    checkpoint_file = Path(checkpoint)
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        if model_name is None:
            logger.error("--model is required for predict command")
            return 1

        model_registry = _get_model_registry()
        if model_name not in model_registry:
            logger.error(f"Unknown model: {model_name}")
            return 1

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
        device = "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
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
            for i, (pred, prob) in enumerate(zip(results["predictions"], results["probabilities"])):
                logger.info(f"  Sample {i}: class={pred}, confidence={max(prob):.4f}")

        return 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1
