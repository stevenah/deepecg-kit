"""Predict command."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wfdb

from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import _get_model_registry


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
        logger.info(f"Loading model from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        if model_name is None:
            logger.error("--model is required for predict command")
            return 1

        model_registry = _get_model_registry()
        if model_name not in model_registry:
            logger.error(f"Unknown model: {model_name}")
            return 1

        model_class = model_registry[model_name]
        model = model_class(input_channels=1, output_size=4)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        logger.info(f"Loading input data from {input_path}...")
        if input_file.suffix == ".npy":
            data = np.load(input_path)
        elif input_file.suffix == ".csv":
            data = pd.read_csv(input_path).values
        elif input_file.suffix in (".dat", ".hea"):
            record = wfdb.rdrecord(str(input_file.with_suffix("")))
            data = record.p_signal
        else:
            logger.error(f"Unsupported input format: {input_file.suffix}")
            logger.info("Supported formats: .npy, .csv, .dat/.hea (WFDB)")
            return 1

        if data.ndim == 1:
            data = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            data = data.reshape(data.shape[0], 1, -1)

        tensor = torch.tensor(data, dtype=torch.float32)

        logger.info("Running inference...")
        device = "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
        model = model.to(device)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
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
