"""Run inference on raw ECG data using a trained checkpoint.

Supports .npy, .csv, and WFDB (.dat/.hea) input formats.
Outputs per-sample predictions with class probabilities.

Usage:
    python examples/inference.py --checkpoint model.ckpt --input ecg_data.npy
    python examples/inference.py --checkpoint model.ckpt --input ecg_data.csv --output predictions.json
    python examples/inference.py --checkpoint model.ckpt --input record_name --class-names Normal AF Other Noisy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from deepecgkit.training import ECGLitModel


def load_ecg_signal(path: str) -> torch.Tensor:
    """Load ECG signal from various file formats."""
    p = Path(path)

    if p.suffix == ".npy":
        data = np.load(str(p))
    elif p.suffix == ".csv":
        data = pd.read_csv(str(p), header=None).values
    elif p.suffix in (".dat", ".hea") or not p.suffix:
        record_path = str(p.with_suffix(""))
        record = wfdb.rdrecord(record_path)
        data = record.p_signal.T
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    data = data.astype(np.float32)

    if data.ndim == 1:
        data = data.reshape(1, 1, -1)
    elif data.ndim == 2:
        data = (
            data.reshape(1, *data.shape)
            if data.shape[0] <= 12
            else data.reshape(1, data.shape[1], data.shape[0])
        )
    elif data.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    return torch.tensor(data, dtype=torch.float32)


def predict(
    lit_model: ECGLitModel,
    signal: torch.Tensor,
    class_names: list[str] | None = None,
) -> list[dict]:
    """Run prediction and return per-sample results."""
    lit_model.eval()
    with torch.no_grad():
        logits = lit_model(signal)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    results = []
    for i in range(len(signal)):
        probs = probabilities[i].numpy()
        pred_idx = predicted_classes[i].item()
        pred_label = class_names[pred_idx] if class_names else str(pred_idx)
        confidence = probs[pred_idx]

        sample_result = {
            "sample_index": i,
            "predicted_class": pred_label,
            "predicted_index": pred_idx,
            "confidence": float(confidence),
            "probabilities": {
                (class_names[j] if class_names else str(j)): float(probs[j])
                for j in range(len(probs))
            },
        }
        results.append(sample_result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="ECG inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.checkpoint}")
    lit_model = ECGLitModel.load_from_checkpoint(args.checkpoint, map_location="cpu")

    print(f"Loading ECG data from: {args.input}")
    signal = load_ecg_signal(args.input)
    print(f"Signal shape: {signal.shape}")

    results = predict(lit_model, signal, class_names=args.class_names)

    print(f"\n=== Predictions ({len(results)} samples) ===\n")
    for r in results:
        print(f"Sample {r['sample_index']}:")
        print(f"  Prediction: {r['predicted_class']} (confidence: {r['confidence']:.4f})")
        print("  Probabilities:")
        for cls, prob in r["probabilities"].items():
            marker = " <--" if cls == r["predicted_class"] else ""
            print(f"    {cls}: {prob:.4f}{marker}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
