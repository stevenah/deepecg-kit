"""Run inference on a raw ECG recording using a trained checkpoint.

Supports .npy, .csv, and WFDB (.dat/.hea) input formats. The recording is put
through the preprocessing recorded in the checkpoint (resampling, segmenting,
normalising) before it reaches the model, and each recording gets one
prediction averaged over its segments.

Usage:
    python examples/inference.py --checkpoint runs/my-run/checkpoints/best-epoch.pt --input ecg.npy
    python examples/inference.py --checkpoint model.pt --input ecg.csv --sampling-rate 250
    python examples/inference.py --checkpoint model.pt --input record.hea --output predictions.json
"""

import argparse
import json

import numpy as np
import torch

from deepecgkit.datasets import load_recordings, prepare_recording
from deepecgkit.models import build_model_from_checkpoint


def predict(
    model: torch.nn.Module,
    segments: np.ndarray,
    class_names: list | None = None,
) -> dict:
    """Predict one recording from its preprocessed segments."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(segments, dtype=torch.float32))
        probabilities = torch.softmax(logits, dim=1).mean(dim=0).numpy()

    predicted_index = int(probabilities.argmax())
    names = class_names or [str(i) for i in range(len(probabilities))]
    return {
        "predicted_class": names[predicted_index],
        "predicted_index": predicted_index,
        "confidence": float(probabilities[predicted_index]),
        "segments": len(segments),
        "probabilities": {name: float(p) for name, p in zip(names, probabilities, strict=False)},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="ECG inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name, only needed for checkpoints that do not record it",
    )
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Sampling rate of the input (read from the header for WFDB files)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint = build_model_from_checkpoint(args.checkpoint, name=args.model)
    data_config = checkpoint.get("data_config") or {}
    preprocessing = data_config.get("preprocessing")
    class_names = data_config.get("class_names")
    input_channels = (checkpoint.get("model_config") or {}).get("input_channels")
    if preprocessing is None:
        print("Warning: this checkpoint records no preprocessing; the input is used as it is.")

    print(f"Loading ECG data from: {args.input}")
    recordings, file_rate, _ = load_recordings(args.input, input_channels)
    sampling_rate = args.sampling_rate if args.sampling_rate is not None else file_rate

    results = []
    for index, recording in enumerate(recordings):
        prepared = prepare_recording(recording, preprocessing, sampling_rate)
        if index == 0 and prepared.notes:
            print(f"Preprocessing: {'; '.join(prepared.notes)}")
        result = predict(model, prepared.segments, class_names=class_names)
        result["recording_index"] = index
        results.append(result)

    print(f"\n=== Predictions ({len(results)} recordings) ===\n")
    for r in results:
        print(f"Recording {r['recording_index']}:")
        print(
            f"  Prediction: {r['predicted_class']} "
            f"(confidence: {r['confidence']:.4f}, {r['segments']} segment(s))"
        )
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
