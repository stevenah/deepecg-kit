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
from deepecgkit.utils.device import resolve_device

BATCH_SIZE = 32


def segment_probabilities(
    model: torch.nn.Module, segments: np.ndarray, multi_label: bool
) -> torch.Tensor:
    """Turn the model's logits into one row of class probabilities per segment.

    The three kinds of head need different treatment: independent sigmoids for
    a multi-label model, a single logit for a binary one (the probability of
    the positive class), and a softmax over the classes otherwise.
    """
    device = next(model.parameters()).device
    rows = []
    with torch.no_grad():
        # A long recording is thousands of segments, so they go through in batches.
        for start in range(0, len(segments), BATCH_SIZE):
            batch = torch.as_tensor(segments[start : start + BATCH_SIZE], dtype=torch.float32)
            logits = model(batch.to(device))
            if multi_label:
                rows.append(torch.sigmoid(logits))
            elif logits.shape[-1] == 1:
                positive = torch.sigmoid(logits.squeeze(-1))
                rows.append(torch.stack([1 - positive, positive], dim=-1))
            else:
                rows.append(torch.softmax(logits, dim=-1))
    return torch.cat(rows).cpu()


def predict(
    model: torch.nn.Module,
    segments: np.ndarray,
    class_names: list | None = None,
    multi_label: bool = False,
) -> dict:
    """Predict one recording from its preprocessed segments."""
    model.eval()
    probabilities = segment_probabilities(model, segments, multi_label).mean(dim=0).numpy()

    names = class_names or [str(i) for i in range(len(probabilities))]
    result = {
        "segments": len(segments),
        "probabilities": {name: float(p) for name, p in zip(names, probabilities, strict=False)},
    }
    if multi_label:
        # Any number of classes can be present at once, including none.
        result["predicted_classes"] = [
            name for name, p in zip(names, probabilities, strict=False) if p > 0.5
        ]
    else:
        predicted_index = int(probabilities.argmax())
        result["predicted_class"] = names[predicted_index]
        result["predicted_index"] = predicted_index
        result["confidence"] = float(probabilities[predicted_index])
    return result


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
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint = build_model_from_checkpoint(args.checkpoint, name=args.model)
    model.to(resolve_device(args.device))
    data_config = checkpoint.get("data_config") or {}
    preprocessing = data_config.get("preprocessing")
    class_names = data_config.get("class_names")
    train_config = checkpoint.get("train_config") or {}
    multi_label = bool(data_config.get("multi_label") or train_config.get("multi_label"))
    input_channels = (checkpoint.get("model_config") or {}).get("input_channels")
    if preprocessing is None:
        print("Warning: this checkpoint records no preprocessing; the input is used as it is.")

    print(f"Loading ECG data from: {args.input}")
    recordings, file_rate, lead_notes = load_recordings(args.input, input_channels)
    sampling_rate = args.sampling_rate if args.sampling_rate is not None else file_rate

    results = []
    for index, recording in enumerate(recordings):
        prepared = prepare_recording(recording, preprocessing, sampling_rate)
        if index == 0 and (lead_notes or prepared.notes):
            print(f"Preprocessing: {'; '.join([*lead_notes, *prepared.notes])}")
        result = predict(model, prepared.segments, class_names=class_names, multi_label=multi_label)
        result["recording_index"] = index
        results.append(result)

    print(f"\n=== Predictions ({len(results)} recordings) ===\n")
    for r in results:
        print(f"Recording {r['recording_index']}:")
        if multi_label:
            predicted = r["predicted_classes"]
            print(f"  Prediction: {', '.join(predicted) or 'none'} ({r['segments']} segment(s))")
        else:
            predicted = [r["predicted_class"]]
            print(
                f"  Prediction: {r['predicted_class']} "
                f"(confidence: {r['confidence']:.4f}, {r['segments']} segment(s))"
            )
        print("  Probabilities:")
        for cls, prob in r["probabilities"].items():
            marker = " <--" if cls in predicted else ""
            print(f"    {cls}: {prob:.4f}{marker}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
