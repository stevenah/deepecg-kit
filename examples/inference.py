"""Run inference on a raw ECG recording using a trained checkpoint.

Supports .npy, .csv, and WFDB (.dat/.hea) input formats. The recording is put
through the preprocessing recorded in the checkpoint (resampling, segmenting,
normalising) before it reaches the model, and each recording gets one
prediction averaged over its segments.

For a single recording, one call is enough:

    from deepecgkit import predict
    result = predict("ecg.npy", checkpoint="runs/my-run", sampling_rate=250)
    print(result.label, result.probabilities)

Usage:
    python examples/inference.py --checkpoint runs/my-run --input ecg.npy
    python examples/inference.py --checkpoint model.pt --input ecg.csv --sampling-rate 250
    python examples/inference.py --checkpoint model.pt --input record.hea --output predictions.json
"""

import argparse
import json

from deepecgkit import ECGPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="ECG inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Checkpoint file or run directory"
    )
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
    # Loaded once, the predictor serves every recording in the file.
    predictor = ECGPredictor.from_checkpoint(
        args.checkpoint, model_name=args.model, device=args.device
    )
    if predictor.preprocessing is None:
        print("Warning: this checkpoint records no preprocessing; the input is used as it is.")

    print(f"Loading ECG data from: {args.input}")
    predictions = predictor.predict_all(args.input, sampling_rate=args.sampling_rate)
    if predictions and predictions[0].notes:
        print(f"Preprocessing: {'; '.join(predictions[0].notes)}")

    results = []
    print(f"\n=== Predictions ({len(predictions)} recordings) ===\n")
    for index, prediction in enumerate(predictions):
        print(f"Recording {index}:")
        detail = f"{prediction.num_segments} segment(s)"
        if not prediction.multi_label:
            detail = f"confidence: {prediction.confidence:.4f}, {detail}"
        print(f"  Prediction: {prediction.label} ({detail})")
        print("  Probabilities:")
        for name, probability in prediction.probabilities.items():
            marker = " <--" if name in prediction.labels else ""
            print(f"    {name}: {probability:.4f}{marker}")
        print()

        result = {
            "recording_index": index,
            "segments": prediction.num_segments,
            "probabilities": prediction.probabilities,
        }
        if prediction.multi_label:
            # Any number of classes can be present at once, including none.
            result["predicted_classes"] = prediction.labels
        else:
            result["predicted_class"] = prediction.label
            result["confidence"] = prediction.confidence
        results.append(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
