"""Feature extraction using pretrained ECG models.

Demonstrates using any model's extract_features() method to obtain embeddings
for downstream tasks like clustering, visualization, or transfer learning.

Usage:
    python examples/feature_extraction.py
    python examples/feature_extraction.py --model resnet --input-channels 1 --output-size 4
    python examples/feature_extraction.py --model kanres --batch-size 16 --signal-length 3000
"""

import argparse
import time

import numpy as np
import torch

from deepecgkit.registry import get_model, get_model_names


def extract_features_batch(
    model: torch.nn.Module,
    signals: torch.Tensor,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Extract features from a batch of ECG signals."""
    model = model.to(device)
    model.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            batch = signals[i : i + batch_size].to(device)
            features = model.extract_features(batch)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="ECG feature extraction")
    parser.add_argument("--model", type=str, default="kanres")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--signal-length", type=int, default=3000)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--all-models", action="store_true")
    return parser.parse_args()


def extract_single_model(args, model_name: str):
    """Extract features using a single model and print stats."""
    model_class = get_model(model_name)
    model = model_class(
        input_channels=args.input_channels,
        output_size=args.output_size,
    )

    print(f"\nModel: {model_name}")
    print(f"  Feature dimension: {model.feature_dim}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    signals = torch.randn(args.num_samples, args.input_channels, args.signal_length)

    start = time.perf_counter()
    features = extract_features_batch(
        model, signals, batch_size=args.batch_size, device=args.device
    )
    elapsed = time.perf_counter() - start

    print(f"  Input shape: {signals.shape}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature mean: {features.mean():.4f}")
    print(f"  Feature std: {features.std():.4f}")
    print(f"  Feature min: {features.min():.4f}")
    print(f"  Feature max: {features.max():.4f}")
    print(f"  Extraction time: {elapsed:.3f}s ({elapsed / args.num_samples * 1000:.1f}ms/sample)")

    return features


def main():
    args = parse_args()

    if args.all_models:
        print("Extracting features from all registered models:")
        print(f"  Signal shape: ({args.num_samples}, {args.input_channels}, {args.signal_length})")

        results = {}
        for model_name in get_model_names():
            try:
                features = extract_single_model(args, model_name)
                results[model_name] = features
            except Exception as e:
                print(f"\n  {model_name}: FAILED - {e}")

        print("\n=== Feature Dimension Summary ===")
        print(f"{'Model':<20} {'Feature Dim':<15} {'Shape'}")
        print("-" * 55)
        for name, feats in results.items():
            print(f"{name:<20} {feats.shape[1]:<15} {feats.shape}")
    else:
        extract_single_model(args, args.model)


if __name__ == "__main__":
    main()
