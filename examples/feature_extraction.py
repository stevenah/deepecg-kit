"""Feature extraction using trained ECG models.

Obtains deep embeddings for downstream tasks such as clustering, visualisation,
or transfer learning. The default run uses synthetic signals so it works offline;
pass --dataset to extract over a real dataset and fit a linear probe.

Usage:
    python examples/feature_extraction.py
    python examples/feature_extraction.py --model resnet --batch-size 16
    python examples/feature_extraction.py --all-models
    python examples/feature_extraction.py --dataset af-classification --checkpoint runs/best.ckpt
"""

import argparse
import time

import numpy as np
import torch

from deepecgkit import FeatureExtractor
from deepecgkit.datasets import ECGDataModule
from deepecgkit.models.factory import build_model
from deepecgkit.registry import get_dataset, get_model_names

MINIMUM_LENGTHS = {"afmodel": 7971, "deep-res-cnn": 127, "mamba": 50, "medformer": 50}

MAXIMUM_LENGTHS = {"dualnet": 5000, "medformer": 5009, "transformer": 5000, "mamba": 25049}


def signal_length_for(model_name: str, requested: int) -> int:
    """Clamp a requested signal length into a model's supported range."""
    length = max(requested, MINIMUM_LENGTHS.get(model_name, 1))
    return min(length, MAXIMUM_LENGTHS.get(model_name, length))


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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def extract_single_model(args, model_name: str) -> np.ndarray:
    """Extract features from synthetic signals and print summary statistics."""
    length = signal_length_for(model_name, args.signal_length)
    model = build_model(
        model_name, input_channels=args.input_channels, output_size=args.output_size
    )
    extractor = FeatureExtractor(
        model, model_name=model_name, batch_size=args.batch_size, device=args.device
    )

    signals = torch.randn(args.num_samples, args.input_channels, length)

    start = time.perf_counter()
    result = extractor.extract(signals)
    elapsed = time.perf_counter() - start

    features = result.features
    print(f"\nModel: {model_name}")
    print(f"  Feature dimension: {extractor.feature_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input shape: {tuple(signals.shape)}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature mean: {features.mean():.4f}")
    print(f"  Feature std: {features.std():.4f}")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Extraction time: {elapsed:.3f}s ({elapsed / args.num_samples * 1000:.1f}ms/sample)")

    return features


def compare_all_models(args) -> None:
    """Extract from every registered model and summarise the feature dimensions."""
    print("Extracting features from all registered models:")

    results = {}
    for model_name in get_model_names():
        try:
            results[model_name] = extract_single_model(args, model_name)
        except Exception as error:
            print(f"\n  {model_name}: FAILED - {error}")

    print("\n=== Feature Dimension Summary ===")
    print(f"{'Model':<24} {'Feature Dim':<14} {'Shape'}")
    print("-" * 58)
    for name, features in results.items():
        print(f"{name:<24} {features.shape[1]:<14} {features.shape}")


def linear_probe(args) -> None:
    """Extract over a real dataset and fit a linear probe on the embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    dataset_kwargs = {"data_dir": args.data_dir} if args.data_dir else {}
    data_module = ECGDataModule(
        dataset_class=get_dataset(args.dataset),
        batch_size=args.batch_size,
        download=True,
        dataset_kwargs=dataset_kwargs,
    )
    data_module.setup()

    sample_signal, _ = data_module.dataset[0]
    input_channels = int(sample_signal.shape[0])

    if args.checkpoint:
        extractor = FeatureExtractor.from_checkpoint(
            args.checkpoint,
            args.model,
            input_channels=input_channels,
            batch_size=args.batch_size,
            device=args.device,
            progress=True,
        )
    else:
        print("No --checkpoint given; using an untrained model, so accuracy will be near chance.")
        model = build_model(args.model, input_channels=input_channels, output_size=args.output_size)
        extractor = FeatureExtractor(
            model,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
            progress=True,
        )

    result = extractor.extract(data_module, split="all")
    print(f"\nExtracted {result.features.shape} features from {args.dataset}")

    if args.output:
        result.save(args.output)
        print(f"Saved to {args.output}")

    if result.labels is None or result.labels.ndim != 1:
        print("Skipping the linear probe: this dataset has no single-label targets.")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        result.features, result.labels, test_size=0.2, random_state=42, stratify=result.labels
    )
    probe = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    print(f"Linear probe accuracy: {probe.score(x_test, y_test):.4f}")


def main():
    args = parse_args()

    if args.dataset:
        linear_probe(args)
    elif args.all_models:
        compare_all_models(args)
    else:
        extract_single_model(args, args.model)


if __name__ == "__main__":
    main()
