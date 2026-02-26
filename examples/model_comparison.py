"""Compare all registered model architectures.

Instantiates every model from the registry and compares parameter counts,
feature dimensions, and inference speed on synthetic input.

Usage:
    python examples/model_comparison.py
    python examples/model_comparison.py --input-channels 12 --output-size 5 --signal-length 5000
    python examples/model_comparison.py --device mps
"""

import argparse
import time

import torch

from deepecgkit.registry import get_model, get_model_info, get_model_names


def benchmark_model(
    model_name: str,
    input_channels: int,
    output_size: int,
    signal_length: int,
    device: str,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> dict:
    """Benchmark a single model on synthetic input."""
    model_class = get_model(model_name)
    model = model_class(input_channels=input_channels, output_size=output_size)
    model = model.to(device)
    model.eval()

    x = torch.randn(1, input_channels, signal_length, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    feature_dim = model.feature_dim

    with torch.no_grad():
        output = model(x)
        output_shape = tuple(output.shape)

        for _ in range(warmup_runs):
            model(x)

        if device != "cpu":
            torch.cuda.synchronize() if device == "cuda" else None

        start = time.perf_counter()
        for _ in range(num_runs):
            model(x)
        if device != "cpu" and "cuda" in device:
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_runs

    return {
        "name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "feature_dim": feature_dim,
        "output_shape": output_shape,
        "inference_ms": elapsed * 1000,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ECG model architectures")
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--output-size", type=int, default=4)
    parser.add_argument("--signal-length", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-runs", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== ECG Model Architecture Comparison ===")
    print(f"Input: ({args.input_channels}, {args.signal_length})")
    print(f"Output classes: {args.output_size}")
    print(f"Device: {args.device}")
    print()

    results = []
    model_names = get_model_names()

    for model_name in model_names:
        try:
            info = get_model_info(model_name)
            result = benchmark_model(
                model_name,
                args.input_channels,
                args.output_size,
                args.signal_length,
                args.device,
                num_runs=args.num_runs,
            )
            result["description"] = info.get("description", "")
            results.append(result)
            print(f"  {model_name}: OK ({result['total_params']:,} params)")
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")

    results.sort(key=lambda r: r["total_params"])

    header = f"{'Model':<20} {'Params':>12} {'Feature Dim':>12} {'Inference':>12} {'Output':>15}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        params_str = f"{r['total_params']:,}"
        inference_str = f"{r['inference_ms']:.2f}ms"
        output_str = str(r["output_shape"])
        print(
            f"{r['name']:<20} {params_str:>12} {r['feature_dim']:>12} "
            f"{inference_str:>12} {output_str:>15}"
        )

    print(f"\n{'Model':<20} {'Description'}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['description']}")


if __name__ == "__main__":
    main()
