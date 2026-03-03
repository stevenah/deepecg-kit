"""Explore and visualize ECG datasets.

Load a dataset, inspect metadata, class distribution, signal shapes,
and optionally plot sample ECG waveforms.

Usage:
    python examples/explore_dataset.py --dataset af-classification --download
    python examples/explore_dataset.py --dataset af-classification --num-samples 5 --save-dir plots/
    python examples/explore_dataset.py --list
"""

import argparse
import random
from pathlib import Path

import numpy as np

from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation.visualization import plot_ecg_signals
from deepecgkit.registry import get_dataset, get_dataset_info, get_dataset_names


def parse_args():
    parser = argparse.ArgumentParser(description="Explore ECG datasets")
    parser.add_argument("--dataset", type=str, default="af-classification")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_datasets")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_all_datasets():
    """Print info for all registered datasets."""
    print("=== Available Datasets ===\n")
    for name in get_dataset_names():
        info = get_dataset_info(name)
        print(f"  {name}")
        print(f"    Input channels: {info['input_channels']}")
        print(f"    Num classes: {info.get('num_classes', 'N/A')}")
        if info.get("description"):
            print(f"    Description: {info['description']}")
        print()


def main():
    args = parse_args()

    if args.list_datasets:
        list_all_datasets()
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_info = get_dataset_info(args.dataset)
    print(f"=== Dataset: {args.dataset} ===")
    print(f"  Input channels: {dataset_info['input_channels']}")
    print(f"  Num classes: {dataset_info.get('num_classes', 'N/A')}")
    if dataset_info.get("description"):
        print(f"  Description: {dataset_info['description']}")

    dataset_class = get_dataset(args.dataset)
    data_module = ECGDataModule(
        dataset_class=dataset_class,
        batch_size=args.batch_size,
        val_split=0.2,
        test_split=0.1,
        seed=args.seed,
        stratify=True,
        download=args.download,
    )
    data_module.setup(stage="fit")

    print("\n--- Metadata ---")
    data_module.print_metadata()

    dataset = data_module.dataset
    print("\n--- Split Sizes ---")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train: {len(data_module.train_dataset)}")
    print(f"  Val: {len(data_module.val_dataset)}")
    print(f"  Test: {len(data_module.test_dataset)}")

    signal, label = dataset[0]
    print("\n--- Signal Info ---")
    print(f"  Signal shape: {tuple(signal.shape)}")
    print(f"  Signal dtype: {signal.dtype}")
    print(f"  Label example: {label}")

    if hasattr(dataset, "get_class_distribution"):
        dist = dataset.get_class_distribution()
        print("\n--- Class Distribution ---")
        total = sum(dist.values())
        for cls_name, count in dist.items():
            pct = count / total * 100
            print(f"  {cls_name}: {count} ({pct:.1f}%)")

    class_names = getattr(dataset_class, "CLASS_LABELS", None)

    if args.num_samples > 0:
        print(f"\n--- Plotting {args.num_samples} sample(s) ---")
        indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

        for idx in indices:
            signal, label = dataset[idx]
            signal_np = signal.numpy()

            label_str = class_names[int(label)] if class_names else str(int(label))
            title = f"Sample {idx} (label: {label_str})"

            save_path = None
            if args.save_dir:
                save_dir = Path(args.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(save_dir / f"sample_{idx}.png")

            plot_ecg_signals(
                signal_np,
                title=title,
                save_path=save_path,
            )

            if save_path:
                print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
