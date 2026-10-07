"""Basic training workflow for deepecg-kit.

Train any registered model on any registered dataset with sensible defaults.

Usage:
    python examples/train_basic.py --download        # first run: fetches the dataset
    python examples/train_basic.py --model resnet --dataset af-classification --epochs 30
    python examples/train_basic.py --model tcn --dataset af-classification --batch-size 64
"""

import argparse
from pathlib import Path

from deepecgkit.cli import defaults
from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ECG model")
    parser.add_argument("--model", type=str, default="kanres")
    parser.add_argument("--dataset", type=str, default="af-classification")
    parser.add_argument("--epochs", type=int, default=defaults.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=defaults.LEARNING_RATE)
    parser.add_argument("--val-split", type=float, default=defaults.VAL_SPLIT)
    parser.add_argument("--test-split", type=float, default=defaults.TEST_SPLIT)
    parser.add_argument("--num-workers", type=int, default=defaults.NUM_WORKERS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    ECGTrainer.seed_everything(args.seed)

    dataset_info = get_dataset_info(args.dataset)
    input_channels = dataset_info["input_channels"]
    num_classes = dataset_info["num_classes"]

    dataset_class = get_dataset(args.dataset)
    data_module = ECGDataModule(
        dataset_class=dataset_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        stratify=True,
        download=args.download,
    )
    data_module.setup(stage="fit")
    data_module.print_metadata()

    model_class = get_model(args.model)
    model = model_class(input_channels=input_channels, output_size=num_classes)

    train_config = {
        "learning_rate": args.lr,
        "scheduler": {"factor": 0.5, "patience": 5},
        "task_type": "classification",
    }

    if args.output_dir is None:
        output_path = Path("runs") / f"{args.model}-{args.dataset}"
    else:
        output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer = ECGTrainer(model=model, train_config=train_config, device=args.device)
    trainer.fit(
        data_module,
        epochs=args.epochs,
        early_stopping_patience=10,
        checkpoint_dir=str(output_path / "checkpoints"),
    )
    results = trainer.test(data_module)
    print(f"\nTest loss: {results['test_loss']:.4f}  Test accuracy: {results['test_acc']:.4f}")

    if trainer.best_checkpoint_path:
        print(f"\nBest checkpoint: {trainer.best_checkpoint_path}")
        print(f"Best val_loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
