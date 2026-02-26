"""Basic training workflow for deepecg-kit.

Train any registered model on any registered dataset with sensible defaults.

Usage:
    python examples/train_basic.py
    python examples/train_basic.py --model resnet --dataset af-classification --epochs 30
    python examples/train_basic.py --model tcn --dataset af-classification --batch-size 64 --download
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from deepecgkit.datasets import ECGDataModule
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGLitModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ECG model")
    parser.add_argument("--model", type=str, default="kanres")
    parser.add_argument("--dataset", type=str, default="af-classification")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

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
        "binary_classification": num_classes == 2,
        "task_type": "classification",
    }
    lit_model = ECGLitModel(model=model, train_config=train_config)

    if args.output_dir is None:
        output_path = Path("runs") / f"{args.model}-{args.dataset}"
    else:
        output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename=f"{args.model}-{args.dataset}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=str(output_path),
    )

    trainer.fit(lit_model, data_module)
    trainer.test(lit_model, data_module)

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_loss: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
