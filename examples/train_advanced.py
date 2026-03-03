"""Advanced training with pretrained weights, fine-tuning, and post-training visualization.

Demonstrates loading pretrained weights, freezing the backbone for transfer learning,
CSV logging, and generating evaluation plots after training.

Usage:
    python examples/train_advanced.py --weights kanres-af-30s --freeze-backbone --epochs 20
    python examples/train_advanced.py --model kanres --dataset af-classification --download
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation.metrics import (
    calculate_classification_metrics,
    confusion_matrix_analysis,
)
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_training_curves,
)
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGTrainer
from deepecgkit.utils.weights import load_pretrained_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced ECG model training")
    parser.add_argument("--model", type=str, default="kanres")
    parser.add_argument("--dataset", type=str, default="af-classification")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
        val_split=0.2,
        test_split=0.1,
        seed=args.seed,
        stratify=True,
        download=args.download,
    )
    data_module.setup(stage="fit")
    data_module.print_metadata()

    model_class = get_model(args.model)
    model = model_class(input_channels=input_channels, output_size=num_classes)

    if args.weights:
        print(f"\nLoading pretrained weights: {args.weights}")
        state_dict = load_pretrained_weights(args.weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    if args.freeze_backbone:
        print("Freezing backbone parameters (training classifier head only)")
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name and "head" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    train_config = {
        "learning_rate": args.lr,
        "scheduler": {"factor": 0.5, "patience": 5},
        "binary_classification": num_classes == 2,
        "task_type": "classification",
    }

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("runs") / f"{timestamp}-{args.model}-{args.dataset}"
    else:
        output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_dir = output_path / "training_logs"

    trainer = ECGTrainer(model=model, train_config=train_config, device=args.device)
    trainer.fit(
        data_module,
        epochs=args.epochs,
        early_stopping_patience=10,
        checkpoint_dir=str(output_path / "checkpoints"),
        log_dir=str(log_dir),
        gradient_clip_val=1.0,
    )
    trainer.test(data_module)

    y_pred, y_true, y_prob = trainer.get_test_results()
    if y_pred is not None:
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_confusion_matrix(
            y_true,
            y_pred,
            title=f"{args.model} Confusion Matrix",
            save_path=str(plots_dir / "confusion_matrix.png"),
        )

        plot_calibration_curve(
            y_true,
            y_prob,
            save_dir=str(plots_dir),
        )

        metrics_path = log_dir / "metrics.csv"
        if metrics_path.exists():
            plot_training_curves(
                str(metrics_path),
                save_dir=str(plots_dir),
            )

        cm_results = confusion_matrix_analysis(y_true, y_pred)
        print("\nPer-class metrics:")
        for key, value in cm_results.items():
            print(f"  {key}: {value}")

        cls_metrics = calculate_classification_metrics(y_true, y_prob)
        metrics_df = pd.DataFrame(
            [
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "accuracy": cls_metrics.get("accuracy", float("nan")),
                    "precision": cls_metrics.get("precision", float("nan")),
                    "recall": cls_metrics.get("recall", float("nan")),
                    "f1": cls_metrics.get("f1", float("nan")),
                    "auroc": cls_metrics.get("auc", float("nan")),
                    "mcc": cls_metrics.get("mcc", float("nan")),
                }
            ]
        )
        metrics_csv_path = output_path / "classification_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Classification metrics saved to {metrics_csv_path}")

        print(f"\nPlots saved to: {plots_dir}")

    if trainer.best_checkpoint_path:
        print(f"Best checkpoint: {trainer.best_checkpoint_path}")


if __name__ == "__main__":
    main()
