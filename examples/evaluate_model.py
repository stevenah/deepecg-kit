"""Evaluate a trained model with comprehensive metrics and visualizations.

Loads a checkpoint, runs evaluation on the test set, computes classification metrics
with bootstrap confidence intervals, and generates diagnostic plots.

Usage:
    python examples/evaluate_model.py --checkpoint runs/kanres-af/checkpoints/best.pt --model kanres --dataset af-classification
    python examples/evaluate_model.py --checkpoint model.pt --model afmodel --dataset af-classification --output-dir results/
"""

import argparse
from pathlib import Path

from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation import ECGEvaluator
from deepecgkit.evaluation.metrics import confusion_matrix_analysis
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_roc_curve,
)
from deepecgkit.registry import get_dataset, get_dataset_info, get_model
from deepecgkit.training import ECGTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="af-classification")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_info = get_dataset_info(args.dataset)
    dataset_class = get_dataset(args.dataset)
    input_channels = dataset_info["input_channels"]
    num_classes = dataset_info["num_classes"]

    model_class = get_model(args.model)
    model = model_class(input_channels=input_channels, output_size=num_classes)

    print(f"Loading checkpoint: {args.checkpoint}")
    trainer = ECGTrainer.load_checkpoint(args.checkpoint, model=model, device=args.device)

    data_module = ECGDataModule(
        dataset_class=dataset_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    data_module.setup(stage="test")

    trainer.test(data_module)

    y_pred, y_true, y_prob = trainer.get_test_results()
    if y_pred is None:
        print("No test results available.")
        return

    evaluator = ECGEvaluator(
        metrics=["accuracy", "precision", "recall", "f1"],
        task_type="classification",
    )
    metrics = evaluator.evaluate(
        model=y_pred,
        test_data=(y_pred, y_true),
    )

    print("\n=== Evaluation Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    cm_results = confusion_matrix_analysis(y_true, y_pred)
    print("\n=== Per-Class Analysis ===")
    for key, value in cm_results.items():
        print(f"  {key}: {value}")

    print(f"\nRunning bootstrap evaluation ({args.n_bootstrap} samples)...")
    bootstrap_results = evaluator.bootstrap_evaluate(
        model=y_pred,
        test_data=(y_pred, y_true),
        n_bootstrap=args.n_bootstrap,
    )

    print("\n=== Bootstrap Confidence Intervals (95%) ===")
    for metric_name, stats in bootstrap_results.items():
        print(
            f"  {metric_name}: {stats['mean']:.4f} "
            f"[{stats['lower_ci']:.4f}, {stats['upper_ci']:.4f}]"
        )

    report = evaluator.generate_report(
        model=y_pred,
        test_data=(y_pred, y_true),
        save_path=str(output_path / "evaluation_report.csv"),
    )
    print(f"\n{report.to_string(index=False)}")

    class_names = getattr(dataset_class, "CLASS_LABELS", None)

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        title="Test Set Confusion Matrix",
        save_path=str(output_path / "confusion_matrix.png"),
    )

    if y_prob is not None and dataset_info.get("num_classes", 0) == 2:
        plot_roc_curve(
            y_true,
            y_prob[:, 1],
            title="Test Set ROC Curve",
            save_path=str(output_path / "roc_curve.png"),
        )

    if y_prob is not None:
        plot_calibration_curve(
            y_true,
            y_prob,
            class_names=class_names,
            save_dir=str(output_path),
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
