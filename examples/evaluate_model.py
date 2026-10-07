"""Evaluate a trained model with comprehensive metrics and visualizations.

Loads a checkpoint, runs evaluation on the test set, computes classification metrics
with bootstrap confidence intervals, and generates diagnostic plots.

The dataset, its sampling rate, its labelling (four-class or binary) and the
split are taken from the checkpoint, so the model is tested on the samples it
never saw, prepared the way its training data was. `deepecg evaluate` does the
same and also covers multi-label models and the remaining dataset options.

Usage:
    python examples/evaluate_model.py --checkpoint runs/kanres-af/checkpoints/epoch=07-val_loss=0.4123.pt
    python examples/evaluate_model.py --checkpoint model.pt --dataset af-classification --output-dir results/
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
from deepecgkit.models import build_model_from_checkpoint
from deepecgkit.registry import get_dataset
from deepecgkit.training import ECGTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name, only needed for checkpoints that do not record it",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to evaluate on (default: the one the checkpoint was trained on)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _ = build_model_from_checkpoint(args.checkpoint, name=args.model)
    trainer = ECGTrainer.load_checkpoint(args.checkpoint, model=model, device=args.device)

    data_config = trainer.data_config
    if data_config.get("multi_label"):
        raise SystemExit(
            "This example covers single-label models. For a multi-label checkpoint use "
            "`deepecg evaluate --checkpoint ...`."
        )

    # Rebuild the data the way the checkpoint was trained: same sampling rate,
    # labelling and split, so "test" means the samples the model never saw.
    dataset_name = args.dataset or data_config.get("dataset") or "af-classification"
    preprocessing = data_config.get("preprocessing") or {}
    dataset_kwargs = {}
    if data_config.get("binary_classification"):
        dataset_kwargs["binary_classification"] = True
    data_module = ECGDataModule(
        dataset_class=get_dataset(dataset_name),
        sampling_rate=preprocessing.get("sampling_rate"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        dataset_kwargs=dataset_kwargs,
        **(data_config.get("split") or {}),
    )
    data_module.setup(stage="test")

    trainer.test(data_module)

    y_pred, y_true, y_prob = trainer.get_test_results()
    if y_pred is None:
        print("No test results available.")
        return

    # The evaluator takes predictions computed elsewhere in place of a model;
    # the probabilities are what AUC and the calibration rows are computed from.
    evaluator = ECGEvaluator(
        metrics=["accuracy", "precision", "recall", "f1", "auc"],
        task_type="classification",
    )
    metrics = evaluator.evaluate(y_pred, y_true, y_scores=y_prob)

    print("\n=== Evaluation Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    cm_results = confusion_matrix_analysis(y_true, y_pred)
    print("\n=== Per-Class Analysis ===")
    for key, value in cm_results.items():
        print(f"  {key}: {value}")

    print(f"\nRunning bootstrap evaluation ({args.n_bootstrap} samples)...")
    bootstrap_results = evaluator.bootstrap_evaluate(
        y_pred, y_true, n_bootstrap=args.n_bootstrap, y_scores=y_prob
    )

    print("\n=== Bootstrap Confidence Intervals (95%) ===")
    for metric_name, stats in bootstrap_results.items():
        print(
            f"  {metric_name}: {stats['mean']:.4f} "
            f"[{stats['lower_ci']:.4f}, {stats['upper_ci']:.4f}]"
        )

    report = evaluator.generate_report(
        y_pred,
        y_true,
        save_path=str(output_path / "evaluation_report.csv"),
        y_scores=y_prob,
    )
    print(f"\n{report.to_string(index=False)}")

    class_names = data_config.get("class_names")

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        title="Test Set Confusion Matrix",
        save_path=str(output_path / "confusion_matrix.png"),
    )

    if y_prob is not None and y_prob.shape[1] == 2:
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
