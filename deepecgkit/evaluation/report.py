"""Writing evaluation artifacts: metrics CSV, predictions, and plots.

This is the single evaluation-artifact writer. The train command uses it for
its post-training test pass, and `deepecg evaluate -o` writes the same set,
so the analysis pipeline can consume either kind of run directory:

- classification_metrics.csv (one wide row, calibration columns included)
- predictions.txt
- confusion_matrix.png, calibration_curve.png, prediction_distribution.png

The same numbers are always reported to the console; passing no output
directory skips the files and nothing else.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from deepecgkit.cli.logger import CLILogger
from deepecgkit.evaluation.calibration import calibration_report
from deepecgkit.evaluation.metrics import (
    DEFAULT_CALIBRATION_BINS,
    calculate_classification_metrics,
    confusion_matrix_analysis,
    resolve_average,
)
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_multilabel_calibration_curve,
    plot_multilabel_confusion_matrices,
)


def _calibration_columns(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    logger: CLILogger,
    val_targets: np.ndarray | None = None,
    val_logits: np.ndarray | None = None,
    test_logits: np.ndarray | None = None,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """Compute calibration for the metrics row and report it to the console.

    The temperature is fitted on the validation split and every number is
    reported on the split passed as y_true, so the scaled figures are honest
    held-out ones. Without a validation split the scaled columns come back as
    NaN and the row is still well formed.

    Args:
        y_true: True labels for the reported split
        y_prob: Predicted probabilities for that split
        logger: CLI logger
        val_targets: True labels for the validation split
        val_logits: Raw model outputs for the validation split
        test_logits: Raw model outputs for the reported split
        n_bins: Number of bins for the binned metrics

    Returns:
        The calibration columns to merge into classification_metrics.csv.
    """
    report = calibration_report(
        y_true,
        y_prob,
        n_bins=n_bins,
        fit_targets=val_targets,
        fit_logits=val_logits,
        eval_logits=test_logits,
    )

    scaled = not np.isnan(report["temperature"])
    logger.info("")
    logger.info(f"  Calibration ({n_bins} bins):")
    if scaled:
        logger.info(f"  {'Metric':<16} {'Before':>10} {'After T':>10}")
        logger.info(f"  {'-' * 38}")
        for name, label in (
            ("ece", "ECE"),
            ("adaptive_ece", "Adaptive ECE"),
            ("mce", "MCE"),
            ("brier", "Brier"),
        ):
            logger.info(f"  {label:<16} {report[name]:>10.4f} {report[name + '_temp']:>10.4f}")
        logger.info(f"  {'Temperature':<16} {report['temperature']:>21.4f}")
    else:
        for name, label in (
            ("ece", "ECE"),
            ("adaptive_ece", "Adaptive ECE"),
            ("mce", "MCE"),
            ("brier", "Brier"),
        ):
            logger.info(f"  {label:<16} {report[name]:>10.4f}")
        logger.info("  No validation split was supplied, so no temperature was fitted.")

    return report


def save_single_label_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    class_names: list | None,
    output_path: Path | None,
    logger: CLILogger,
    val_targets: np.ndarray | None = None,
    val_logits: np.ndarray | None = None,
    test_logits: np.ndarray | None = None,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> None:
    """Report single-label results, writing the artifact set when output_path is given."""
    display_names = class_names or [f"Class {i}" for i in range(y_prob.shape[1])]
    if output_path is not None:
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names=class_names,
            title=f"{model_name} on {dataset_name}",
            save_path=str(output_path / "confusion_matrix.png"),
        )
        logger.info(f"Confusion matrix saved to {output_path / 'confusion_matrix.png'}")

        plot_calibration_curve(
            y_true,
            y_prob,
            class_names=class_names,
            title=f"{model_name} on {dataset_name} - Calibration",
            save_dir=str(output_path),
        )
        logger.info(f"Calibration curve saved to {output_path / 'calibration_curve.png'}")
        logger.info(
            f"Prediction distribution saved to {output_path / 'prediction_distribution.png'}"
        )

        predictions_path = output_path / "predictions.txt"
        with open(predictions_path, "w") as f:
            header = "index\ttrue_label\tpredicted_label"
            for name in display_names:
                header += f"\tprob_{name}"
            f.write(header + "\n")
            for idx in range(len(y_true)):
                row = f"{idx}\t{y_true[idx]}\t{y_pred[idx]}"
                for c in range(y_prob.shape[1]):
                    row += f"\t{y_prob[idx, c]:.6f}"
                f.write(row + "\n")
        logger.info(f"Test predictions saved to {predictions_path}")

    cm_results = confusion_matrix_analysis(y_true, y_pred)
    overall_acc = (y_pred == y_true).mean()

    cls_metrics = calculate_classification_metrics(y_true, y_prob)

    logger.info("\nTest Results:")
    logger.info("=" * 60)
    logger.info(f"  Overall Accuracy: {overall_acc:.4f}")
    logger.info(f"  Macro Precision:  {cm_results['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:     {cm_results['macro_recall']:.4f}")
    logger.info(f"  Macro F1:         {cm_results['macro_f1']:.4f}")
    logger.info(f"  AUROC:            {cls_metrics.get('auc', np.nan):.4f}")
    logger.info("")
    logger.info("  Per-Class Results:")
    logger.info(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    logger.info(f"  {'-' * 45}")

    display_names = class_names or [f"Class {i}" for i in range(len(cm_results["per_class_f1"]))]
    for i, name in enumerate(display_names):
        logger.info(
            f"  {name:<15} "
            f"{cm_results['per_class_precision'][i]:>10.4f} "
            f"{cm_results['per_class_recall'][i]:>10.4f} "
            f"{cm_results['per_class_f1'][i]:>10.4f}"
        )
    logger.info("=" * 60)

    calibration = _calibration_columns(
        y_true, y_prob, logger, val_targets, val_logits, test_logits, n_bins=n_bins
    )
    logger.info("=" * 60)

    if output_path is None:
        return

    averaging = resolve_average(y_true)
    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "dataset": dataset_name,
                "accuracy": cls_metrics.get("accuracy", np.nan),
                "precision": cls_metrics.get("precision", np.nan),
                "recall": cls_metrics.get("recall", np.nan),
                "f1": cls_metrics.get("f1", np.nan),
                "auroc": cls_metrics.get("auc", np.nan),
                "mcc": cls_metrics.get("mcc", np.nan),
                "average": averaging,
                **calibration,
            }
        ]
    )
    metrics_csv_path = output_path / "classification_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Classification metrics saved to {metrics_csv_path}")


def save_multi_label_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    class_names: list | None,
    output_path: Path | None,
    logger: CLILogger,
    val_targets: np.ndarray | None = None,
    val_logits: np.ndarray | None = None,
    test_logits: np.ndarray | None = None,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> None:
    """Report multi-label results, writing the artifact set when output_path is given."""

    num_classes = y_true.shape[1]
    display_names = class_names or [f"Class {i}" for i in range(num_classes)]

    if output_path is not None:
        # Per-class predictions file
        predictions_path = output_path / "predictions.txt"
        with open(predictions_path, "w") as f:
            header = "index"
            for name in display_names:
                header += f"\ttrue_{name}\tpred_{name}\tprob_{name}"
            f.write(header + "\n")
            for idx in range(len(y_true)):
                row = str(idx)
                for c in range(num_classes):
                    row += f"\t{y_true[idx, c]}\t{y_pred[idx, c]}\t{y_prob[idx, c]:.6f}"
                f.write(row + "\n")
        logger.info(f"Test predictions saved to {predictions_path}")

        plot_multilabel_confusion_matrices(
            y_true,
            y_pred,
            class_names=display_names,
            title=f"{model_name} on {dataset_name} - Per-Class Confusion Matrices",
            save_path=str(output_path / "confusion_matrix.png"),
        )
        logger.info(f"Confusion matrices saved to {output_path / 'confusion_matrix.png'}")

        plot_multilabel_calibration_curve(
            y_true,
            y_prob,
            class_names=display_names,
            title=f"{model_name} on {dataset_name} - Calibration",
            save_path=str(output_path / "calibration_curve.png"),
        )
        logger.info(f"Calibration curve saved to {output_path / 'calibration_curve.png'}")

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    sample_acc = (y_pred == y_true).all(axis=1).mean()
    label_acc = (y_pred == y_true).mean()

    try:
        auroc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        auroc = np.nan

    logger.info("\nTest Results (Multi-Label):")
    logger.info("=" * 60)
    logger.info(f"  Exact Match Accuracy: {sample_acc:.4f}")
    logger.info(f"  Label Accuracy:       {label_acc:.4f}")
    logger.info(f"  Macro Precision:      {macro_p:.4f}")
    logger.info(f"  Macro Recall:         {macro_r:.4f}")
    logger.info(f"  Macro F1:             {macro_f1:.4f}")
    logger.info(f"  Macro AUROC:          {auroc:.4f}")
    logger.info("")
    logger.info("  Per-Class Results:")
    logger.info(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    logger.info(f"  {'-' * 45}")

    for i, name in enumerate(display_names):
        logger.info(f"  {name:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")

    calibration = _calibration_columns(
        y_true, y_prob, logger, val_targets, val_logits, test_logits, n_bins=n_bins
    )
    logger.info("=" * 60)

    if output_path is None:
        return

    metrics_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "dataset": dataset_name,
                "exact_match_accuracy": sample_acc,
                "label_accuracy": label_acc,
                "precision": macro_p,
                "recall": macro_r,
                "f1": macro_f1,
                "auroc": auroc,
                **calibration,
            }
        ]
    )
    metrics_csv_path = output_path / "classification_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Classification metrics saved to {metrics_csv_path}")


def save_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    class_names: list | None,
    output_path: Path | None,
    logger: CLILogger,
    val_targets: np.ndarray | None = None,
    val_logits: np.ndarray | None = None,
    test_logits: np.ndarray | None = None,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> None:
    """Report an evaluation, dispatching on the label shape.

    The metrics are always logged. With an output_path the full artifact set
    is written there as well; with None the console report is all there is.
    """
    saver = save_multi_label_evaluation if y_true.ndim > 1 else save_single_label_evaluation
    saver(
        y_true,
        y_pred,
        y_prob,
        model_name,
        dataset_name,
        class_names,
        output_path,
        logger,
        val_targets=val_targets,
        val_logits=val_logits,
        test_logits=test_logits,
        n_bins=n_bins,
    )
