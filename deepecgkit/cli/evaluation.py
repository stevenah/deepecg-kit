"""Evaluation artifact saving helpers for the CLI."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from deepecgkit.cli.logger import CLILogger
from deepecgkit.evaluation.metrics import (
    calculate_classification_metrics,
    confusion_matrix_analysis,
)
from deepecgkit.evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
)


def _save_single_label_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    class_names: Optional[list],
    output_path: Path,
    logger: CLILogger,
) -> None:
    """Save evaluation artifacts for single-label classification."""
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
    logger.info(f"Prediction distribution saved to {output_path / 'prediction_distribution.png'}")

    display_names = class_names or [f"Class {i}" for i in range(y_prob.shape[1])]
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

    logger.info("\nTest Results:")
    logger.info("=" * 60)
    logger.info(f"  Overall Accuracy: {overall_acc:.4f}")
    logger.info(f"  Macro Precision:  {cm_results['macro_precision']:.4f}")
    logger.info(f"  Macro Recall:     {cm_results['macro_recall']:.4f}")
    logger.info(f"  Macro F1:         {cm_results['macro_f1']:.4f}")
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

    cls_metrics = calculate_classification_metrics(y_true, y_prob)
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
            }
        ]
    )
    metrics_csv_path = output_path / "classification_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Classification metrics saved to {metrics_csv_path}")


def _save_multi_label_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    dataset_name: str,
    class_names: Optional[list],
    output_path: Path,
    logger: CLILogger,
) -> None:
    """Save evaluation artifacts for multi-label classification."""

    num_classes = y_true.shape[1]
    display_names = class_names or [f"Class {i}" for i in range(num_classes)]

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

    # Per-class binary confusion matrices
    ncols = min(num_classes, 3)
    nrows = (num_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if num_classes > 1 else [axes]

    for i in range(num_classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=axes_flat[i],
        )
        axes_flat[i].set_title(display_names[i])
        axes_flat[i].set_ylabel("True")
        axes_flat[i].set_xlabel("Predicted")

    for i in range(num_classes, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"{model_name} on {dataset_name} - Per-Class Confusion Matrices")
    fig.tight_layout()
    fig.savefig(str(output_path / "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices saved to {output_path / 'confusion_matrix.png'}")

    # Calibration curves (per-class, using multi-hot targets directly)
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if y_true[:, i].sum() == 0:
            continue
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_true[:, i],
                y_prob[:, i],
                n_bins=10,
                strategy="uniform",
            )
            plt.plot(mean_pred, fraction_pos, "s-", label=display_names[i])
        except ValueError:
            continue
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"{model_name} on {dataset_name} - Calibration")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path / "calibration_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
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
    logger.info("=" * 60)

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
            }
        ]
    )
    metrics_csv_path = output_path / "classification_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Classification metrics saved to {metrics_csv_path}")
