"""
Visualization functions for ECG evaluation.
"""

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None or len(class_names) != len(labels):
        class_names = [str(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
):
    """Plot ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except ImportError:
        warnings.warn("Could not plot ROC curve - missing dependencies", stacklevel=2)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs True Values",
    save_path: Optional[str] = None,
):
    """Plot predictions vs true values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_ecg_signals(
    data: np.ndarray,
    sampling_rate: float = 500.0,
    leads: Optional[List[str]] = None,
    title: str = "ECG Signals",
    save_path: Optional[str] = None,
):
    """Plot ECG signals."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_leads = data.shape
    time = np.arange(n_samples) / sampling_rate

    if leads is None:
        leads = [f"Lead {i + 1}" for i in range(n_leads)]

    _fig, axes = plt.subplots(n_leads, 1, figsize=(12, 2 * n_leads), sharex=True)
    if n_leads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, data[:, i])
        ax.set_ylabel(f"{leads[i]} (mV)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    metrics_path: str,
    save_dir: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot training and validation loss and accuracy as separate figures.

    Args:
        metrics_path: Path to CSVLogger metrics.csv file.
        save_dir: Directory to save separate loss.png and accuracy.png files.
        save_path: Deprecated combined path; if provided and save_dir is not,
            saves loss plot to this path for backward compatibility.
    """
    df = pd.read_csv(metrics_path)

    has_acc = "train_acc" in df.columns

    resolved_dir = None
    if save_dir:
        resolved_dir = Path(save_dir)
    elif save_path:
        resolved_dir = Path(save_path).parent

    train_loss = df.dropna(subset=["train_loss"]).groupby("epoch")["train_loss"].mean()
    val_loss = df.dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss.index, train_loss.values, label="Train Loss")
    plt.plot(val_loss.index, val_loss.values, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if resolved_dir:
        plt.savefig(str(resolved_dir / "loss.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if has_acc:
        train_acc = df.dropna(subset=["train_acc"]).groupby("epoch")["train_acc"].mean()
        val_acc = df.dropna(subset=["val_acc"]).groupby("epoch")["val_acc"].mean()

        plt.figure(figsize=(10, 5))
        plt.plot(train_acc.index, train_acc.values, label="Train Acc")
        plt.plot(val_acc.index, val_acc.values, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if resolved_dir:
            plt.savefig(str(resolved_dir / "accuracy.png"), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_bins: int = 10,
    title: str = "Calibration Plot",
    save_dir: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot reliability diagram and prediction distribution as separate figures.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities (n_samples, n_classes).
        class_names: Optional list of class names.
        n_bins: Number of bins for calibration curve.
        title: Title for the calibration curve plot.
        save_dir: Directory to save calibration_curve.png and prediction_distribution.png.
        save_path: Deprecated combined path; if provided and save_dir is not,
            uses its parent directory for backward compatibility.
    """
    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    resolved_dir = None
    if save_dir:
        resolved_dir = Path(save_dir)
    elif save_path:
        resolved_dir = Path(save_path).parent

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        prob_class = y_prob[:, i]
        if y_binary.sum() == 0:
            continue
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, prob_class, n_bins=n_bins, strategy="uniform"
            )
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=class_names[i])
        except ValueError:
            continue

    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if resolved_dir:
        plt.savefig(str(resolved_dir / "calibration_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.hist(y_prob[:, i], bins=n_bins, alpha=0.5, label=class_names[i])
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.legend(fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if resolved_dir:
        plt.savefig(str(resolved_dir / "prediction_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
