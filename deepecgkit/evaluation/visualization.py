"""
Visualization functions for ECG evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, roc_curve

from deepecgkit.utils.plotting import finalize_figure


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Plot confusion matrix."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None or len(class_names) != len(labels):
        class_names = [str(lbl) for lbl in labels]

    _, ax = plt.subplots(figsize=(8, 6))

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

    finalize_figure(save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: str | None = None,
) -> None:
    """Plot the ROC curve of a two-class problem.

    Args:
        y_true: True labels, two classes
        y_scores: Scores of the positive class, or the two-column probability
            matrix that ECGTrainer.get_test_results returns for a binary
            model, whose second column is used
        title: Figure title
        save_path: Where to save the figure; None shows it instead

    Raises:
        ValueError: If y_scores holds a column per class for more than two
            classes.
    """
    y_scores = np.asarray(y_scores)
    if y_scores.ndim == 2:
        if y_scores.shape[1] > 2:
            raise ValueError(
                f"plot_roc_curve draws one curve for a two-class problem; got scores for "
                f"{y_scores.shape[1]} classes. For class k against the rest, pass "
                f"y_true == k and y_scores[:, k]."
            )
        y_scores = y_scores[:, -1]
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

    finalize_figure(save_path)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs True Values",
    save_path: str | None = None,
) -> None:
    """Plot predictions vs true values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(title)

    finalize_figure(save_path)


def plot_ecg_signals(
    data: np.ndarray,
    sampling_rate: float = 500.0,
    leads: list[str] | None = None,
    title: str = "ECG Signals",
    save_path: str | None = None,
) -> None:
    """Plot one ECG recording, one row per lead.

    Args:
        data: A single lead as a one-dimensional array, or several leads as a
            two-dimensional one. Both layouts are accepted for the latter,
            (leads, time) as the datasets and models use and (time, leads);
            the longer axis is taken to be time.
        sampling_rate: Sampling rate of the recording in Hz, for the time axis
        leads: Name of each lead, in order
        title: Figure title
        save_path: Where to save the figure; None shows it instead

    Raises:
        ValueError: If data has more than two dimensions, or leads does not
            name every lead.
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim != 2:
        raise ValueError(
            f"plot_ecg_signals draws one recording; got an array of shape {data.shape}. "
            f"Pass a single (leads, time) recording, for example batch[0]."
        )
    elif data.shape[0] < data.shape[1]:
        data = data.T

    n_samples, n_leads = data.shape
    if leads is not None and len(leads) != n_leads:
        raise ValueError(f"Got {len(leads)} lead names for a recording with {n_leads} leads.")
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

    finalize_figure(save_path)


def plot_training_curves(
    metrics_path: str,
    save_dir: str | None = None,
    save_path: str | None = None,
) -> None:
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
    if resolved_dir:
        resolved_dir.mkdir(parents=True, exist_ok=True)

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
    class_names: list[str] | None = None,
    n_bins: int = 10,
    title: str = "Calibration Plot",
    save_dir: str | None = None,
    save_path: str | None = None,
) -> None:
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
    if resolved_dir:
        resolved_dir.mkdir(parents=True, exist_ok=True)

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


def plot_transfer_matrix(
    values: np.ndarray,
    sources: list[str],
    targets: list[str],
    metric: str = "auroc_macro",
    baseline_rows: int = 0,
    title: str = "Cross-Dataset Transfer",
    save_path: str | None = None,
) -> None:
    """Plot probe performance for every encoder source against every probe target.

    Args:
        values: Probe scores of shape (len(sources), len(targets)). Cells that
            could not be scored are NaN and render blank.
        sources: Row labels, the datasets the frozen encoders were trained on.
        targets: Column labels, the datasets the probes were fitted and scored on.
        metric: Name of the plotted metric, shown on the colour bar.
        baseline_rows: Number of leading rows holding a baseline rather than a
            trained encoder, separated from the rest by a rule.
        title: Title for the plot.
        save_path: Path to save the figure to.
    """
    matrix = np.asarray(values, dtype=float)

    _, ax = plt.subplots(figsize=(1.6 * len(targets) + 4, 0.9 * len(sources) + 3))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=list(targets),
        yticklabels=list(sources),
        cbar_kws={"label": metric},
        ax=ax,
    )

    if baseline_rows:
        ax.axhline(baseline_rows, color="white", lw=3)

    ax.set_title(title)
    ax.set_xlabel("Probe target")
    ax.set_ylabel("Encoder source")
    plt.tight_layout()

    finalize_figure(save_path)


def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray | None = None,
    class_names: list[str] | None = None,
    title: str = "Embedding",
    save_path: str | None = None,
) -> None:
    """Scatter a two-dimensional embedding, coloured by class.

    Args:
        embedding: Coordinates of shape (n_samples, 2), as returned by
            deepecgkit.evaluation.representation.embed_2d.
        labels: Class indices of shape (n_samples,), or a multi-hot matrix of
            shape (n_samples, n_classes) in which case each point is coloured by
            its first positive label.
        class_names: Names for the classes, defaulting to their indices.
        title: Title for the plot.
        save_path: Path to save the figure to.

    Raises:
        ValueError: If the coordinates are not two-dimensional, or the labels do
            not align with them.
    """
    coords = np.asarray(embedding, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Embedding must have shape (n_samples, 2); got {coords.shape}. Reduce it with "
            f"embed_2d first."
        )

    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.7, color="tab:blue")
    else:
        groups = np.asarray(labels)
        if groups.ndim > 1 and groups.shape[-1] > 1:
            groups = np.argmax(groups, axis=1)
        groups = groups.reshape(-1).astype(int)
        if groups.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Labels hold {groups.shape[0]} rows but the embedding holds {coords.shape[0]}."
            )

        classes = np.unique(groups)
        palette = plt.get_cmap("tab20" if len(classes) > 10 else "tab10")
        for position, value in enumerate(classes):
            mask = groups == value
            name = (
                class_names[value]
                if class_names is not None and value < len(class_names)
                else f"Class {value}"
            )
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.7,
                color=palette(position % palette.N),
                label=name,
            )
        plt.legend(fontsize="small", markerscale=2)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    finalize_figure(save_path)


def plot_multilabel_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Plot one binary confusion matrix per label of a multi-label problem.

    Args:
        y_true: Multi-hot true labels, shape (n_samples, n_classes)
        y_pred: Multi-hot predicted labels, same shape
        class_names: One name per label
        title: Figure title
        save_path: Where to save the figure; None shows it instead
    """
    from sklearn.metrics import confusion_matrix as _confusion_matrix

    num_classes = y_true.shape[1]
    display_names = class_names or [f"Class {i}" for i in range(num_classes)]

    ncols = min(num_classes, 3)
    nrows = (num_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if num_classes > 1 else [axes]

    for i in range(num_classes):
        cm = _confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
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

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    finalize_figure(save_path)


def plot_multilabel_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    title: str | None = None,
    n_bins: int = 10,
    save_path: str | None = None,
) -> None:
    """Plot per-label reliability curves for a multi-label problem.

    Args:
        y_true: Multi-hot true labels, shape (n_samples, n_classes)
        y_prob: Predicted probabilities, same shape
        class_names: One name per label
        title: Figure title
        n_bins: Reliability-curve bin count
        save_path: Where to save the figure; None shows it instead
    """
    from sklearn.calibration import calibration_curve as _calibration_curve

    num_classes = y_true.shape[1]
    display_names = class_names or [f"Class {i}" for i in range(num_classes)]

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if y_true[:, i].sum() == 0:
            continue
        try:
            fraction_pos, mean_pred = _calibration_curve(
                y_true[:, i],
                y_prob[:, i],
                n_bins=n_bins,
                strategy="uniform",
            )
            plt.plot(mean_pred, fraction_pos, "s-", label=display_names[i])
        except ValueError:
            continue
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    if title:
        plt.title(title)
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    finalize_figure(save_path)
