"""
Command-line interface for deepecg-kit.

Usage:
    deepecg train --model <model> --dataset <dataset> [options]
    deepecg evaluate --checkpoint <path> --dataset <dataset> [options]
    deepecg predict --checkpoint <path> --input <file> [options]
    deepecg resume --checkpoint <path> [options]
    deepecg list-models
    deepecg list-datasets
    deepecg info --model <model>

Datasets are automatically downloaded when first used if not already present.
Use --force-download to re-download an existing dataset.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import torch
import wfdb
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

import deepecgkit.datasets
import deepecgkit.models  # noqa: F401
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
from deepecgkit.registry import (
    _DATASET_REGISTRY,
    _MODEL_REGISTRY,
    get_dataset_info,
    get_dataset_names,
    get_model_names,
)
from deepecgkit.training import ECGLitModel
from deepecgkit.utils.weights import load_pretrained_weights

MODEL_NAMES = get_model_names()
DATASET_NAMES = get_dataset_names()
DATASET_NUM_CLASSES = {n: get_dataset_info(n)["num_classes"] for n in DATASET_NAMES}
DATASET_INPUT_CHANNELS = {n: info["input_channels"] for n, info in _DATASET_REGISTRY.items()}
MODEL_DESCRIPTIONS = {n: info["description"] for n, info in _MODEL_REGISTRY.items()}

MODEL_WEIGHTS = {
    "afmodel": ["afmodel-30s"],
    "kanres": ["kanres-af-30s"],
}


class CLILogger:
    """Simple logger with verbosity control."""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet

    def info(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def debug(self, msg: str) -> None:
        if self.verbose and not self.quiet:
            print(f"[DEBUG] {msg}")

    def error(self, msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)

    def warning(self, msg: str) -> None:
        if not self.quiet:
            print(f"[WARNING] {msg}", file=sys.stderr)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content)
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}. Use .yaml, .yml, or .json")


def _get_model_registry():
    """Return model registry mapping names to classes."""
    return {name: info["class"] for name, info in _MODEL_REGISTRY.items()}


def _get_dataset_registry():
    """Return dataset registry mapping names to classes."""
    return {name: info["class"] for name, info in _DATASET_REGISTRY.items()}


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
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
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
                y_true[:, i], y_prob[:, i], n_bins=10, strategy="uniform",
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
        y_true, y_pred, average=None, zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
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
        logger.info(
            f"  {name:<15} "
            f"{precision[i]:>10.4f} "
            f"{recall[i]:>10.4f} "
            f"{f1[i]:>10.4f}"
        )
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


def train(
    model_name: str,
    dataset_name: str,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    weights: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: int = 1,
    force_download: bool = False,
    early_stopping_patience: int = 10,
    seed: int = 42,
    multi_label: bool = False,
    sampling_rate: Optional[int] = None,
    binary_classification: bool = False,
    normalization: Optional[str] = None,
    logger: Optional[CLILogger] = None,
) -> int:
    """Train a model on a dataset."""
    logger = logger or CLILogger()

    model_registry = _get_model_registry()
    dataset_registry = _get_dataset_registry()

    if model_name not in model_registry:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODEL_NAMES)}")
        return 1

    if dataset_name not in dataset_registry:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
        return 1

    pl.seed_everything(seed)

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("runs") / f"{timestamp}-{model_name}-{dataset_name}"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_class = dataset_registry[dataset_name]
    model_class = model_registry[model_name]

    dataset_kwargs: Dict[str, Any] = {"force_download": force_download}
    if data_dir:
        dataset_kwargs["data_dir"] = data_dir

    af_datasets = {"mitbih-afdb", "ltafdb", "unified-af"}

    if dataset_name == "ptbxl":
        dataset_kwargs["task"] = "superclass"
        dataset_kwargs["multi_label"] = multi_label
        if sampling_rate is not None:
            dataset_kwargs["sampling_rate"] = sampling_rate
            dataset_kwargs["use_high_resolution"] = sampling_rate > 100
        if normalization is not None:
            dataset_kwargs["normalization"] = normalization

    if dataset_name in af_datasets:
        dataset_kwargs["binary_classification"] = binary_classification

    try:
        logger.info(f"Creating data module with {dataset_name} dataset...")
        data_module = ECGDataModule(
            dataset_class=dataset_class,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            stratify=True,
            download=True,
            dataset_kwargs=dataset_kwargs,
        )
        data_module.setup(stage="fit")
    except FileNotFoundError:
        logger.error(f"Dataset '{dataset_name}' not found.")
        if data_dir:
            logger.info(f"Verify the data directory exists: {data_dir}")
        return 1
    except Exception as e:
        logger.error(f"Failed to create data module: {e}")
        return 1

    sample_signal, _ = data_module.dataset[0]
    input_channels = sample_signal.shape[0]
    num_classes = DATASET_NUM_CLASSES.get(dataset_name, 4)
    if binary_classification and dataset_name in af_datasets:
        num_classes = 2

    pos_weight = None
    if multi_label:
        try:
            train_ds = data_module.train_dataset
            all_labels = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
            pos_counts = all_labels.sum(dim=0)
            neg_counts = len(all_labels) - pos_counts
            pos_weight = (neg_counts / pos_counts.clamp(min=1)).tolist()
            logger.info(f"Multi-label pos_weight: {[f'{w:.2f}' for w in pos_weight]}")
        except Exception as e:
            logger.warning(f"Could not compute pos_weight: {e}")
            pos_weight = None

    logger.info(
        f"Creating {model_name} model with {input_channels} input channel(s) and {num_classes} output classes..."
    )
    model = model_class(input_channels=input_channels, output_size=num_classes)

    if weights:
        try:
            weights_path = Path(weights)
            if weights_path.exists():
                logger.info(f"Loading weights from {weights}...")
                state_dict = torch.load(weights, map_location="cpu", weights_only=True)
            elif any(weights in w for ws in MODEL_WEIGHTS.values() for w in ws):
                logger.info(f"Loading pretrained weights '{weights}'...")
                state_dict = load_pretrained_weights(weights, map_location="cpu")
            else:
                all_weights = [w for ws in MODEL_WEIGHTS.values() for w in ws]
                logger.error(f"Weights '{weights}' not found.")
                logger.info(f"Available pretrained weights: {', '.join(all_weights)}")
                logger.info("Or provide a valid file path to a checkpoint.")
                return 1
            model.load_state_dict(state_dict, strict=False)
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return 1

    train_config = {
        "learning_rate": learning_rate,
        "scheduler": {"factor": 0.5, "patience": 5},
        "binary_classification": num_classes == 2,
        "multi_label": multi_label,
        "task_type": "classification",
        "pos_weight": pos_weight,
    }

    lit_model = ECGLitModel(model=model, train_config=train_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path / "checkpoints",
        filename=f"{model_name}-{dataset_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
    )

    csv_logger = CSVLogger(save_dir=output_path, name="training_logs")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping],
        logger=csv_logger,
        default_root_dir=output_path,
        enable_progress_bar=not logger.quiet,
    )

    try:
        logger.info(f"Starting training for {epochs} epochs...")
        trainer.fit(lit_model, data_module)

        if data_module.test_dataloader() is not None:
            logger.info("Running test evaluation...")
            trainer.test(lit_model, data_module)

        logger.info(f"Training complete. Checkpoints saved to {output_path}")

        try:
            matplotlib.use("Agg")

            class_names = None
            if hasattr(data_module, "dataset") and hasattr(data_module.dataset, "class_names"):
                class_names = data_module.dataset.class_names

            metrics_file = Path(csv_logger.log_dir) / "metrics.csv"
            if metrics_file.exists():
                plot_training_curves(
                    str(metrics_file),
                    save_dir=str(output_path),
                )
                logger.info(f"Loss plot saved to {output_path / 'loss.png'}")
                logger.info(f"Accuracy plot saved to {output_path / 'accuracy.png'}")

            y_pred, y_true, y_prob = lit_model.get_test_results()

            if y_pred is not None:
                is_multi_label = y_true.ndim > 1

                if is_multi_label:
                    _save_multi_label_evaluation(
                        y_true, y_pred, y_prob,
                        model_name, dataset_name, class_names,
                        output_path, logger,
                    )
                else:
                    _save_single_label_evaluation(
                        y_true, y_pred, y_prob,
                        model_name, dataset_name, class_names,
                        output_path, logger,
                    )

        except Exception as e:
            logger.warning(f"Post-training evaluation failed: {e}")

        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def resume(
    checkpoint: str,
    epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
    accelerator: str = "auto",
    devices: int = 1,
    early_stopping_patience: int = 10,
    logger: Optional[CLILogger] = None,
) -> int:
    """Resume training from a checkpoint."""
    logger = logger or CLILogger()

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    try:
        logger.info(f"Loading checkpoint from {checkpoint}...")
        lit_model = ECGLitModel.load_from_checkpoint(checkpoint)

        output_path = Path(output_dir) if output_dir else checkpoint_path.parent
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=output_path,
            filename="resumed-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
        )

        trainer_kwargs: Dict[str, Any] = {
            "accelerator": accelerator,
            "devices": devices,
            "callbacks": [checkpoint_callback, early_stopping],
            "default_root_dir": output_path,
            "enable_progress_bar": not logger.quiet,
        }
        if epochs:
            trainer_kwargs["max_epochs"] = epochs

        trainer = pl.Trainer(**trainer_kwargs)

        logger.info("Resuming training...")
        trainer.fit(lit_model, ckpt_path=checkpoint)

        logger.info(f"Training complete. Checkpoints saved to {output_path}")
        return 0
    except Exception as e:
        logger.error(f"Failed to resume training: {e}")
        return 1


def evaluate(
    checkpoint: str,
    dataset_name: str,
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: int = 1,
    force_download: bool = False,
    split: str = "test",
    logger: Optional[CLILogger] = None,
) -> int:
    """Evaluate a trained model on a dataset."""
    logger = logger or CLILogger()

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    dataset_registry = _get_dataset_registry()

    if dataset_name not in dataset_registry:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
        return 1

    try:
        logger.info(f"Loading model from {checkpoint}...")
        lit_model = ECGLitModel.load_from_checkpoint(checkpoint)

        dataset_class = dataset_registry[dataset_name]
        dataset_kwargs: Dict[str, Any] = {"force_download": force_download}
        if data_dir:
            dataset_kwargs["data_dir"] = data_dir

        logger.info(f"Creating data module with {dataset_name} dataset...")
        data_module = ECGDataModule(
            dataset_class=dataset_class,
            batch_size=batch_size,
            num_workers=num_workers,
            download=True,
            dataset_kwargs=dataset_kwargs,
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            enable_progress_bar=not logger.quiet,
        )

        logger.info(f"Running evaluation on {split} split...")
        if split == "test":
            results = trainer.test(lit_model, data_module)
        elif split == "val":
            results = trainer.validate(lit_model, data_module)
        else:
            logger.error(f"Invalid split: {split}. Use 'test' or 'val'.")
            return 1

        if results:
            logger.info("\nEvaluation Results:")
            logger.info("-" * 40)
            for key, value in results[0].items():
                logger.info(f"  {key}: {value:.4f}")

        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def predict(
    checkpoint: str,
    input_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 1,
    accelerator: str = "auto",
    logger: Optional[CLILogger] = None,
) -> int:
    """Run inference on ECG data."""
    logger = logger or CLILogger()

    checkpoint_file = Path(checkpoint)
    if not checkpoint_file.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    input_file = Path(input_path)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        logger.info(f"Loading model from {checkpoint}...")
        lit_model = ECGLitModel.load_from_checkpoint(checkpoint)
        lit_model.eval()

        logger.info(f"Loading input data from {input_path}...")
        if input_file.suffix == ".npy":
            data = np.load(input_path)
        elif input_file.suffix == ".csv":
            data = pd.read_csv(input_path).values
        elif input_file.suffix in (".dat", ".hea"):
            record = wfdb.rdrecord(str(input_file.with_suffix("")))
            data = record.p_signal
        else:
            logger.error(f"Unsupported input format: {input_file.suffix}")
            logger.info("Supported formats: .npy, .csv, .dat/.hea (WFDB)")
            return 1

        if data.ndim == 1:
            data = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            data = data.reshape(data.shape[0], 1, -1)

        tensor = torch.tensor(data, dtype=torch.float32)

        logger.info("Running inference...")
        device = "cuda" if accelerator == "gpu" and torch.cuda.is_available() else "cpu"
        lit_model = lit_model.to(device)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = lit_model(tensor)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        results = {
            "predictions": predictions.cpu().numpy().tolist(),
            "probabilities": probabilities.cpu().numpy().tolist(),
        }

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.info("\nPrediction Results:")
            logger.info("-" * 40)
            for i, (pred, prob) in enumerate(zip(results["predictions"], results["probabilities"])):
                logger.info(f"  Sample {i}: class={pred}, confidence={max(prob):.4f}")

        return 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def list_models() -> None:
    """List all available models."""
    print("Available models:")
    print("-" * 60)
    for name in sorted(MODEL_NAMES):
        desc = MODEL_DESCRIPTIONS.get(name, "")
        weights = MODEL_WEIGHTS.get(name, [])
        print(f"  {name}")
        if desc:
            print(f"    Description: {desc}")
        if weights:
            print(f"    Pretrained weights: {', '.join(weights)}")
        print()


def list_datasets() -> None:
    """List all available datasets."""
    print("Available datasets:")
    print("-" * 60)
    for name in sorted(DATASET_NAMES):
        num_classes = DATASET_NUM_CLASSES[name]
        print(f"  {name}")
        print(f"    Classes: {num_classes}")
        print()


def show_info(model_name: str, logger: Optional[CLILogger] = None) -> int:
    """Show detailed information about a model."""
    logger = logger or CLILogger()

    if model_name not in MODEL_NAMES:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODEL_NAMES)}")
        return 1

    try:
        model_registry = _get_model_registry()
        model_class = model_registry[model_name]
        model = model_class(input_channels=1, output_size=4)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model: {model_name}")
        print("=" * 60)
        print(f"Description: {MODEL_DESCRIPTIONS.get(model_name, 'N/A')}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Pretrained weights: {', '.join(MODEL_WEIGHTS.get(model_name, ['None']))}")
        print()
        print("Architecture:")
        print("-" * 60)
        print(model)

        return 0
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="deepecg",
        description="DeepECG-Kit: Deep learning toolkit for ECG analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepecg train -m kanres -d af-classification
  deepecg train -m kanres -d af-classification --force-download
  deepecg evaluate --checkpoint model.ckpt -d af-classification
  deepecg predict --checkpoint model.ckpt --input ecg.npy
  deepecg resume --checkpoint model.ckpt --epochs 100
  deepecg list-models
  deepecg info -m kanres

Note: Datasets are automatically downloaded when first used.
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model on a dataset")
    train_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to train",
    )
    train_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Dataset to train on",
    )
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset (default: auto-detect)",
    )
    train_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: runs/{timestamp}-{model}-{dataset})",
    )
    train_parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to weights file or pretrained weight name",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    train_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    train_parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    train_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    train_parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    train_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset even if it exists",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--multi-label",
        action="store_true",
        default=False,
        help="Use multi-label classification with BCE loss (default: single-label with CE loss)",
    )
    train_parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Target sampling rate in Hz for ECG signals (e.g., 100 or 500 for PTB-XL)",
    )
    train_parser.add_argument(
        "--binary-classification",
        action="store_true",
        default=False,
        help="Use binary classification (AF vs Non-AF) for AF datasets (default: 4-class)",
    )
    train_parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["zscore", "minmax", "none"],
        help="Normalization method for ECG signals (default: dataset-specific)",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Dataset to evaluate on",
    )
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset",
    )
    eval_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    eval_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    eval_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    eval_parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    eval_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset even if it exists",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on (default: test)",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference on ECG data")
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    predict_parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input ECG file (.npy, .csv, or WFDB .dat/.hea)",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save predictions (JSON format)",
    )
    predict_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    predict_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume training from a checkpoint")
    resume_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to resume from",
    )
    resume_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Additional epochs to train (default: continue original)",
    )
    resume_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save new checkpoints",
    )
    resume_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    resume_parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    resume_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model to show information for",
    )

    subparsers.add_parser("list-models", help="List all available models")
    subparsers.add_parser("list-datasets", help="List all available datasets")

    args = parser.parse_args(argv)

    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
            return 1

    logger = CLILogger(
        verbose=args.verbose or config.get("verbose", False),
        quiet=args.quiet or config.get("quiet", False),
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        train_args = {
            "model_name": args.model,
            "dataset_name": args.dataset,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "weights": args.weights,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "num_workers": args.num_workers,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "force_download": args.force_download,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
            "multi_label": args.multi_label,
            "sampling_rate": args.sampling_rate,
            "binary_classification": args.binary_classification,
            "normalization": args.normalization,
        }
        for key, value in config.get("train", {}).items():
            if key in train_args and train_args[key] == parser.get_default(key):
                train_args[key] = value
        return train(logger=logger, **train_args)

    elif args.command == "evaluate":
        return evaluate(
            checkpoint=args.checkpoint,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
            force_download=args.force_download,
            split=args.split,
            logger=logger,
        )

    elif args.command == "predict":
        return predict(
            checkpoint=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            logger=logger,
        )

    elif args.command == "resume":
        return resume(
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            output_dir=args.output_dir,
            accelerator=args.accelerator,
            devices=args.devices,
            early_stopping_patience=args.early_stopping_patience,
            logger=logger,
        )

    elif args.command == "info":
        return show_info(model_name=args.model, logger=logger)

    elif args.command == "list-models":
        list_models()
        return 0

    elif args.command == "list-datasets":
        list_datasets()
        return 0

    return 0


__all__ = [
    "DATASET_INPUT_CHANNELS",
    "DATASET_NAMES",
    "DATASET_NUM_CLASSES",
    "MODEL_DESCRIPTIONS",
    "MODEL_NAMES",
    "MODEL_WEIGHTS",
    "CLILogger",
    "evaluate",
    "list_datasets",
    "list_models",
    "load_config",
    "main",
    "predict",
    "resume",
    "show_info",
    "train",
]


if __name__ == "__main__":
    sys.exit(main())
