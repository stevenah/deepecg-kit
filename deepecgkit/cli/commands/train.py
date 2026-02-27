"""Train and resume commands."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from deepecgkit.cli.evaluation import _save_multi_label_evaluation, _save_single_label_evaluation
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    DATASET_NUM_CLASSES,
    MODEL_NAMES,
    MODEL_WEIGHTS,
    _get_dataset_registry,
    _get_model_registry,
)
from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation.visualization import plot_training_curves
from deepecgkit.training import ECGLitModel
from deepecgkit.utils.weights import load_pretrained_weights


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

    # BCE loss expects a single logit output for binary classification
    model_output_size = (
        1 if (binary_classification and dataset_name in af_datasets) else num_classes
    )

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
    model = model_class(input_channels=input_channels, output_size=model_output_size)

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
                        y_true,
                        y_pred,
                        y_prob,
                        model_name,
                        dataset_name,
                        class_names,
                        output_path,
                        logger,
                    )
                else:
                    _save_single_label_evaluation(
                        y_true,
                        y_pred,
                        y_prob,
                        model_name,
                        dataset_name,
                        class_names,
                        output_path,
                        logger,
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
