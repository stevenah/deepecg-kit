"""Train and resume commands."""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import torch

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    build_data_module,
    dataset_kwargs_for,
    infer_input_channels_from,
    require_dataset,
    require_model,
)
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    DATASET_NUM_CLASSES,
    MODEL_NAMES,
    MODEL_WEIGHTS,
    _get_model_registry,
)
from deepecgkit.cli.run_dir import write_run_metadata
from deepecgkit.evaluation.report import save_evaluation
from deepecgkit.evaluation.visualization import plot_training_curves
from deepecgkit.training import ECGTrainer
from deepecgkit.utils.weights import load_pretrained_weights


def _write_split_info(data_module: Any, output_path: Path, seed: int, logger: CLILogger) -> None:
    """Record the split this run trained and evaluated on.

    The number of independent units in the test partition, not the number of
    samples, governs whether any later comparison can support a confidence
    interval. Grouped datasets hold entire recordings out, so a large sample
    count can rest on very few subjects: MIT-BIH AFDB has 23 records, and a
    ten percent test fraction yields two. Nothing else records this.

    Args:
        data_module: The configured data module, already set up.
        output_path: Run directory.
        seed: Seed that produced the split.
        logger: CLI logger.
    """
    try:
        n_train = len(data_module.train_dataset) if data_module.train_dataset is not None else 0
        n_val = len(data_module.val_dataset) if data_module.val_dataset is not None else 0
        n_test = len(data_module.test_dataset) if data_module.test_dataset is not None else 0

        group_ids = list(getattr(data_module.dataset, "group_ids", None) or [])
        test_groups: list[str] = []
        test_subset = data_module.test_dataset
        if group_ids and test_subset is not None and hasattr(test_subset, "indices"):
            test_groups = sorted({str(group_ids[index]) for index in test_subset.indices})

        payload = {
            "seed": seed,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_test_groups": len(test_groups) if test_groups else n_test,
            "grouped": bool(test_groups),
            "test_groups": test_groups,
        }
        (output_path / "split_info.json").write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(
            f"Split: {n_train} train, {n_val} val, {n_test} test across "
            f"{payload['n_test_groups']} independent unit(s)"
        )
    except Exception as e:
        logger.warning(f"Could not record split info: {e}")


def train(
    model_name: str,
    dataset_name: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    weights: str | None = None,
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
    sampling_rate: int | None = None,
    binary_classification: bool = False,
    normalization: str | None = None,
    logger: CLILogger | None = None,
) -> int:
    """Train a model on a dataset."""
    logger = logger or CLILogger()

    for code in (require_model(model_name, logger), require_dataset(dataset_name, logger)):
        if code is not None:
            return code

    model_registry = _get_model_registry()

    ECGTrainer.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("runs") / f"{timestamp}-{model_name}-{dataset_name}-seed{seed}"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    write_run_metadata(
        output_path,
        {
            "model": model_name,
            "dataset": dataset_name,
            "seed": seed,
            "val_split": val_split,
            "test_split": test_split,
            "multi_label": multi_label,
            "binary_classification": binary_classification,
            "sampling_rate": sampling_rate,
            "normalization": normalization,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    )

    model_class = model_registry[model_name]

    try:
        data_module = build_data_module(
            dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            dataset_kwargs=dataset_kwargs_for(
                dataset_name,
                force_download=force_download,
                data_dir=data_dir,
                multi_label=multi_label,
                binary_classification=binary_classification,
                sampling_rate=sampling_rate,
                normalization=normalization,
            ),
            stage="fit",
            logger=logger,
        )
    except FileNotFoundError:
        logger.error(f"Dataset '{dataset_name}' not found.")
        if data_dir:
            logger.info(f"Verify the data directory exists: {data_dir}")
        return 1
    except Exception as e:
        logger.error(f"Failed to create data module: {e}")
        return 1

    _write_split_info(data_module, output_path, seed, logger)

    input_channels = infer_input_channels_from(data_module)
    num_classes = DATASET_NUM_CLASSES.get(dataset_name, 4)
    if binary_classification and dataset_name in defaults.AF_DATASETS:
        num_classes = 2

    model_output_size = (
        1 if (binary_classification and dataset_name in defaults.AF_DATASETS) else num_classes
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

    trainer = ECGTrainer(model=model, train_config=train_config, device=accelerator)
    trainer.model_config = {
        "name": model_name,
        "input_channels": int(input_channels),
        "output_size": int(model_output_size),
    }

    checkpoint_dir = output_path / "checkpoints"
    log_dir = output_path / "training_logs"

    try:
        logger.info(f"Starting training for {epochs} epochs...")
        trainer.fit(
            data_module,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            progress_bar=not logger.quiet,
        )

        val_targets = None
        val_logits = None
        if data_module.val_dataset is not None and len(data_module.val_dataset) > 0:
            try:
                logger.info("Running validation pass to fit calibration temperature...")
                trainer.validate(data_module)
                _, val_targets, _ = trainer.get_test_results()
                val_logits = trainer.get_test_logits()
            except Exception as e:
                logger.warning(f"Validation pass failed, so no temperature will be fitted: {e}")
                val_targets = None
                val_logits = None

        if data_module.test_dataset is not None and len(data_module.test_dataset) > 0:
            logger.info("Running test evaluation...")
            trainer.test(data_module)

        logger.info(f"Training complete. Checkpoints saved to {output_path}")

        matplotlib.use("Agg")

        class_names = None
        if hasattr(data_module, "dataset") and hasattr(data_module.dataset, "class_names"):
            class_names = data_module.dataset.class_names

        try:
            metrics_file = log_dir / "metrics.csv"
            if metrics_file.exists():
                plot_training_curves(
                    str(metrics_file),
                    save_dir=str(output_path),
                )
                logger.info(f"Loss plot saved to {output_path / 'loss.png'}")
                logger.info(f"Accuracy plot saved to {output_path / 'accuracy.png'}")
        except Exception as e:
            logger.warning(f"Training curve plotting failed: {e}")

        y_pred, y_true, y_prob = trainer.get_test_results()
        test_logits = trainer.get_test_logits()

        if y_pred is not None:
            save_evaluation(
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
            )
        elif data_module.test_dataset is not None and len(data_module.test_dataset) > 0:
            raise RuntimeError(
                "Test evaluation produced no predictions, so no metrics were written. "
                "The run is incomplete and must not be treated as successful."
            )

        write_run_metadata(output_path, {"returncode": 0})
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


def add_parser(subparsers: Any) -> Any:
    """Register the train subcommand's arguments."""
    parser = subparsers.add_parser("train", help="Train a model on a dataset")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: runs/{timestamp}-{model}-{dataset}-seed{seed})",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to weights file or pretrained weight name",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=defaults.EPOCHS,
        help=f"Number of training epochs (default: {defaults.EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=defaults.BATCH_SIZE,
        help=f"Batch size (default: {defaults.BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=defaults.LEARNING_RATE,
        help=f"Learning rate (default: {defaults.LEARNING_RATE})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=defaults.VAL_SPLIT,
        help=f"Validation split ratio (default: {defaults.VAL_SPLIT})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=defaults.TEST_SPLIT,
        help=f"Test split ratio (default: {defaults.TEST_SPLIT})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.NUM_WORKERS,
        help=f"Number of data loading workers (default: {defaults.NUM_WORKERS})",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=defaults.ACCELERATOR,
        choices=defaults.ACCELERATOR_CHOICES,
        help="Accelerator to use (default: auto)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=defaults.DEVICES,
        help=f"Number of devices to use (default: {defaults.DEVICES})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset even if it exists",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience (default: {defaults.EARLY_STOPPING_PATIENCE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.SEED,
        help=f"Random seed (default: {defaults.SEED})",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        default=False,
        help="Use multi-label classification with BCE loss (default: single-label with CE loss)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Target sampling rate in Hz for ECG signals (e.g., 100 or 500 for PTB-XL)",
    )
    parser.add_argument(
        "--binary-classification",
        action="store_true",
        default=False,
        help="Use binary classification (AF vs Non-AF) for AF datasets (default: 4-class)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["zscore", "minmax", "none"],
        help="Normalization method for ECG signals (default: dataset-specific)",
    )
    parser.set_defaults(handler="train", to_kwargs=_to_kwargs)
    return parser


def _to_kwargs(args: Any) -> dict[str, Any]:
    return {
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
