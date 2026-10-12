"""Train command."""

import json
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
import torch

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    build_data_module,
    check_dataset_options,
    check_input_length,
    dataset_kwargs_for,
    describe_data,
    report_data_error,
    report_shape_error,
    require_dataset,
    require_model,
    split_record_names,
)
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import MODEL_WEIGHTS, _get_model_registry
from deepecgkit.cli.run_dir import resolve_checkpoint, write_run_metadata
from deepecgkit.evaluation.report import save_evaluation
from deepecgkit.evaluation.visualization import plot_training_curves
from deepecgkit.models.pretrained import load_matching_weights
from deepecgkit.registry import get_dataset_names, get_model_names
from deepecgkit.training import ECGTrainer
from deepecgkit.utils.device import resolve_device
from deepecgkit.utils.provenance import dataset_fingerprint, environment_summary, git_commit
from deepecgkit.utils.weights import load_pretrained_weights

#: How many weight names a message about partly loaded weights spells out.
NAMED_WEIGHTS = 5


def _package_version() -> str | None:
    try:
        return version("deepecgkit")
    except PackageNotFoundError:
        return None


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


def _log_data_summary(data_config: dict[str, Any], logger: CLILogger) -> None:
    """Say what the model is about to be trained on, in one line.

    Args:
        data_config: Description from describe_data
        logger: CLI logger
    """
    preprocessing = data_config.get("preprocessing") or {}
    parts = [f"{data_config.get('input_channels')} lead(s)"]
    parts.append(f"{data_config.get('input_length')} samples")
    rate, duration = (
        preprocessing.get("sampling_rate"),
        preprocessing.get("segment_duration_seconds"),
    )
    if rate and duration:
        parts.append(f"{duration:g} s at {rate:g} Hz")
    elif rate:
        parts.append(f"{rate:g} Hz")
    if preprocessing.get("normalization"):
        parts.append(f"{preprocessing['normalization']} normalisation")
    logger.info(f"Input: {', '.join(parts)}")


def train(
    model_name: str,
    dataset_name: str,
    data_dir: str | None = None,
    output_dir: str | None = None,
    weights: str | None = None,
    epochs: int = defaults.EPOCHS,
    batch_size: int = defaults.BATCH_SIZE,
    learning_rate: float = defaults.LEARNING_RATE,
    val_split: float = defaults.VAL_SPLIT,
    test_split: float = defaults.TEST_SPLIT,
    num_workers: int = defaults.NUM_WORKERS,
    accelerator: str = "auto",
    devices: int = 1,
    force_download: bool = False,
    early_stopping_patience: int = defaults.EARLY_STOPPING_PATIENCE,
    seed: int = defaults.SEED,
    multi_label: bool = False,
    sampling_rate: int | None = None,
    binary_classification: bool = False,
    normalization: str | None = None,
    segment_duration: float | None = None,
    logger: CLILogger | None = None,
) -> int:
    """Train a model on a dataset."""
    logger = logger or CLILogger()

    for code in (require_model(model_name, logger), require_dataset(dataset_name, logger)):
        if code is not None:
            return code

    code = check_dataset_options(
        dataset_name,
        logger,
        multi_label=multi_label,
        binary_classification=binary_classification,
        sampling_rate=sampling_rate,
        normalization=normalization,
        segment_duration=segment_duration,
    )
    if code is not None:
        return code

    if devices != 1:
        logger.warning(
            f"--devices {devices} has no effect: training runs on a single device. "
            f"The flag is deprecated and will be removed."
        )

    model_registry = _get_model_registry()

    ECGTrainer.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

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
                segment_duration=segment_duration,
            ),
            stage="fit",
            logger=logger,
        )
    except Exception as e:
        return report_data_error(e, dataset_name, logger, data_dir)

    dataset = data_module.dataset
    if dataset is None or len(dataset) == 0:
        logger.error(
            f"The '{dataset_name}' dataset loaded no samples, so there is nothing to train on."
        )
        return 1

    if (code := check_input_length(model_name, dataset_name, data_module, logger)) is not None:
        return code

    num_classes = int(getattr(dataset, "num_classes", 0) or 0)
    if not num_classes:
        logger.error(f"The '{dataset_name}' dataset does not report its number of classes.")
        return 1
    model_output_size = 1 if binary_classification else num_classes

    data_config = describe_data(data_module, dataset_name)
    preprocessing = data_config.get("preprocessing") or {}
    input_channels = data_config["input_channels"]
    _log_data_summary(data_config, logger)

    # Nothing is written until the run can actually start, so a run that
    # fails on its inputs leaves no directory behind.
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
            "sampling_rate": preprocessing.get("sampling_rate"),
            "normalization": preprocessing.get("normalization"),
            "segment_duration_seconds": preprocessing.get("segment_duration_seconds"),
            "input_channels": input_channels,
            "input_length": data_config.get("input_length"),
            "class_names": data_config.get("class_names"),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "deepecgkit_version": _package_version(),
            "git": git_commit(),
            "environment": environment_summary(resolve_device(accelerator)),
            "dataset_source": getattr(type(dataset), "DOWNLOAD_URL", None),
            "dataset_files": dataset_fingerprint(getattr(dataset, "data_dir", None)),
        },
    )

    _write_split_info(data_module, output_path, seed, logger)

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
                # A run directory stands for its best checkpoint, as it does for --checkpoint.
                weights_path = resolve_checkpoint(weights_path)
                logger.info(f"Loading weights from {weights_path}...")
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            elif any(weights in w for ws in MODEL_WEIGHTS.values() for w in ws):
                logger.info(f"Loading pretrained weights '{weights}'...")
                state_dict = load_pretrained_weights(weights, map_location="cpu")
            else:
                all_weights = [w for ws in MODEL_WEIGHTS.values() for w in ws]
                logger.error(f"Weights '{weights}' not found.")
                logger.info(f"Available pretrained weights: {', '.join(all_weights)}")
                logger.info("Or provide a valid file path to a checkpoint.")
                return _fail(output_path, f"Weights '{weights}' not found.")
            report = load_matching_weights(model, state_dict)
            logger.info(
                f"Loaded {len(report['loaded'])} of the model's {len(model.state_dict())} "
                f"weight tensors."
            )
            left_out = {
                "Left as initialised, their shape differs": report["mismatched"],
                "Left as initialised, not in the file": report["missing"],
                "In the file but not in the model": report["unexpected"],
            }
            for reason, names in left_out.items():
                if names:
                    shown = ", ".join(names[:NAMED_WEIGHTS])
                    more = f" and {len(names) - NAMED_WEIGHTS} more" * (len(names) > NAMED_WEIGHTS)
                    logger.warning(f"{reason}: {shown}{more}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return _fail(output_path, f"Failed to load weights: {e}")

    train_config = {
        "learning_rate": learning_rate,
        "scheduler": {"factor": 0.5, "patience": 5},
        "binary_classification": binary_classification,
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
    trainer.data_config = data_config

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

        class_names = data_config.get("class_names")

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
                record_names=split_record_names(data_module, "test"),
            )
        elif data_module.test_dataset is not None and len(data_module.test_dataset) > 0:
            raise RuntimeError(
                "Test evaluation produced no predictions, so no metrics were written. "
                "The run is incomplete and must not be treated as successful."
            )

        write_run_metadata(
            output_path,
            {"returncode": 0, "best_checkpoint": trainer.best_checkpoint_path, "error": None},
        )
        if trainer.best_checkpoint_path:
            logger.info(f"\nBest checkpoint: {trainer.best_checkpoint_path}")
            logger.info(
                f"Next: deepecg evaluate --checkpoint {output_path} -d {dataset_name} -o "
                f"{output_path / 'evaluation'}"
            )
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        write_run_metadata(output_path, {"returncode": 130, "error": "interrupted"})
        return 130
    except RuntimeError as e:
        report_shape_error(e, model_name, "Training", logger)
        return _fail(output_path, str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.debug(traceback.format_exc())
        return _fail(output_path, str(e))


def _fail(output_path: Path, error: str) -> int:
    """Record a failed run in its metadata and return the failure exit code.

    Args:
        output_path: Run directory
        error: What went wrong, first line kept

    Returns:
        The exit code 1.
    """
    summary = error.splitlines()[0] if error else "unknown error"
    write_run_metadata(output_path, {"returncode": 1, "error": summary})
    return 1


def add_parser(subparsers: Any) -> Any:
    """Register the train subcommand's arguments."""
    parser = subparsers.add_parser("train", help="Train a model on a dataset")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=get_model_names(),
        metavar="MODEL",
        help="Model architecture to train (see: deepecg list-models)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=get_dataset_names(),
        metavar="DATASET",
        help="Dataset to train on (see: deepecg list-datasets)",
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
        help="Deprecated and ignored: training runs on a single device",
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
        help="Multi-label classification with BCE loss, for datasets that support it (ptbxl)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Sampling rate in Hz the signals are resampled to (default: the dataset's own)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Length of each training segment in seconds, for datasets cut from longer "
        "recordings (default: the dataset's own, 10)",
    )
    parser.add_argument(
        "--binary-classification",
        action="store_true",
        default=False,
        help="Collapse labels to AF vs Non-AF, for datasets that support it "
        "(mitbih-afdb, ltafdb, unified-af)",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["zscore", "minmax", "none"],
        help="Normalization method for ECG signals (default: the dataset's own, zscore)",
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
        "segment_duration": args.segment_duration,
    }
