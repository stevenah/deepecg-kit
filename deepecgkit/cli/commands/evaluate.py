"""Evaluate command."""

import traceback
from typing import Any

import numpy as np

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    build_data_module,
    checkpoint_dataset_kwargs,
    infer_input_channels_from,
    prepare_output_dir,
    read_checkpoint,
    report_data_error,
    report_shape_error,
    require_checkpoint,
    require_dataset,
    require_model,
    resolve_model_name,
)
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.run_dir import load_run_context, verify_split
from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation.metrics import DEFAULT_CALIBRATION_BINS
from deepecgkit.evaluation.report import save_evaluation
from deepecgkit.models.factory import build_model_from_checkpoint
from deepecgkit.registry import get_dataset_names, get_model_names
from deepecgkit.training import ECGTrainer


def _read_train_config(checkpoint: str) -> dict[str, Any]:
    """Read the training configuration recorded in a checkpoint.

    Args:
        checkpoint: Path to the checkpoint file

    Returns:
        The stored train_config, or an empty mapping when it is absent.
    """
    config = read_checkpoint(checkpoint).get("train_config")
    return config if isinstance(config, dict) else {}


def _dataset_kwargs_for_checkpoint(
    dataset_name: str,
    train_config: dict[str, Any],
    force_download: bool,
    data_dir: str | None,
) -> tuple[dict[str, Any], bool]:
    """Rebuild the dataset in the labelling mode the checkpoint was trained in.

    A checkpoint records whether it was trained multi-label or as a collapsed
    binary problem, and the dataset must be reconstructed the same way. Building
    it in the default mode instead yields labels of the wrong shape, which
    surfaces later as an opaque broadcasting error rather than as the
    configuration mismatch it is.

    Args:
        dataset_name: Registered dataset name
        train_config: Configuration recovered from the checkpoint
        force_download: Whether to re-download the dataset
        data_dir: Optional dataset directory

    Returns:
        Tuple of (dataset keyword arguments, whether targets are multi-label).
    """
    return checkpoint_dataset_kwargs(
        dataset_name,
        {"train_config": train_config},
        CLILogger(quiet=True),
        force_download=force_download,
        data_dir=data_dir,
    )


def _check_target_shape(data_module: ECGDataModule, multi_label: bool) -> str | None:
    """Confirm the rebuilt dataset produces targets of the expected rank.

    Args:
        data_module: Data module whose setup has already run
        multi_label: Whether the checkpoint expects multi-hot targets

    Returns:
        A description of the mismatch, or None when the shapes agree.
    """
    _, target = data_module.dataset[0]
    observed = int(getattr(target, "ndim", 0)) > 0 and int(np.prod(target.shape)) > 1
    if observed == multi_label:
        return None
    expected = "multi-label" if multi_label else "single-label"
    actual = "multi-label" if observed else "single-label"
    return (
        f"Checkpoint was trained on {expected} targets but the dataset produced {actual} "
        "ones. The checkpoint and the dataset configuration do not match."
    )


def _collect_split(trainer: ECGTrainer, data_module: ECGDataModule, split: str) -> tuple:
    """Run one split and take a copy of the arrays it produced.

    ECGTrainer reuses a single set of buffers for every evaluation pass, so the
    results of one split must be copied out before the next pass clears them.

    Args:
        trainer: Trainer holding the loaded model
        data_module: Data module supplying the loaders
        split: Either "test" or "val"

    Returns:
        Tuple of (summary metrics, predictions, targets, probabilities, logits).
    """
    results = trainer.test(data_module) if split == "test" else trainer.validate(data_module)
    predictions, targets, probabilities = trainer.get_test_results()
    return results, predictions, targets, probabilities, trainer.get_test_logits()


def evaluate(
    checkpoint: str,
    dataset_name: str,
    model_name: str | None = None,
    data_dir: str | None = None,
    batch_size: int = defaults.BATCH_SIZE,
    num_workers: int = defaults.NUM_WORKERS,
    accelerator: str = "auto",
    devices: int = 1,
    force_download: bool = False,
    split: str = "test",
    seed: int = defaults.SEED,
    val_split: float = defaults.VAL_SPLIT,
    test_split: float = defaults.TEST_SPLIT,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    calibrate: bool = True,
    output_dir: str | None = None,
    sampling_rate: int | None = None,
    logger: CLILogger | None = None,
) -> int:
    """Evaluate a trained model on a dataset.

    The dataset is rebuilt the way the checkpoint was trained: its labelling
    mode and its preprocessing (sampling rate, segment length, normalisation)
    are read back from the checkpoint, so the model sees inputs shaped like
    the ones it learned from.

    The split is redrawn here rather than read back from the run, so seed,
    val_split and test_split must match the values the checkpoint was trained
    with. Mismatched values silently produce a different partition, which for
    calibration matters more than usual: a temperature fitted on a validation
    split that overlaps the original training data is fitted on data the model
    has already seen.

    Args:
        checkpoint: Checkpoint file, or a run directory whose best checkpoint
            is used
        dataset_name: Registered dataset name
        model_name: Registered model name; optional when the checkpoint
            records it
        data_dir: Optional dataset directory
        batch_size: Evaluation batch size
        num_workers: Data loading workers
        accelerator: Device to evaluate on
        devices: Accepted for interface compatibility and not used
        force_download: Whether to re-download the dataset
        split: Split to report on, "test" or "val"
        seed: Seed that produced the split during training
        val_split: Validation fraction used during training
        test_split: Test fraction used during training
        calibration_bins: Bins for the calibration metrics
        calibrate: Whether to run the extra validation pass that fits a
            temperature
        output_dir: When given, write the full artifact set (metrics CSV,
            predictions, plots) into this directory
        sampling_rate: Override for the sampling rate the dataset is
            resampled to, for checkpoints that do not record theirs
        logger: CLI logger

    Returns:
        Zero on success, one on failure.
    """
    logger = logger or CLILogger()

    resolved = require_checkpoint(checkpoint, logger)
    if resolved is None:
        return 1
    checkpoint = resolved
    for code in (
        require_dataset(dataset_name, logger),
        require_model(model_name, logger) if model_name is not None else None,
    ):
        if code is not None:
            return code

    if split not in ("test", "val"):
        logger.error(f"Invalid split: {split}. Use 'test' or 'val'.")
        return 1

    context = load_run_context(checkpoint)
    if context.source != "none":
        for attr, recorded, default in (
            ("seed", context.seed, defaults.SEED),
            ("val_split", context.val_split, defaults.VAL_SPLIT),
            ("test_split", context.test_split, defaults.TEST_SPLIT),
        ):
            current = {"seed": seed, "val_split": val_split, "test_split": test_split}[attr]
            if recorded is None or recorded == current:
                continue
            if current == default:
                logger.info(f"Using {attr}={recorded} recorded in {context.run_dir}")
                if attr == "seed":
                    seed = recorded
                elif attr == "val_split":
                    val_split = recorded
                else:
                    test_split = recorded
            else:
                logger.warning(
                    f"--{attr.replace('_', '-')} {current} differs from the recorded "
                    f"training value {recorded}; the rebuilt split will not match training."
                )

    try:
        stored = read_checkpoint(checkpoint)
    except Exception as e:
        logger.error(f"Evaluation failed: could not read checkpoint {checkpoint}: {e}")
        return 1

    model_name = resolve_model_name(stored, model_name, logger)
    if model_name is None:
        return 1

    # Runs recorded before schema version 2 stored --sampling-rate as typed,
    # and only PTB-XL honoured it.
    legacy_rate = context.sampling_rate if dataset_name == "ptbxl" else None
    dataset_kwargs, multi_label = checkpoint_dataset_kwargs(
        dataset_name,
        stored,
        logger,
        force_download=force_download,
        data_dir=data_dir,
        sampling_rate=sampling_rate,
        legacy_sampling_rate=legacy_rate,
    )
    if not stored.get("train_config"):
        logger.warning(
            "Checkpoint records no train_config, so the dataset is built in its default "
            "labelling mode. Results will be wrong if the model was trained multi-label "
            "or as a collapsed binary problem."
        )

    try:
        data_module = build_data_module(
            dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            dataset_kwargs=dataset_kwargs,
            logger=logger,
        )
    except Exception as e:
        return report_data_error(e, dataset_name, logger, data_dir)

    try:
        if context.run_dir is not None:
            verify_split(data_module, context.run_dir, logger)

        mismatch = _check_target_shape(data_module, multi_label)
        if mismatch:
            logger.error(mismatch)
            return 1

        input_channels = infer_input_channels_from(data_module)

        logger.info(f"Loading model from {checkpoint}...")
        model, _ = build_model_from_checkpoint(
            checkpoint, name=model_name, input_channels=input_channels
        )
        trainer = ECGTrainer.load_checkpoint(checkpoint, model=model, device=accelerator)

        fit_targets = None
        fit_logits = None
        if calibrate and split != "val":
            logger.info("Running validation pass to fit calibration temperature...")
            try:
                _, _, fit_targets, _, fit_logits = _collect_split(trainer, data_module, "val")
            except Exception as e:
                logger.warning(f"Validation pass failed, so no temperature will be fitted: {e}")
                fit_targets = None
                fit_logits = None
        elif calibrate:
            logger.warning(
                "Reporting on the validation split, so no temperature is fitted. Fitting and "
                "reporting on the same split measures nothing."
            )

        logger.info(f"Running evaluation on {split} split...")
        results, predictions, targets, probabilities, logits = _collect_split(
            trainer, data_module, split
        )

        if results:
            logger.info("\nEvaluation Results:")
            logger.info("-" * 40)
            for key, value in results.items():
                logger.info(f"  {key}: {value:.4f}")

        if targets is not None and predictions is not None and probabilities is not None:
            output_path = prepare_output_dir(output_dir) if output_dir is not None else None
            class_names = getattr(data_module.dataset, "class_names", None)
            save_evaluation(
                targets,
                predictions,
                probabilities,
                model_name,
                dataset_name,
                class_names,
                output_path,
                logger,
                val_targets=fit_targets,
                val_logits=fit_logits,
                test_logits=logits,
                n_bins=calibration_bins,
            )
            if output_path is None:
                logger.info("Pass -o DIR to save these metrics, the predictions and the plots.")

        return 0
    except RuntimeError as e:
        return report_shape_error(e, model_name, "Evaluation", logger)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


def add_parser(subparsers: Any) -> Any:
    """Register the evaluate subcommand's arguments."""
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file, or a run directory whose best checkpoint is used",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        choices=get_model_names(),
        metavar="MODEL",
        help="Model architecture (default: the one recorded in the checkpoint)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=get_dataset_names(),
        metavar="DATASET",
        help="Dataset to evaluate on (see: deepecg list-datasets)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=defaults.BATCH_SIZE,
        help=f"Batch size (default: {defaults.BATCH_SIZE})",
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
        help="Deprecated and ignored: evaluation runs on a single device",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Sampling rate in Hz to resample the dataset to (default: the rate recorded "
        "in the checkpoint)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset even if it exists",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.SEED,
        help=f"Seed that produced the split during training (default: {defaults.SEED})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=defaults.VAL_SPLIT,
        help=f"Validation fraction used during training (default: {defaults.VAL_SPLIT})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=defaults.TEST_SPLIT,
        help=f"Test fraction used during training (default: {defaults.TEST_SPLIT})",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=DEFAULT_CALIBRATION_BINS,
        help=f"Bins for the calibration metrics (default: {DEFAULT_CALIBRATION_BINS})",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip the validation pass that fits the temperature",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Directory to write metrics CSV, predictions and plots into (default: console only)",
    )
    parser.set_defaults(handler="evaluate", to_kwargs=_to_kwargs)
    return parser


def _to_kwargs(args: Any) -> dict[str, Any]:
    return {
        "checkpoint": args.checkpoint,
        "dataset_name": args.dataset,
        "model_name": args.model,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "force_download": args.force_download,
        "split": args.split,
        "seed": args.seed,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "calibration_bins": args.calibration_bins,
        "calibrate": not args.no_calibration,
        "output_dir": args.output,
        "sampling_rate": args.sampling_rate,
    }
