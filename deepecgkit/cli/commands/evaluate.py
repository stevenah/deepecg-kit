"""Evaluate command."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    MODEL_NAMES,
    _get_dataset_registry,
    _get_model_registry,
)
from deepecgkit.datasets import ECGDataModule
from deepecgkit.evaluation.calibration import calibration_report
from deepecgkit.evaluation.metrics import DEFAULT_CALIBRATION_BINS
from deepecgkit.models.factory import build_model_from_checkpoint
from deepecgkit.training import ECGTrainer

AF_DATASETS = {"mitbih-afdb", "ltafdb", "unified-af"}
DEFAULT_SEED = 42
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.1


def _read_train_config(checkpoint: str) -> Dict[str, Any]:
    """Read the training configuration recorded in a checkpoint.

    Args:
        checkpoint: Path to the checkpoint file

    Returns:
        The stored train_config, or an empty mapping when it is absent.
    """
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(stored, dict):
        return {}
    config = stored.get("train_config")
    return config if isinstance(config, dict) else {}


def _dataset_kwargs_for_checkpoint(
    dataset_name: str,
    train_config: Dict[str, Any],
    force_download: bool,
    data_dir: Optional[str],
) -> Tuple[Dict[str, Any], bool]:
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
    multi_label = bool(train_config.get("multi_label", False))
    binary_classification = bool(train_config.get("binary_classification", False))

    kwargs: Dict[str, Any] = {"force_download": force_download}
    if data_dir:
        kwargs["data_dir"] = data_dir

    if dataset_name == "ptbxl":
        kwargs["task"] = "superclass"
        kwargs["multi_label"] = multi_label

    if dataset_name in AF_DATASETS:
        kwargs["binary_classification"] = binary_classification

    return kwargs, multi_label


def _check_target_shape(data_module: ECGDataModule, multi_label: bool) -> Optional[str]:
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


def _collect_split(trainer: ECGTrainer, data_module: ECGDataModule, split: str) -> Tuple:
    """Run one split and take a copy of the arrays it produced.

    ECGTrainer reuses a single set of buffers for every evaluation pass, so the
    results of one split must be copied out before the next pass clears them.

    Args:
        trainer: Trainer holding the loaded model
        data_module: Data module supplying the loaders
        split: Either "test" or "val"

    Returns:
        Tuple of (summary metrics, targets, probabilities, logits).
    """
    results = trainer.test(data_module) if split == "test" else trainer.validate(data_module)
    _, targets, probabilities = trainer.get_test_results()
    return results, targets, probabilities, trainer.get_test_logits()


def _log_calibration(report: Dict[str, Any], n_bins: int, logger: CLILogger) -> None:
    """Print the calibration block, before and after temperature scaling.

    Args:
        report: Mapping from calibration_report
        n_bins: Bin count that produced the binned metrics
        logger: CLI logger
    """
    logger.info(f"\nCalibration ({n_bins} bins):")
    logger.info("-" * 40)
    scaled = not np.isnan(report["temperature"])

    if not scaled:
        for name, label in (
            ("ece", "ECE"),
            ("adaptive_ece", "Adaptive ECE"),
            ("mce", "MCE"),
            ("brier", "Brier"),
        ):
            logger.info(f"  {label:<16} {report[name]:>10.4f}")
        logger.info("  No temperature was fitted, so no rescaled figures are shown.")
        return

    logger.info(f"  {'Metric':<16} {'Before':>10} {'After T':>10}")
    for name, label in (
        ("ece", "ECE"),
        ("adaptive_ece", "Adaptive ECE"),
        ("mce", "MCE"),
        ("brier", "Brier"),
    ):
        logger.info(f"  {label:<16} {report[name]:>10.4f} {report[name + '_temp']:>10.4f}")
    logger.info(f"  {'Temperature':<16} {report['temperature']:>21.4f}")
    logger.info("  Temperature fitted on the validation split, reported on the evaluated split.")


def evaluate(
    checkpoint: str,
    dataset_name: str,
    model_name: str,
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: int = 1,
    force_download: bool = False,
    split: str = "test",
    seed: int = DEFAULT_SEED,
    val_split: float = DEFAULT_VAL_SPLIT,
    test_split: float = DEFAULT_TEST_SPLIT,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    calibrate: bool = True,
    logger: Optional[CLILogger] = None,
) -> int:
    """Evaluate a trained model on a dataset.

    The split is redrawn here rather than read back from the run, so seed,
    val_split and test_split must match the values the checkpoint was trained
    with. Mismatched values silently produce a different partition, which for
    calibration matters more than usual: a temperature fitted on a validation
    split that overlaps the original training data is fitted on data the model
    has already seen.

    Args:
        checkpoint: Path to the checkpoint to evaluate
        dataset_name: Registered dataset name
        model_name: Registered model name
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
        logger: CLI logger

    Returns:
        Zero on success, one on failure.
    """
    logger = logger or CLILogger()

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    dataset_registry = _get_dataset_registry()
    model_registry = _get_model_registry()

    if dataset_name not in dataset_registry:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
        return 1

    if model_name not in model_registry:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODEL_NAMES)}")
        return 1

    if split not in ("test", "val"):
        logger.error(f"Invalid split: {split}. Use 'test' or 'val'.")
        return 1

    try:
        train_config = _read_train_config(checkpoint)
        dataset_kwargs, multi_label = _dataset_kwargs_for_checkpoint(
            dataset_name, train_config, force_download, data_dir
        )
        if not train_config:
            logger.warning(
                "Checkpoint records no train_config, so the dataset is built in its default "
                "labelling mode. Results will be wrong if the model was trained multi-label "
                "or as a collapsed binary problem."
            )

        logger.info(f"Creating data module with {dataset_name} dataset...")
        data_module = ECGDataModule(
            dataset_class=dataset_registry[dataset_name],
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            stratify=True,
            download=True,
            dataset_kwargs=dataset_kwargs,
        )
        data_module.setup()

        mismatch = _check_target_shape(data_module, multi_label)
        if mismatch:
            logger.error(mismatch)
            return 1

        sample_signal, _ = data_module.dataset[0]
        input_channels = int(sample_signal.shape[0])

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
                _, fit_targets, _, fit_logits = _collect_split(trainer, data_module, "val")
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
        results, targets, probabilities, logits = _collect_split(trainer, data_module, split)

        if results:
            logger.info("\nEvaluation Results:")
            logger.info("-" * 40)
            for key, value in results.items():
                logger.info(f"  {key}: {value:.4f}")

        if targets is not None and probabilities is not None:
            report = calibration_report(
                targets,
                probabilities,
                n_bins=calibration_bins,
                fit_targets=fit_targets,
                fit_logits=fit_logits,
                eval_logits=logits,
            )
            _log_calibration(report, calibration_bins, logger)

        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
