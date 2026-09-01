"""Shared validation and construction helpers for CLI commands (internal).

Every command used to restate the same guards (registry membership, file
existence, weight/model agreement) and the same data-module construction.
They live here once. The ``require_*`` helpers return an exit code (1) when
validation fails and None when it passes, so commands read as::

    if (code := require_model(model_name, logger)) is not None:
        return code
"""

from pathlib import Path
from typing import Any

from deepecgkit.cli import defaults
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    MODEL_NAMES,
    _get_dataset_registry,
    _get_model_registry,
)
from deepecgkit.datasets import ECGDataModule
from deepecgkit.models.pretrained import model_name_for_weights

#: Substrings of RuntimeError messages that indicate a signal-shape problem.
LENGTH_MARKERS = ("Kernel size", "must match the size of tensor", "needs at least")
LEAD_MARKER = "leads but received"


def require_model(model_name: str, logger: CLILogger) -> int | None:
    """Fail with the available models when model_name is not registered."""
    if model_name not in _get_model_registry():
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODEL_NAMES)}")
        return 1
    return None


def require_dataset(dataset_name: str | None, logger: CLILogger) -> int | None:
    """Fail with the available datasets when dataset_name is not registered.

    A None dataset_name passes: commands with a --dataset/--input choice call
    this unconditionally.
    """
    if dataset_name is not None and dataset_name not in _get_dataset_registry():
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
        return 1
    return None


def require_file(path: str | None, label: str, logger: CLILogger) -> int | None:
    """Fail when a path was given but does not exist.

    Args:
        path: The path to check, or None to skip the check
        label: How the file is referred to in the error ("Checkpoint",
            "Input file", ...)
        logger: CLI logger
    """
    if path is not None and not Path(path).exists():
        logger.error(f"{label} not found: {path}")
        return 1
    return None


def check_pretrained_weights(weights: str | None, model_name: str, logger: CLILogger) -> int | None:
    """Fail when named pretrained weights belong to a different model."""
    if weights is not None and not Path(weights).exists():
        target = model_name_for_weights(weights)
        if target is not None and target != model_name:
            logger.error(
                f"Pretrained weights '{weights}' belong to model '{target}', not '{model_name}'."
            )
            return 1
    return None


def dataset_kwargs_for(
    dataset_name: str,
    *,
    force_download: bool = False,
    data_dir: str | None = None,
    multi_label: bool = False,
    binary_classification: bool = False,
    sampling_rate: int | None = None,
    normalization: str | None = None,
) -> dict[str, Any]:
    """Build the per-dataset constructor kwargs the commands previously restated.

    Args:
        dataset_name: Registered dataset name
        force_download: Re-download the dataset
        data_dir: Explicit dataset directory
        multi_label: Multi-label task (PTB-XL)
        binary_classification: Collapse AF datasets to Non-AF/AF
        sampling_rate: Target sampling rate override
        normalization: Normalization method override
    """
    kwargs: dict[str, Any] = {"force_download": force_download}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if dataset_name == "ptbxl":
        kwargs["task"] = "superclass"
        kwargs["multi_label"] = multi_label
        if sampling_rate is not None:
            kwargs["sampling_rate"] = sampling_rate
            kwargs["use_high_resolution"] = sampling_rate > 100
        if normalization is not None:
            kwargs["normalization"] = normalization
    if dataset_name in defaults.AF_DATASETS:
        kwargs["binary_classification"] = binary_classification
    return kwargs


def build_data_module(
    dataset_name: str,
    *,
    batch_size: int,
    num_workers: int,
    val_split: float = defaults.VAL_SPLIT,
    test_split: float = defaults.TEST_SPLIT,
    seed: int = defaults.SEED,
    stratify: bool = True,
    dataset_kwargs: dict[str, Any] | None = None,
    stage: str | None = None,
    logger: CLILogger | None = None,
) -> ECGDataModule:
    """Construct and set up a data module under the training split protocol.

    Every command splits the dataset the same way so that "the test split"
    means the same samples in train, evaluate, extract-features, and explain.

    Args:
        dataset_name: Registered dataset name
        batch_size: Loader batch size
        num_workers: Loader workers
        val_split: Validation fraction (must match training to reproduce it)
        test_split: Test fraction
        seed: Split seed
        stratify: Stratify the split by label
        dataset_kwargs: Constructor kwargs from dataset_kwargs_for
        stage: Optional setup stage ("fit" restricts to train/val)
        logger: CLI logger for the progress line
    """
    if logger is not None:
        logger.info(f"Creating data module with {dataset_name} dataset...")
    data_module = ECGDataModule(
        dataset_class=_get_dataset_registry()[dataset_name],
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        stratify=stratify,
        download=True,
        dataset_kwargs=dataset_kwargs or {},
    )
    if stage is not None:
        data_module.setup(stage=stage)
    else:
        data_module.setup()
    return data_module


def infer_input_channels_from(data_module: ECGDataModule) -> int:
    """Read the lead count off the first sample of the dataset."""
    sample_signal, _ = data_module.dataset[0]
    return int(sample_signal.shape[0])


def prepare_output_dir(path: str | Path) -> Path:
    """Create the output directory (parents included) and return it."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def report_shape_error(error: Exception, model_name: str, action: str, logger: CLILogger) -> int:
    """Log a RuntimeError, translating known shape errors into guidance.

    Args:
        error: The caught exception
        model_name: Registered model name involved
        action: Capitalised description of what failed ("Feature extraction")
        logger: CLI logger

    Returns:
        The exit code 1, so callers can ``return report_shape_error(...)``.
    """
    message = str(error)
    if any(marker in message for marker in LENGTH_MARKERS) or LEAD_MARKER in message:
        logger.error(
            f"{action} failed: {message}\n"
            f"The signal shape is incompatible with '{model_name}'. Check the model's "
            f"length constraints, for example AFModel's recording_length, or the 5000 "
            f"sample ceiling on the attention architectures."
        )
    else:
        logger.error(f"{action} failed: {message}")
    return 1
