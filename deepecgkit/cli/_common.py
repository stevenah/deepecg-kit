"""Shared validation and construction helpers for CLI commands (internal).

Every command used to restate the same guards (registry membership, file
existence, weight/model agreement) and the same data-module construction.
They live here once. The ``require_*`` helpers return an exit code (1) when
validation fails and None when it passes, so commands read as::

    if (code := require_model(model_name, logger)) is not None:
        return code
"""

import inspect
import math
import traceback
from pathlib import Path
from typing import Any

from deepecgkit.cli import defaults
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import _get_dataset_registry, _get_model_registry
from deepecgkit.cli.run_dir import resolve_checkpoint
from deepecgkit.datasets import ECGDataModule
from deepecgkit.models.factory import recorded_model_name
from deepecgkit.models.pretrained import model_name_for_weights
from deepecgkit.registry import get_dataset_names, get_model_info, get_model_names
from deepecgkit.utils.checkpoints import load_checkpoint_file

#: Substrings of RuntimeError messages that indicate a signal-shape problem.
LENGTH_MARKERS = ("Kernel size", "must match the size of tensor", "needs at least")
LEAD_MARKER = "leads but received"
#: Substrings that indicate the device ran out of memory.
MEMORY_MARKERS = ("out of memory", "Invalid buffer size", "not enough memory")

#: Sampling rate every dataset was resampled to by releases that did not yet
#: record preprocessing in their checkpoints.
LEGACY_SAMPLING_RATE = 500

#: Command-line flag for each dataset constructor option a command can set.
_OPTION_FLAGS = {
    "multi_label": "--multi-label",
    "binary_classification": "--binary-classification",
    "sampling_rate": "--sampling-rate",
    "normalization": "--normalization",
    "segment_duration_seconds": "--segment-duration",
}


def require_model(model_name: str, logger: CLILogger) -> int | None:
    """Fail with the available models when model_name is not registered."""
    if model_name not in _get_model_registry():
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(get_model_names())}")
        return 1
    return None


def require_dataset(dataset_name: str | None, logger: CLILogger) -> int | None:
    """Fail with the available datasets when dataset_name is not registered.

    A None dataset_name passes: commands with a --dataset/--input choice call
    this unconditionally.
    """
    if dataset_name is not None and dataset_name not in _get_dataset_registry():
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(get_dataset_names())}")
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


def require_checkpoint(checkpoint: str, logger: CLILogger) -> str | None:
    """Resolve a --checkpoint value to a checkpoint file.

    A run directory (or its checkpoints/ folder) stands for the best
    checkpoint inside it, so the path printed at the end of training can be
    passed along as it is.

    Args:
        checkpoint: The --checkpoint value
        logger: CLI logger

    Returns:
        The checkpoint file to load, or None after the reason was logged.
    """
    if require_file(checkpoint, "Checkpoint", logger) is not None:
        return None
    try:
        resolved = resolve_checkpoint(checkpoint)
    except FileNotFoundError as error:
        logger.error(str(error))
        return None
    if str(resolved) != str(checkpoint):
        logger.info(f"Using checkpoint {resolved}")
    return str(resolved)


def read_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    """Load a checkpoint file, returning an empty mapping for anything else.

    Args:
        checkpoint: Path to the checkpoint file

    Returns:
        The checkpoint dictionary, or an empty one when the file holds
        something other than a dictionary.
    """
    stored = load_checkpoint_file(checkpoint)
    return stored if isinstance(stored, dict) else {}


def resolve_model_name(
    stored: dict[str, Any], requested: str | None, logger: CLILogger
) -> str | None:
    """Decide which architecture a checkpoint is loaded as.

    Checkpoints written by this version name their architecture, which makes
    --model optional and lets a contradicting value be reported in one line
    instead of as a list of mismatched tensors.

    Args:
        stored: The loaded checkpoint
        requested: The --model value, or None when it was left out
        logger: CLI logger

    Returns:
        The model name, or None after the reason was logged.
    """
    recorded = recorded_model_name(stored)
    if requested is None:
        if recorded is None:
            logger.error(
                "This checkpoint does not record which architecture it holds; pass --model."
            )
            return None
        requested = recorded
    elif recorded is not None and recorded != requested:
        logger.error(
            f"The checkpoint was trained as '{recorded}', but --model {requested} was given. "
            f"Leave --model out, or pass --model {recorded}."
        )
        return None
    if require_model(requested, logger) is not None:
        return None
    return requested


def check_pretrained_weights(
    weights: str | None, model_name: str | None, logger: CLILogger
) -> int | None:
    """Fail when named pretrained weights belong to a different model."""
    if weights is not None and model_name is not None and not Path(weights).exists():
        target = model_name_for_weights(weights)
        if target is not None and target != model_name:
            logger.error(
                f"Pretrained weights '{weights}' belong to model '{target}', not '{model_name}'."
            )
            return 1
    return None


def dataset_accepts(dataset_name: str, option: str) -> bool:
    """Report whether a dataset's constructor takes the given option.

    Reading this off the constructor keeps the answer right for datasets
    registered outside the package.

    Args:
        dataset_name: Registered dataset name
        option: Constructor parameter name, such as "binary_classification"
    """
    return option in inspect.signature(_get_dataset_registry()[dataset_name]).parameters


def check_dataset_options(
    dataset_name: str,
    logger: CLILogger,
    *,
    multi_label: bool = False,
    binary_classification: bool = False,
    sampling_rate: int | None = None,
    normalization: str | None = None,
    segment_duration: float | None = None,
) -> int | None:
    """Fail when a flag was given that the dataset cannot honour.

    A flag the dataset ignores would otherwise be recorded as part of the run
    while having had no effect on it.

    Args:
        dataset_name: Registered dataset name
        logger: CLI logger
        multi_label: Whether --multi-label was given
        binary_classification: Whether --binary-classification was given
        sampling_rate: The --sampling-rate value, if any
        normalization: The --normalization value, if any
        segment_duration: The --segment-duration value, if any

    Returns:
        An exit code when a flag does not apply, or None when all of them do.
    """
    given = {
        "multi_label": multi_label,
        "binary_classification": binary_classification,
        "sampling_rate": sampling_rate is not None,
        "normalization": normalization is not None,
        "segment_duration_seconds": segment_duration is not None,
    }
    for option, was_given in given.items():
        if not was_given or dataset_accepts(dataset_name, option):
            continue
        supported = [name for name in get_dataset_names() if dataset_accepts(name, option)]
        logger.error(
            f"{_OPTION_FLAGS[option]} does not apply to the '{dataset_name}' dataset. "
            f"Datasets that accept it: {', '.join(supported) or 'none'}."
        )
        return 1
    if segment_duration is not None and segment_duration <= 0:
        logger.error(f"--segment-duration must be positive, got {segment_duration}")
        return 1
    if sampling_rate is not None and sampling_rate <= 0:
        logger.error(f"--sampling-rate must be positive, got {sampling_rate}")
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
    segment_duration: float | None = None,
) -> dict[str, Any]:
    """Build the per-dataset constructor kwargs the commands previously restated.

    Options left at None keep the dataset's own default. An option the
    dataset does not take is rejected rather than dropped; commands screen
    for that first with check_dataset_options so the user sees a plain error.

    Args:
        dataset_name: Registered dataset name
        force_download: Re-download the dataset
        data_dir: Explicit dataset directory
        multi_label: Multi-label task (PTB-XL)
        binary_classification: Collapse AF datasets to Non-AF/AF
        sampling_rate: Target sampling rate override
        normalization: Normalization method override
        segment_duration: Segment length override, in seconds

    Raises:
        ValueError: If an option is set that the dataset does not accept.
    """
    kwargs: dict[str, Any] = {"force_download": force_download}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if dataset_name == "ptbxl":
        kwargs["task"] = "superclass"
        if sampling_rate is not None:
            kwargs["use_high_resolution"] = sampling_rate > 100

    requested: dict[str, Any] = {
        "multi_label": multi_label or None,
        "binary_classification": binary_classification or None,
        "sampling_rate": sampling_rate,
        "normalization": normalization,
        "segment_duration_seconds": segment_duration,
    }
    for option, value in requested.items():
        if value is None:
            continue
        if not dataset_accepts(dataset_name, option):
            raise ValueError(
                f"{_OPTION_FLAGS[option]} does not apply to the '{dataset_name}' dataset."
            )
        kwargs[option] = value

    # The labelling mode is always stated for the datasets that have one, so
    # their own defaults (PTB-XL's is multi-label) never decide it.
    for option, value in (
        ("multi_label", multi_label),
        ("binary_classification", binary_classification),
    ):
        if dataset_accepts(dataset_name, option):
            kwargs[option] = value
    return kwargs


def checkpoint_dataset_kwargs(
    dataset_name: str,
    stored: dict[str, Any],
    logger: CLILogger,
    *,
    force_download: bool = False,
    data_dir: str | None = None,
    sampling_rate: int | None = None,
    legacy_sampling_rate: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Rebuild a dataset the way a checkpoint's model expects its inputs.

    The labelling mode and the preprocessing both come from the checkpoint:
    a model trained on z-scored 300 Hz segments must be evaluated on z-scored
    300 Hz segments, whichever dataset they are drawn from.

    Checkpoints from before preprocessing was recorded were all trained by a
    CLI that resampled to 500 Hz, so that rate is assumed for them and the
    assumption is reported.

    Args:
        dataset_name: Registered dataset name
        stored: The loaded checkpoint
        logger: CLI logger
        force_download: Whether to re-download the dataset
        data_dir: Optional dataset directory
        sampling_rate: Explicit --sampling-rate, overriding the checkpoint
        legacy_sampling_rate: Rate recorded next to a legacy checkpoint that
            can be trusted over the 500 Hz assumption

    Returns:
        Tuple of (dataset keyword arguments, whether targets are multi-label).
    """
    train_config = stored.get("train_config")
    train_config = train_config if isinstance(train_config, dict) else {}
    data_config = stored.get("data_config")
    data_config = data_config if isinstance(data_config, dict) else {}
    preprocessing = data_config.get("preprocessing") or {}

    multi_label = bool(train_config.get("multi_label", False))
    binary_classification = bool(train_config.get("binary_classification", False))

    options: dict[str, Any] = {}
    if preprocessing:
        options = {
            "sampling_rate": preprocessing.get("sampling_rate"),
            "normalization": preprocessing.get("normalization"),
            "segment_duration": preprocessing.get("segment_duration_seconds"),
        }
    elif stored.get("model_state_dict") or stored.get("state_dict"):
        assumed = legacy_sampling_rate or LEGACY_SAMPLING_RATE
        options = {"sampling_rate": assumed}
        if sampling_rate is None:
            logger.warning(
                f"This checkpoint does not record its preprocessing, so the dataset is "
                f"resampled to {assumed} Hz, the rate earlier releases trained at. Pass "
                f"--sampling-rate if it was trained at another rate."
            )
    if sampling_rate is not None:
        options["sampling_rate"] = sampling_rate

    option_names = {
        "sampling_rate": "sampling_rate",
        "normalization": "normalization",
        "segment_duration": "segment_duration_seconds",
    }
    accepted = {
        key: value
        for key, value in options.items()
        if value is not None and dataset_accepts(dataset_name, option_names[key])
    }
    kwargs = dataset_kwargs_for(
        dataset_name,
        force_download=force_download,
        data_dir=data_dir,
        multi_label=multi_label and dataset_accepts(dataset_name, "multi_label"),
        binary_classification=binary_classification
        and dataset_accepts(dataset_name, "binary_classification"),
        **accepted,
    )
    return kwargs, multi_label


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


def report_data_error(
    error: Exception, dataset_name: str, logger: CLILogger, data_dir: str | None = None
) -> int:
    """Log a failure to load a dataset, keeping the detail the user needs to act.

    Args:
        error: The caught exception
        dataset_name: Registered dataset name
        logger: CLI logger
        data_dir: The --data-dir value, if one was given

    Returns:
        The exit code 1, so callers can ``return report_data_error(...)``.
    """
    detail = str(error) or type(error).__name__
    if isinstance(error, FileNotFoundError):
        logger.error(f"Dataset '{dataset_name}' not found. {detail}")
        if data_dir:
            logger.info(f"Verify the data directory exists: {data_dir}")
    else:
        logger.error(f"Failed to load the '{dataset_name}' dataset: {detail}")
    logger.debug(traceback.format_exc())
    return 1


def split_record_names(data_module: ECGDataModule, split: str = "test") -> list[str] | None:
    """Name the recording behind each sample of a split, in the order it is evaluated.

    Args:
        data_module: The data module, after setup
        split: "test" or "val"

    Returns:
        One name per sample, or None when the dataset does not say which
        recording its samples come from.
    """
    dataset = data_module.dataset
    names = getattr(dataset, "record_names", None) or getattr(dataset, "group_ids", None)
    indices = getattr(getattr(data_module, f"{split}_dataset", None), "indices", None)
    if not names or indices is None or any(index >= len(names) for index in indices):
        return None
    return [str(names[index]) for index in indices]


def describe_data(data_module: ECGDataModule, dataset_name: str) -> dict[str, Any]:
    """Describe the loaded data in the form checkpoints and run metadata record.

    Args:
        data_module: A data module whose setup has already run
        dataset_name: Registered dataset name the command was given
    """
    description = data_module.describe()
    description["dataset"] = dataset_name
    return description


def _first_signal(data_module: ECGDataModule) -> Any:
    """Return the first sample's signal from a data module that has been set up."""
    dataset = data_module.dataset
    if dataset is None:
        raise RuntimeError("The data module has not been set up.")
    sample_signal, _ = dataset[0]
    return sample_signal


def infer_input_channels_from(data_module: ECGDataModule) -> int:
    """Read the lead count off the first sample of the dataset."""
    return int(_first_signal(data_module).shape[0])


def check_input_length(
    model_name: str, dataset_name: str, data_module: ECGDataModule, logger: CLILogger
) -> int | None:
    """Fail before training when the model cannot take the dataset's sample length.

    Args:
        model_name: Registered model name
        dataset_name: Registered dataset name
        data_module: A data module whose setup has already run
        logger: CLI logger

    Returns:
        An exit code when the length is outside the model's limits, else None.
    """
    info = get_model_info(model_name)
    minimum, maximum = info.get("min_input_length"), info.get("max_input_length")
    length = int(_first_signal(data_module).shape[-1])

    too_short = minimum is not None and length < minimum
    too_long = maximum is not None and length > maximum
    if not too_short and not too_long:
        return None

    preprocessing: dict[str, Any] = {}
    get_preprocessing = getattr(data_module.dataset, "get_preprocessing", None)
    if callable(get_preprocessing):
        preprocessing = get_preprocessing() or {}
    rate = preprocessing.get("sampling_rate")
    duration = preprocessing.get("segment_duration_seconds")
    shape = f"{length} samples per lead"
    if rate and duration:
        shape += f" ({duration:g} s at {rate:g} Hz)"

    can_resegment = dataset_accepts(dataset_name, "segment_duration_seconds")
    if too_short:
        message = f"'{model_name}' needs at least {minimum} samples per lead, but "
        fix = "Use --sampling-rate to raise the rate, or choose another model."
        if rate and can_resegment:
            seconds = math.ceil(minimum / rate)
            fix = f"Pass --segment-duration {seconds} or longer, or choose another model."
    else:
        message = f"'{model_name}' accepts at most {maximum} samples per lead, but "
        fix = "Lower --sampling-rate, or choose another model."
        if rate and can_resegment:
            seconds = math.floor(maximum / rate)
            fix = (
                f"Pass --segment-duration {seconds} or shorter, lower --sampling-rate, "
                f"or choose another model."
            )
    logger.error(f"{message}'{dataset_name}' yields {shape}. {fix}")
    return 1


def prepare_output_dir(path: str | Path) -> Path:
    """Create the output directory (parents included) and return it."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def report_shape_error(error: Exception, model_name: str, action: str, logger: CLILogger) -> int:
    """Log a RuntimeError, translating known shape and memory errors into guidance.

    Args:
        error: The caught exception
        model_name: Registered model name involved
        action: Capitalised description of what failed ("Feature extraction")
        logger: CLI logger

    Returns:
        The exit code 1, so callers can ``return report_shape_error(...)``.
    """
    message = str(error)
    if any(marker in message for marker in MEMORY_MARKERS):
        logger.error(
            f"{action} failed: the device ran out of memory ({message.splitlines()[0]}).\n"
            f"Lower --batch-size first. The attention models (transformer, dualnet) need "
            f"memory that grows with the square of the signal length, so a lower "
            f"--sampling-rate or shorter --segment-duration helps them most."
        )
    elif any(marker in message for marker in LENGTH_MARKERS) or LEAD_MARKER in message:
        logger.error(
            f"{action} failed: {message}\n"
            f"The signal shape is incompatible with '{model_name}'. Check the model's "
            f"length constraints, for example AFModel's recording_length, or the 5000 "
            f"sample ceiling on the attention architectures."
        )
    else:
        logger.error(f"{action} failed: {message}")
    logger.debug(traceback.format_exc())
    return 1
