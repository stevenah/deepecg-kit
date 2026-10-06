"""Info and listing commands."""

import inspect
import traceback
from typing import Any

from deepecgkit.cli._common import dataset_accepts, require_model
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import _get_model_registry
from deepecgkit.models.pretrained import model_name_for_weights
from deepecgkit.registry import (
    get_dataset_info,
    get_dataset_names,
    get_model_info,
    get_model_names,
)
from deepecgkit.utils.weights import WEIGHTS_REGISTRY, is_published


def _published_weights(model_name: str) -> list[str]:
    """Names of the downloadable pretrained weights that target a model."""
    return [
        name
        for name in WEIGHTS_REGISTRY
        if is_published(name) and model_name_for_weights(name) == model_name
    ]


def _length_limits(info: dict[str, Any]) -> str | None:
    """Describe a model's input-length limits, or None when it has none."""
    minimum, maximum = info.get("min_input_length"), info.get("max_input_length")
    if minimum and maximum:
        return f"{minimum} to {maximum} samples per lead"
    if minimum:
        return f"at least {minimum} samples per lead"
    if maximum:
        return f"at most {maximum} samples per lead"
    return None


def list_models() -> None:
    """List all available models."""
    print("Available models:")
    print("-" * 60)
    for name in get_model_names():
        info = get_model_info(name)
        weights = _published_weights(name)
        limits = _length_limits(info)
        print(f"  {name}")
        if info.get("description"):
            print(f"    Description: {info['description']}")
        if limits:
            print(f"    Input length: {limits}")
        if weights:
            print(f"    Pretrained weights: {', '.join(weights)}")
        print()


def _default_of(dataset_class: type, parameter: str) -> Any:
    """Read a constructor default off a dataset class, or None when absent."""
    found = inspect.signature(dataset_class).parameters.get(parameter)
    if found is None or found.default is inspect.Parameter.empty:
        return None
    return found.default


def list_datasets() -> None:
    """List all available datasets, with what a download involves."""
    print("Available datasets:")
    print("-" * 60)
    for name in get_dataset_names():
        info = get_dataset_info(name)
        dataset_class = info["class"]
        class_labels = list(
            getattr(dataset_class, "CLASS_LABELS", None)
            or getattr(dataset_class, "CLASS_LABELS_SUPERCLASS", None)
            or []
        )
        classes = str(info["num_classes"])
        if class_labels and len(class_labels) == info["num_classes"]:
            classes += f" ({', '.join(class_labels)})"

        rate = _default_of(dataset_class, "sampling_rate")
        duration = _default_of(dataset_class, "segment_duration_seconds")
        signal = f"{info['input_channels']} lead(s)"
        if rate:
            signal += f", {rate} Hz"
        if duration:
            signal += f", {duration:g} s segments"

        options = [
            flag
            for option, flag in (
                ("binary_classification", "--binary-classification"),
                ("multi_label", "--multi-label"),
                ("segment_duration_seconds", "--segment-duration"),
            )
            if dataset_accepts(name, option)
        ]

        print(f"  {name}")
        if info.get("description"):
            print(f"    Description: {info['description']}")
        print(f"    Classes: {classes}")
        print(f"    Signal: {signal}")
        if options:
            print(f"    Options: {', '.join(options)}")
        size = getattr(dataset_class, "DOWNLOAD_SIZE", None)
        get_dir = getattr(dataset_class, "get_default_data_dir", None)
        is_downloaded = getattr(dataset_class, "is_downloaded", None)
        if callable(get_dir) and callable(is_downloaded):
            try:
                present = bool(is_downloaded())
            except Exception:
                present = False
            status = "downloaded" if present else f"not downloaded ({size or 'size unknown'})"
            print(f"    Local copy: {status}, {get_dir()}")
        print()


def show_info(
    model_name: str,
    input_channels: int = 1,
    num_classes: int = 4,
    logger: CLILogger | None = None,
) -> int:
    """Show detailed information about a model."""
    logger = logger or CLILogger()

    if (code := require_model(model_name, logger)) is not None:
        return code

    try:
        model_registry = _get_model_registry()
        model_class = model_registry[model_name]
        model = model_class(input_channels=input_channels, output_size=num_classes)
        info = get_model_info(model_name)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model: {model_name}")
        print("=" * 60)
        print(f"Description: {info.get('description') or 'N/A'}")
        print(f"Parameter counts for a {input_channels}-lead, {num_classes}-class configuration:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if hasattr(model, "feature_dim"):
            print(f"Feature dimension: {model.feature_dim}")
        print(f"Input length: {_length_limits(info) or 'any'}")
        print(
            f"Pretrained weights: {', '.join(_published_weights(model_name)) or 'none published'}"
        )
        print()
        print("Architecture:")
        print("-" * 60)
        print(model)

        return 0
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        logger.debug(traceback.format_exc())
        return 1


def add_parser(subparsers: Any) -> Any:
    """Register the info, list-models and list-datasets subcommands."""
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=get_model_names(),
        metavar="MODEL",
        help="Model to show information for (see: deepecg list-models)",
    )
    info_parser.add_argument(
        "--input-channels",
        type=int,
        default=1,
        help="Lead count used for the parameter counts (default: 1)",
    )
    info_parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Class count used for the parameter counts (default: 4)",
    )
    info_parser.set_defaults(
        handler="show_info",
        to_kwargs=lambda args: {
            "model_name": args.model,
            "input_channels": args.input_channels,
            "num_classes": args.num_classes,
        },
    )

    list_models_parser = subparsers.add_parser("list-models", help="List all available models")
    list_models_parser.set_defaults(
        handler="list_models", to_kwargs=lambda args: {}, pass_logger=False
    )

    list_datasets_parser = subparsers.add_parser(
        "list-datasets", help="List all available datasets"
    )
    list_datasets_parser.set_defaults(
        handler="list_datasets", to_kwargs=lambda args: {}, pass_logger=False
    )
    return info_parser
