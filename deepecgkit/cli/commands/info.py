"""Info and listing commands."""

import traceback
from typing import Any

from deepecgkit.cli._common import require_model
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    DATASET_NUM_CLASSES,
    MODEL_DESCRIPTIONS,
    MODEL_NAMES,
    MODEL_WEIGHTS,
    _get_model_registry,
)


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

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model: {model_name}")
        print("=" * 60)
        print(f"Description: {MODEL_DESCRIPTIONS.get(model_name, 'N/A')}")
        print(f"Parameter counts for a {input_channels}-lead, {num_classes}-class configuration:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if hasattr(model, "feature_dim"):
            print(f"Feature dimension: {model.feature_dim}")
        print(f"Pretrained weights: {', '.join(MODEL_WEIGHTS.get(model_name, ['None']))}")
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
        choices=MODEL_NAMES,
        help="Model to show information for",
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
