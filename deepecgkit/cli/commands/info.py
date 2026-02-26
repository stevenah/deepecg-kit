"""Info and listing commands."""

from typing import Optional

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
