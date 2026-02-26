"""
Command-line interface for deepecg-kit.

Usage:
    deepecg train --model <model> --dataset <dataset> [options]
    deepecg evaluate --checkpoint <path> --dataset <dataset> [options]
    deepecg predict --checkpoint <path> --input <file> [options]
    deepecg resume --checkpoint <path> [options]
    deepecg list-models
    deepecg list-datasets
    deepecg info --model <model>

Datasets are automatically downloaded when first used if not already present.
Use --force-download to re-download an existing dataset.
"""

from deepecgkit.cli.commands import (
    evaluate,
    list_datasets,
    list_models,
    predict,
    resume,
    show_info,
    train,
)
from deepecgkit.cli.config import load_config
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.main import main
from deepecgkit.training import ECGLitModel
from deepecgkit.cli.registry import (
    DATASET_INPUT_CHANNELS,
    DATASET_NAMES,
    DATASET_NUM_CLASSES,
    MODEL_DESCRIPTIONS,
    MODEL_NAMES,
    MODEL_WEIGHTS,
    _get_dataset_registry,
    _get_model_registry,
)

__all__ = [
    "DATASET_INPUT_CHANNELS",
    "DATASET_NAMES",
    "DATASET_NUM_CLASSES",
    "MODEL_DESCRIPTIONS",
    "MODEL_NAMES",
    "MODEL_WEIGHTS",
    "CLILogger",
    "ECGLitModel",
    "_get_dataset_registry",
    "_get_model_registry",
    "evaluate",
    "list_datasets",
    "list_models",
    "load_config",
    "main",
    "predict",
    "resume",
    "show_info",
    "train",
]
