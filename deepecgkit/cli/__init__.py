"""
Command-line interface for deepecg-kit.

Usage:
    deepecg train --model <model> --dataset <dataset> [options]
    deepecg evaluate --checkpoint <path> --dataset <dataset> [options]
    deepecg predict --checkpoint <path> --input <file> [options]
    deepecg extract-features --checkpoint <path> --model <model> --dataset <dataset> --output <file> [options]
    deepecg explain --checkpoint <path> --model <model> --input <file> --output <dir> [options]
    deepecg probe --model <model> --source-checkpoint <dataset>=<path> --output <dir> [options]
    deepecg list-models
    deepecg list-datasets
    deepecg info --model <model>

Datasets are automatically downloaded when first used if not already present.
Use --force-download to re-download an existing dataset.
"""

from deepecgkit.cli.commands import (
    aggregate,
    evaluate,
    explain,
    extract_features,
    list_datasets,
    list_models,
    predict,
    probe,
    rank,
    show_info,
    train,
)
from deepecgkit.cli.config import load_config
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.main import main
from deepecgkit.cli.registry import (
    DATASET_INPUT_CHANNELS,
    DATASET_NAMES,
    DATASET_NUM_CLASSES,
    MODEL_DESCRIPTIONS,
    MODEL_NAMES,
    MODEL_WEIGHTS,
)
from deepecgkit.training import ECGTrainer

__all__ = [
    "DATASET_INPUT_CHANNELS",
    "DATASET_NAMES",
    "DATASET_NUM_CLASSES",
    "MODEL_DESCRIPTIONS",
    "MODEL_NAMES",
    "MODEL_WEIGHTS",
    "CLILogger",
    "ECGTrainer",
    "aggregate",
    "evaluate",
    "explain",
    "extract_features",
    "list_datasets",
    "list_models",
    "load_config",
    "main",
    "predict",
    "probe",
    "rank",
    "show_info",
    "train",
]
