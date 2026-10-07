"""
Command-line interface for deepecg-kit.

Usage:
    deepecg train --model <model> --dataset <dataset> [options]
    deepecg evaluate --checkpoint <path> --dataset <dataset> [options]
    deepecg predict --checkpoint <path> --input <file> [options]
    deepecg extract-features --checkpoint <path> --dataset <dataset> --output <file> [options]
    deepecg explain --checkpoint <path> --input <file> --output <dir> [options]
    deepecg probe --model <model> --source-checkpoint <dataset>=<path> --output <dir> [options]
    deepecg list-models
    deepecg list-datasets
    deepecg info --model <model>

A checkpoint path may be a run directory, which stands for its best checkpoint.
Checkpoints record their architecture, so --model is only needed for older ones.

Datasets are automatically downloaded when first used if not already present.
Use --force-download to re-download an existing dataset.
"""

import importlib
from typing import TYPE_CHECKING, Any

from deepecgkit.cli.config import load_config
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.main import main

if TYPE_CHECKING:
    from deepecgkit.cli.commands.analysis import aggregate, rank
    from deepecgkit.cli.commands.evaluate import evaluate
    from deepecgkit.cli.commands.explain import explain
    from deepecgkit.cli.commands.extract import extract_features
    from deepecgkit.cli.commands.info import list_datasets, list_models, show_info
    from deepecgkit.cli.commands.predict import predict
    from deepecgkit.cli.commands.probe import probe
    from deepecgkit.cli.commands.train import train
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

#: The module behind each name that is imported on first use. The command
#: functions are looked up here at dispatch, which is what lets a run of one
#: command leave the others, and whatever they import, alone.
_LAZY = {
    "aggregate": "deepecgkit.cli.commands.analysis",
    "rank": "deepecgkit.cli.commands.analysis",
    "evaluate": "deepecgkit.cli.commands.evaluate",
    "explain": "deepecgkit.cli.commands.explain",
    "extract_features": "deepecgkit.cli.commands.extract",
    "list_datasets": "deepecgkit.cli.commands.info",
    "list_models": "deepecgkit.cli.commands.info",
    "show_info": "deepecgkit.cli.commands.info",
    "predict": "deepecgkit.cli.commands.predict",
    "probe": "deepecgkit.cli.commands.probe",
    "train": "deepecgkit.cli.commands.train",
    "DATASET_INPUT_CHANNELS": "deepecgkit.cli.registry",
    "DATASET_NAMES": "deepecgkit.cli.registry",
    "DATASET_NUM_CLASSES": "deepecgkit.cli.registry",
    "MODEL_DESCRIPTIONS": "deepecgkit.cli.registry",
    "MODEL_NAMES": "deepecgkit.cli.registry",
    "MODEL_WEIGHTS": "deepecgkit.cli.registry",
    "ECGTrainer": "deepecgkit.training",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
