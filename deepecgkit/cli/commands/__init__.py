"""CLI command functions."""

from deepecgkit.cli.commands.analysis import aggregate, rank
from deepecgkit.cli.commands.evaluate import evaluate
from deepecgkit.cli.commands.explain import explain
from deepecgkit.cli.commands.extract import extract_features
from deepecgkit.cli.commands.info import list_datasets, list_models, show_info
from deepecgkit.cli.commands.predict import predict
from deepecgkit.cli.commands.probe import probe
from deepecgkit.cli.commands.train import train

__all__ = [
    "aggregate",
    "evaluate",
    "explain",
    "extract_features",
    "list_datasets",
    "list_models",
    "predict",
    "probe",
    "rank",
    "show_info",
    "train",
]
