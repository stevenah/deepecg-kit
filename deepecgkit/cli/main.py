"""Main CLI entry point with argument parsing and command routing."""

import argparse
import sys
from typing import Optional

import deepecgkit.cli as _cli
from deepecgkit.cli.config import load_config
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import DATASET_NAMES, MODEL_NAMES


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="deepecg",
        description="DeepECG-Kit: Deep learning toolkit for ECG analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deepecg train -m kanres -d af-classification
  deepecg train -m kanres -d af-classification --force-download
  deepecg evaluate --checkpoint model.ckpt -d af-classification
  deepecg predict --checkpoint model.ckpt --input ecg.npy
  deepecg resume --checkpoint model.ckpt --epochs 100
  deepecg list-models
  deepecg info -m kanres

Note: Datasets are automatically downloaded when first used.
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model on a dataset")
    train_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to train",
    )
    train_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Dataset to train on",
    )
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset (default: auto-detect)",
    )
    train_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: runs/{timestamp}-{model}-{dataset})",
    )
    train_parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to weights file or pretrained weight name",
    )
    train_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    train_parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    train_parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    train_parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    train_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    train_parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    train_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset even if it exists",
    )
    train_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--multi-label",
        action="store_true",
        default=False,
        help="Use multi-label classification with BCE loss (default: single-label with CE loss)",
    )
    train_parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Target sampling rate in Hz for ECG signals (e.g., 100 or 500 for PTB-XL)",
    )
    train_parser.add_argument(
        "--binary-classification",
        action="store_true",
        default=False,
        help="Use binary classification (AF vs Non-AF) for AF datasets (default: 4-class)",
    )
    train_parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["zscore", "minmax", "none"],
        help="Normalization method for ECG signals (default: dataset-specific)",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Dataset to evaluate on",
    )
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset",
    )
    eval_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    eval_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    eval_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    eval_parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    eval_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset even if it exists",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to evaluate on (default: test)",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference on ECG data")
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    predict_parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input ECG file (.npy, .csv, or WFDB .dat/.hea)",
    )
    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save predictions (JSON format)",
    )
    predict_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    predict_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume training from a checkpoint")
    resume_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to resume from",
    )
    resume_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Additional epochs to train (default: continue original)",
    )
    resume_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save new checkpoints",
    )
    resume_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    resume_parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    resume_parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model to show information for",
    )

    subparsers.add_parser("list-models", help="List all available models")
    subparsers.add_parser("list-datasets", help="List all available datasets")

    args = parser.parse_args(argv)

    config = {}
    if args.config:
        try:
            config = load_config(args.config)
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
            return 1

    logger = CLILogger(
        verbose=args.verbose or config.get("verbose", False),
        quiet=args.quiet or config.get("quiet", False),
    )

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "train":
        train_args = {
            "model_name": args.model,
            "dataset_name": args.dataset,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "weights": args.weights,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "num_workers": args.num_workers,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "force_download": args.force_download,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
            "multi_label": args.multi_label,
            "sampling_rate": args.sampling_rate,
            "binary_classification": args.binary_classification,
            "normalization": args.normalization,
        }
        for key, value in config.get("train", {}).items():
            if key in train_args and train_args[key] == parser.get_default(key):
                train_args[key] = value
        return _cli.train(logger=logger, **train_args)

    elif args.command == "evaluate":
        return _cli.evaluate(
            checkpoint=args.checkpoint,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
            force_download=args.force_download,
            split=args.split,
            logger=logger,
        )

    elif args.command == "predict":
        return _cli.predict(
            checkpoint=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            logger=logger,
        )

    elif args.command == "resume":
        return _cli.resume(
            checkpoint=args.checkpoint,
            epochs=args.epochs,
            output_dir=args.output_dir,
            accelerator=args.accelerator,
            devices=args.devices,
            early_stopping_patience=args.early_stopping_patience,
            logger=logger,
        )

    elif args.command == "info":
        return _cli.show_info(model_name=args.model, logger=logger)

    elif args.command == "list-models":
        _cli.list_models()
        return 0

    elif args.command == "list-datasets":
        _cli.list_datasets()
        return 0

    return 0
