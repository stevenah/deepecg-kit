"""Main CLI entry point with argument parsing and command routing."""

import argparse
import sys
from typing import Optional

import deepecgkit.cli as _cli
from deepecgkit.cli.config import load_config
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import DATASET_NAMES, MODEL_NAMES
from deepecgkit.evaluation.metrics import DEFAULT_CALIBRATION_BINS
from deepecgkit.evaluation.representation import (
    DEFAULT_PROBE_METRIC,
    KEEP_DIAGONAL,
    KEEP_EMBEDDINGS_MODES,
    PROBE_DATASETS,
    PROBE_INPUT_CHANNELS,
    PROBE_METRICS,
    PROBE_SAMPLING_RATE,
)
from deepecgkit.explainability import METHODS


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
  deepecg evaluate --checkpoint model.ckpt -m kanres -d af-classification
  deepecg predict --checkpoint model.ckpt -m kanres --input ecg.npy
  deepecg explain --checkpoint model.ckpt -m kanres --input ecg.npy -o explanations/
  deepecg probe -m kanres --source-checkpoint af-classification=af.pt -o probe/
  deepecg resume --checkpoint model.ckpt -m kanres --epochs 100
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
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
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
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed that produced the split during training (default: 42)",
    )
    eval_parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation fraction used during training (default: 0.2)",
    )
    eval_parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test fraction used during training (default: 0.1)",
    )
    eval_parser.add_argument(
        "--calibration-bins",
        type=int,
        default=DEFAULT_CALIBRATION_BINS,
        help=f"Bins for the calibration metrics (default: {DEFAULT_CALIBRATION_BINS})",
    )
    eval_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip the validation pass that fits the temperature",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference on ECG data")
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    predict_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
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

    extract_parser = subparsers.add_parser(
        "extract-features", help="Extract deep embeddings from ECG data"
    )
    extract_weights = extract_parser.add_mutually_exclusive_group(required=True)
    extract_weights.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a model checkpoint",
    )
    extract_weights.add_argument(
        "--weights",
        "-w",
        type=str,
        help="Pretrained weight name or path to a weights file",
    )
    extract_source = extract_parser.add_mutually_exclusive_group(required=True)
    extract_source.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=DATASET_NAMES,
        help="Dataset to extract features from",
    )
    extract_source.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to an input ECG file (.npy, .csv, or WFDB .dat/.hea)",
    )
    extract_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
    )
    extract_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save features (.npz, .npy, or .csv)",
    )
    extract_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset",
    )
    extract_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to extract from (default: test)",
    )
    extract_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    extract_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loading workers (default: 4)",
    )
    extract_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    extract_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision of the saved features (default: float32)",
    )
    extract_parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        help="Lead count, for architectures whose weights do not encode it",
    )
    extract_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset",
    )

    explain_parser = subparsers.add_parser(
        "explain", help="Attribute a model's prediction back to the ECG signal"
    )
    explain_weights = explain_parser.add_mutually_exclusive_group(required=True)
    explain_weights.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a model checkpoint",
    )
    explain_weights.add_argument(
        "--weights",
        "-w",
        type=str,
        help="Pretrained weight name or path to a weights file",
    )
    explain_source = explain_parser.add_mutually_exclusive_group(required=True)
    explain_source.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=DATASET_NAMES,
        help="Dataset to explain predictions on",
    )
    explain_source.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to an input ECG file (.npy, .csv, or WFDB .dat/.hea)",
    )
    explain_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
    )
    explain_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Directory to write the arrays, summary and plots into",
    )
    explain_parser.add_argument(
        "--method",
        type=str,
        default="saliency",
        choices=list(METHODS),
        help="Attribution method (default: saliency)",
    )
    explain_parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        metavar="CLASS",
        help="Class index to explain, or 'all' (default: the predicted class)",
    )
    explain_parser.add_argument(
        "--sampling-rate",
        type=float,
        default=500.0,
        help="Sampling rate in Hz, used to place the beat windows (default: 500)",
    )
    explain_parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum number of recordings to explain (default: 8)",
    )
    explain_parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
        help="Path steps for integrated gradients (default: 64)",
    )
    explain_parser.add_argument(
        "--occlusion-window",
        type=int,
        default=100,
        help="Occlusion window in samples (default: 100)",
    )
    explain_parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=50,
        help="Occlusion stride in samples (default: 50)",
    )
    explain_parser.add_argument(
        "--detect-lead",
        type=int,
        default=None,
        help="Lead index R-peaks are detected on (default: the second lead)",
    )
    explain_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the arrays and summary only, skipping the figures",
    )
    explain_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset",
    )
    explain_parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to explain (default: test)",
    )
    explain_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    explain_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loading workers (default: 4)",
    )
    explain_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    explain_parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        help="Lead count, for architectures whose weights do not encode it",
    )
    explain_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset",
    )

    probe_parser = subparsers.add_parser(
        "probe", help="Probe frozen encoders across datasets and build a transfer matrix"
    )
    probe_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture matching every checkpoint",
    )
    probe_parser.add_argument(
        "--source-checkpoint",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Encoder trained on DATASET, repeatable once per source",
    )
    probe_parser.add_argument(
        "--target",
        action="append",
        default=[],
        choices=DATASET_NAMES,
        help=f"Dataset to probe, repeatable (default: {', '.join(PROBE_DATASETS)})",
    )
    probe_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Directory for the matrix, plots and table",
    )
    probe_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the datasets",
    )
    probe_parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_PROBE_METRIC,
        choices=list(PROBE_METRICS),
        help=f"Metric filling the matrix (default: {DEFAULT_PROBE_METRIC})",
    )
    probe_parser.add_argument(
        "--input-channels",
        type=int,
        default=PROBE_INPUT_CHANNELS,
        help=f"Lead count the protocol runs at (default: {PROBE_INPUT_CHANNELS})",
    )
    probe_parser.add_argument(
        "--sampling-rate",
        type=int,
        default=PROBE_SAMPLING_RATE,
        help=f"Sampling rate for every dataset in Hz (default: {PROBE_SAMPLING_RATE})",
    )
    probe_parser.add_argument(
        "--no-random-init",
        action="store_true",
        help="Drop the untrained-encoder baseline row",
    )
    probe_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the heatmap and the embedding plots",
    )
    probe_parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip the LaTeX table",
    )
    probe_parser.add_argument(
        "--embedding-plots",
        type=str,
        default=KEEP_DIAGONAL,
        choices=list(KEEP_EMBEDDINGS_MODES),
        help=(
            f"Which cells get a t-SNE plot: none, all, or diagonal for the in-domain cells and "
            f"the random-init row (default: {KEEP_DIAGONAL})"
        ),
    )
    probe_parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity, clamped to what the sample count supports (default: 30)",
    )
    probe_parser.add_argument(
        "--probe-c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength for the probe (default: 1.0)",
    )
    probe_parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Solver iteration ceiling for the probe (default: 1000)",
    )
    probe_parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    probe_parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Held-out split the probe reports on (default: 0.1)",
    )
    probe_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the splits, the random encoder and the projection (default: 42)",
    )
    probe_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    probe_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loading workers (default: 4)",
    )
    probe_parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "mps"],
        help="Accelerator to use (default: auto)",
    )
    probe_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the datasets",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume training from a checkpoint")
    resume_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to resume from",
    )
    resume_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
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
            model_name=args.model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            devices=args.devices,
            force_download=args.force_download,
            split=args.split,
            seed=args.seed,
            val_split=args.val_split,
            test_split=args.test_split,
            calibration_bins=args.calibration_bins,
            calibrate=not args.no_calibration,
            logger=logger,
        )

    elif args.command == "predict":
        return _cli.predict(
            checkpoint=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            accelerator=args.accelerator,
            model_name=args.model,
            logger=logger,
        )

    elif args.command == "extract-features":
        return _cli.extract_features(
            output_path=args.output,
            model_name=args.model,
            checkpoint=args.checkpoint,
            weights=args.weights,
            dataset_name=args.dataset,
            input_path=args.input,
            data_dir=args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            dtype=args.dtype,
            input_channels=args.input_channels,
            force_download=args.force_download,
            logger=logger,
        )

    elif args.command == "explain":
        return _cli.explain(
            output_dir=args.output,
            model_name=args.model,
            checkpoint=args.checkpoint,
            weights=args.weights,
            dataset_name=args.dataset,
            input_path=args.input,
            data_dir=args.data_dir,
            split=args.split,
            method=args.method,
            target_class=args.target_class,
            sampling_rate=args.sampling_rate,
            max_samples=args.max_samples,
            ig_steps=args.ig_steps,
            occlusion_window=args.occlusion_window,
            occlusion_stride=args.occlusion_stride,
            detect_lead=args.detect_lead,
            plots=not args.no_plots,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            input_channels=args.input_channels,
            force_download=args.force_download,
            logger=logger,
        )

    elif args.command == "probe":
        return _cli.probe(
            output_dir=args.output,
            model_name=args.model,
            source_checkpoints=args.source_checkpoint,
            targets=args.target,
            data_dir=args.data_dir,
            metric=args.metric,
            input_channels=args.input_channels,
            sampling_rate=args.sampling_rate,
            random_init=not args.no_random_init,
            plots=not args.no_plots,
            latex=not args.no_latex,
            embedding_plots=args.embedding_plots,
            perplexity=args.perplexity,
            probe_c=args.probe_c,
            max_iter=args.max_iter,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            force_download=args.force_download,
            logger=logger,
        )

    elif args.command == "resume":
        return _cli.resume(
            checkpoint=args.checkpoint,
            model_name=args.model,
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
