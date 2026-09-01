"""Feature extraction command."""

import traceback
from typing import Any

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    build_data_module,
    check_pretrained_weights,
    dataset_kwargs_for,
    infer_input_channels_from,
    report_shape_error,
    require_dataset,
    require_file,
    require_model,
)
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import DATASET_NAMES, MODEL_NAMES
from deepecgkit.features import FeatureExtractor


def extract_features(
    output_path: str,
    model_name: str,
    checkpoint: str | None = None,
    weights: str | None = None,
    dataset_name: str | None = None,
    input_path: str | None = None,
    data_dir: str | None = None,
    split: str = "test",
    val_split: float = defaults.VAL_SPLIT,
    test_split: float = defaults.TEST_SPLIT,
    seed: int = defaults.SEED,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
    dtype: str = "float32",
    input_channels: int | None = None,
    force_download: bool = False,
    logger: CLILogger | None = None,
) -> int:
    """Extract deep embeddings from ECG data and write them to disk."""
    logger = logger or CLILogger()

    for code in (
        require_model(model_name, logger),
        require_file(checkpoint, "Checkpoint", logger),
        require_file(input_path, "Input file", logger),
        require_dataset(dataset_name, logger),
        check_pretrained_weights(weights, model_name, logger),
    ):
        if code is not None:
            return code

    try:
        data: Any = input_path
        if dataset_name is not None:
            data_module = build_data_module(
                dataset_name,
                batch_size=batch_size,
                num_workers=num_workers,
                val_split=val_split,
                test_split=test_split,
                seed=seed,
                dataset_kwargs=dataset_kwargs_for(
                    dataset_name, force_download=force_download, data_dir=data_dir
                ),
                logger=logger,
            )
            data = data_module

            if input_channels is None:
                input_channels = infer_input_channels_from(data_module)

        if checkpoint is not None:
            logger.info(f"Loading model from {checkpoint}...")
            extractor = FeatureExtractor.from_checkpoint(
                checkpoint,
                model_name,
                input_channels=input_channels,
                batch_size=batch_size,
                device=accelerator,
                dtype=dtype,
                progress=not logger.quiet,
            )
        else:
            logger.info(f"Loading pretrained weights '{weights}'...")
            extractor = FeatureExtractor.from_pretrained(
                weights,
                model_name=model_name,
                batch_size=batch_size,
                device=accelerator,
                dtype=dtype,
                progress=not logger.quiet,
            )

        logger.info("Extracting features...")
        result = extractor.extract(data, split=split)
        result.save(output_path)

        logger.info("\nExtracted features")
        logger.info("-" * 40)
        logger.info(f"  model: {model_name}")
        logger.info(f"  samples: {len(result)}")
        logger.info(f"  feature dim: {result.feature_dim}")
        labels_note = f"yes (shape {result.labels.shape})" if result.labels is not None else "no"
        logger.info(f"  labels: {labels_note}")
        logger.info(f"  dtype: {result.features.dtype}")
        logger.info(f"  saved to: {output_path}")

        return 0
    except RuntimeError as error:
        return report_shape_error(error, model_name, "Feature extraction", logger)
    except Exception as error:
        logger.error(f"Feature extraction failed: {error}")
        logger.debug(traceback.format_exc())
        return 1


def add_parser(subparsers: Any) -> Any:
    """Register the extract-features subcommand's arguments."""
    parser = subparsers.add_parser("extract-features", help="Extract deep embeddings from ECG data")
    weights_group = parser.add_mutually_exclusive_group(required=True)
    weights_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a model checkpoint",
    )
    weights_group.add_argument(
        "--weights",
        "-w",
        type=str,
        help="Pretrained weight name or path to a weights file",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=DATASET_NAMES,
        help="Dataset to extract features from",
    )
    source_group.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to an input ECG file (.npy, .csv, or WFDB .dat/.hea)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture to load",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save features (.npz, .npy, or .csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to extract from (default: test)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=defaults.VAL_SPLIT,
        help=f"Validation fraction used during training (default: {defaults.VAL_SPLIT})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=defaults.TEST_SPLIT,
        help=f"Test fraction used during training (default: {defaults.TEST_SPLIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.SEED,
        help=f"Seed that produced the split during training (default: {defaults.SEED})",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=defaults.BATCH_SIZE,
        help=f"Batch size (default: {defaults.BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.NUM_WORKERS,
        help=f"Data loading workers (default: {defaults.NUM_WORKERS})",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default=defaults.ACCELERATOR,
        choices=defaults.ACCELERATOR_CHOICES,
        help="Accelerator to use (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision of the saved features (default: float32)",
    )
    parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        help="Lead count, for architectures whose weights do not encode it",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download the dataset",
    )
    parser.set_defaults(handler="extract_features", to_kwargs=_to_kwargs)
    return parser


def _to_kwargs(args: Any) -> dict[str, Any]:
    return {
        "output_path": args.output,
        "model_name": args.model,
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "dataset_name": args.dataset,
        "input_path": args.input,
        "data_dir": args.data_dir,
        "split": args.split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "dtype": args.dtype,
        "input_channels": args.input_channels,
        "force_download": args.force_download,
    }
