"""Feature extraction command."""

import traceback
from typing import Any

import numpy as np

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    build_data_module,
    check_pretrained_weights,
    checkpoint_dataset_kwargs,
    dataset_kwargs_for,
    infer_input_channels_from,
    read_checkpoint,
    report_data_error,
    report_shape_error,
    require_checkpoint,
    require_dataset,
    require_file,
    require_model,
    resolve_model_name,
)
from deepecgkit.cli.logger import CLILogger
from deepecgkit.datasets.io import load_recordings, prepare_recording
from deepecgkit.features import FeatureExtractor, FeatureSet
from deepecgkit.models.pretrained import model_name_for_weights
from deepecgkit.registry import get_dataset_names, get_model_names

UNPREPROCESSED_WARNING = (
    "This checkpoint does not record how its training data was preprocessed, so the "
    "recording is passed to the model as it is. It has to match the training data "
    "already: same sampling rate, same segment length, same normalisation."
)


def prepare_input_file(
    input_path: str,
    input_channels: int | None,
    preprocessing: dict[str, Any] | None,
    sampling_rate: float | None,
    logger: CLILogger,
    max_recordings: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Read a recording file and put it through a model's training preprocessing.

    Args:
        input_path: Path to a .npy, .csv, or WFDB recording
        input_channels: Lead count the model consumes
        preprocessing: The checkpoint's recorded preprocessing, or None to
            pass the recordings through unchanged
        sampling_rate: Sampling rate of the file, when the caller knows it
        logger: Where the applied steps are reported
        max_recordings: Read at most this many recordings from the file

    Returns:
        Tuple of the segments stacked as (num_segments, leads, samples), the
        index of the recording each segment came from, and the sampling rate
        of the segments (None when unknown).

    Raises:
        ValueError: If the recordings end up with different lengths and so
            cannot be stacked.
    """
    recordings, file_rate, load_notes = load_recordings(input_path, input_channels)
    if max_recordings is not None:
        recordings = recordings[:max_recordings]
    source_rate = sampling_rate if sampling_rate is not None else file_rate

    segments: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    rate: float | None = source_rate
    for index, recording in enumerate(recordings):
        prepared = prepare_recording(recording, preprocessing, source_rate)
        if index == 0 and (load_notes or prepared.notes):
            logger.info(f"Preprocessing: {'; '.join([*load_notes, *prepared.notes])}.")
        rate = prepared.sampling_rate
        segments.append(prepared.segments)
        owners.append(np.full(len(prepared.segments), index))

    lengths = {segment.shape[-1] for segment in segments}
    if len(lengths) > 1:
        raise ValueError(
            f"The recordings in {input_path} have different lengths after preprocessing "
            f"({sorted(lengths)} samples), so they cannot be processed as one batch."
        )
    return np.concatenate(segments, axis=0), np.concatenate(owners, axis=0), rate


def extract_features(
    output_path: str,
    model_name: str | None = None,
    checkpoint: str | None = None,
    weights: str | None = None,
    dataset_name: str | None = None,
    input_path: str | None = None,
    data_dir: str | None = None,
    split: str = "test",
    val_split: float = defaults.VAL_SPLIT,
    test_split: float = defaults.TEST_SPLIT,
    seed: int = defaults.SEED,
    batch_size: int = defaults.BATCH_SIZE,
    num_workers: int = defaults.NUM_WORKERS,
    accelerator: str = "auto",
    dtype: str = "float32",
    input_channels: int | None = None,
    force_download: bool = False,
    sampling_rate: float | None = None,
    preprocess: bool = True,
    logger: CLILogger | None = None,
) -> int:
    """Extract deep embeddings from ECG data and write them to disk.

    Data is prepared the way the checkpoint was trained: a dataset is rebuilt
    with the recorded sampling rate, segment length and normalisation, and a
    recording file is resampled, segmented and normalised to match. A file
    yields one embedding per recording, averaged over its segments.
    """
    logger = logger or CLILogger()

    if checkpoint is None and weights is None:
        logger.error("Pass --checkpoint or --weights to say which model to load.")
        return 1
    if checkpoint is not None:
        checkpoint = require_checkpoint(checkpoint, logger)
        if checkpoint is None:
            return 1
    for code in (
        require_model(model_name, logger) if model_name is not None else None,
        require_file(input_path, "Input file", logger),
        require_dataset(dataset_name, logger),
        check_pretrained_weights(weights, model_name, logger),
    ):
        if code is not None:
            return code

    stored: dict[str, Any] = {}
    if checkpoint is not None:
        try:
            stored = read_checkpoint(checkpoint)
        except Exception as error:
            logger.error(f"Feature extraction failed: could not read {checkpoint}: {error}")
            return 1
        model_name = resolve_model_name(stored, model_name, logger)
    elif model_name is None:
        model_name = model_name_for_weights(str(weights))
        if model_name is None:
            logger.error(f"Could not tell which model '{weights}' belongs to; pass --model.")
    if model_name is None:
        return 1

    data_config = stored.get("data_config") or {}
    preprocessing = data_config.get("preprocessing") if preprocess else None
    if input_channels is None:
        input_channels = (stored.get("model_config") or {}).get("input_channels")

    data: Any = None
    owners: np.ndarray | None = None
    if dataset_name is not None:
        dataset_rate = int(sampling_rate) if sampling_rate is not None else None
        if checkpoint is not None:
            dataset_kwargs, _ = checkpoint_dataset_kwargs(
                dataset_name,
                stored,
                logger,
                force_download=force_download,
                data_dir=data_dir,
                sampling_rate=dataset_rate,
            )
        else:
            dataset_kwargs = dataset_kwargs_for(
                dataset_name,
                force_download=force_download,
                data_dir=data_dir,
                sampling_rate=dataset_rate,
            )
        try:
            data_module = build_data_module(
                dataset_name,
                batch_size=batch_size,
                num_workers=num_workers,
                val_split=val_split,
                test_split=test_split,
                seed=seed,
                dataset_kwargs=dataset_kwargs,
                logger=logger,
            )
        except Exception as error:
            return report_data_error(error, dataset_name, logger, data_dir)
        data = data_module

        if input_channels is None:
            input_channels = infer_input_channels_from(data_module)

    try:
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
                str(weights),
                model_name=model_name,
                batch_size=batch_size,
                device=accelerator,
                dtype=dtype,
                progress=not logger.quiet,
            )

        if input_path is not None:
            if checkpoint is not None and preprocess and not preprocessing:
                logger.warning(UNPREPROCESSED_WARNING)
            data, owners, _ = prepare_input_file(
                input_path,
                extractor.input_channels if input_channels is None else input_channels,
                preprocessing,
                sampling_rate,
                logger,
            )

        logger.info("Extracting features...")
        result = extractor.extract(data, split=split)
        if owners is not None and len(owners) and owners.max() + 1 < len(result):
            # Several segments per recording: one embedding per recording is
            # what a caller indexing by recording expects.
            pooled = np.stack(
                [result.features[owners == index].mean(axis=0) for index in range(owners.max() + 1)]
            ).astype(result.features.dtype)
            result = FeatureSet(pooled, None, result.model_name)
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
        help="Checkpoint file, or a run directory whose best checkpoint is used",
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
        choices=get_dataset_names(),
        metavar="DATASET",
        help="Dataset to extract features from (see: deepecg list-datasets)",
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
        default=None,
        choices=get_model_names(),
        metavar="MODEL",
        help="Model architecture (default: the one recorded in the checkpoint)",
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
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        metavar="HZ",
        help="With --input: sampling rate of the recording, so it can be resampled to the "
        "model's training rate. With --dataset: rate to resample the dataset to (default: "
        "the rate recorded in the checkpoint)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="With --input: pass the recording to the model unchanged",
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
        "sampling_rate": args.sampling_rate,
        "preprocess": not args.no_preprocess,
    }
