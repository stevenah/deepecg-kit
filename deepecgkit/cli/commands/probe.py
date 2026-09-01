"""Representation-quality probing command."""

import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from deepecgkit.analysis.latex import transfer_matrix_table
from deepecgkit.cli import defaults
from deepecgkit.cli._common import require_model
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import DATASET_NAMES, MODEL_NAMES
from deepecgkit.evaluation.representation import (
    DEFAULT_PROBE_METRIC,
    KEEP_DIAGONAL,
    KEEP_EMBEDDINGS_MODES,
    KEEP_NONE,
    PROBE_DATASETS,
    PROBE_INPUT_CHANNELS,
    PROBE_METRICS,
    PROBE_SAMPLING_RATE,
    RANDOM_INIT,
    TRANSFER_KEY_SEPARATOR,
    TransferMatrix,
    cross_dataset_transfer,
    embed_2d,
)
from deepecgkit.evaluation.visualization import plot_embedding, plot_transfer_matrix

MATRIX_JSON = "transfer_matrix.json"
MATRIX_CSV = "transfer_matrix.csv"
MATRIX_PLOT = "transfer_matrix.png"
MATRIX_TABLE = "transfer_matrix.tex"
EMBEDDING_PLOT_TEMPLATE = "embedding_{source}_{target}.png"

SOURCE_SEPARATOR = "="

_LENGTH_MARKERS = ("Kernel size", "must match the size of tensor", "needs at least")
_LEAD_MARKER = "leads but received"


def parse_source_checkpoints(entries: Sequence[str]) -> dict[str, str]:
    """Parse repeated NAME=PATH source arguments.

    Args:
        entries: Raw --source-checkpoint values

    Returns:
        Dataset name mapped to checkpoint path.

    Raises:
        ValueError: If an entry lacks a separator, names a dataset twice, or
            leaves either half empty.
    """
    parsed: dict[str, str] = {}
    for entry in entries:
        if SOURCE_SEPARATOR not in entry:
            raise ValueError(
                f"Malformed source '{entry}'. Write it as DATASET{SOURCE_SEPARATOR}PATH, for "
                f"example af-classification{SOURCE_SEPARATOR}runs/af/best.pt."
            )
        name, _, path = entry.partition(SOURCE_SEPARATOR)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(
                f"Malformed source '{entry}'. Both the dataset and the checkpoint path are "
                f"required."
            )
        if name in parsed:
            raise ValueError(f"Source dataset '{name}' was given more than once.")
        parsed[name] = path
    return parsed


def _validate(
    model_name: str,
    sources: dict[str, str],
    targets: Sequence[str],
    metric: str,
    embedding_plots: str,
    logger: CLILogger,
) -> int | None:
    """Reject an unrunnable request before any data is loaded.

    Args:
        model_name: Registered model name
        sources: Source dataset mapped to checkpoint path
        targets: Probe targets
        metric: Headline metric
        embedding_plots: Which cells to plot embeddings for
        logger: Where to report the failure

    Returns:
        An exit code when the request cannot be run, or None when it can.
    """
    if (code := require_model(model_name, logger)) is not None:
        return code

    for name, path in sources.items():
        if name not in DATASET_NAMES:
            logger.error(f"Unknown source dataset: {name}")
            logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
            return 1
        if not Path(path).exists():
            logger.error(f"Checkpoint not found for source '{name}': {path}")
            return 1

    for name in targets:
        if name not in DATASET_NAMES:
            logger.error(f"Unknown target dataset: {name}")
            logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
            return 1

    if metric not in PROBE_METRICS:
        logger.error(f"Unknown metric: {metric}")
        logger.info(f"Available metrics: {', '.join(PROBE_METRICS)}")
        return 1

    if embedding_plots not in KEEP_EMBEDDINGS_MODES:
        logger.error(f"Unknown embedding plot mode: {embedding_plots}")
        logger.info(f"Available modes: {', '.join(KEEP_EMBEDDINGS_MODES)}")
        return 1

    return None


def _write_embedding_plots(
    matrix: TransferMatrix,
    output_path: Path,
    perplexity: float,
    seed: int,
    logger: CLILogger,
) -> list[str]:
    """Project and plot the held-out embeddings the matrix retained.

    Args:
        matrix: A matrix run with keep_embeddings set
        output_path: Directory to write the plots into
        perplexity: t-SNE perplexity
        seed: Seed for the projection
        logger: Where to report each written plot

    Returns:
        The written filenames.
    """
    written: list[str] = []
    for key, features in sorted(matrix.embeddings.items()):
        source, _, target = key.partition(TRANSFER_KEY_SEPARATOR)
        try:
            projection = embed_2d(features.features, perplexity=perplexity, random_state=seed)
        except ValueError as error:
            logger.warning(f"Skipping embedding plot for {key}: {error}")
            continue
        name = EMBEDDING_PLOT_TEMPLATE.format(source=source, target=target)
        plot_embedding(
            projection,
            features.labels,
            class_names=matrix.class_names.get(target),
            title=f"{matrix.model_name}: {source} encoder on {target}",
            save_path=str(output_path / name),
        )
        written.append(name)
        logger.info(f"Embedding plot saved to {output_path / name}")
    return written


def _report(matrix: TransferMatrix, output_path: Path, logger: CLILogger) -> None:
    """Print the matrix and a short provenance block.

    Args:
        matrix: The completed matrix
        output_path: Directory the artifacts were written to
        logger: Where to print
    """
    frame = matrix.to_frame()
    logger.info("\nCross-dataset transfer")
    logger.info("-" * 40)
    logger.info(frame.round(4).to_string())
    logger.info("-" * 40)
    logger.info(f"  model: {matrix.model_name}")
    logger.info(f"  metric: {matrix.metric}")
    logger.info(f"  sources: {len(matrix.sources)}")
    logger.info(f"  targets: {len(matrix.targets)}")
    logger.info(f"  input channels: {matrix.protocol.get('input_channels')}")
    logger.info(f"  sampling rate: {matrix.protocol.get('sampling_rate')} Hz")
    logger.info(f"  saved to: {output_path}")

    unscored = [key for key, result in matrix.results.items() if not np.isfinite(result.score)]
    if unscored:
        logger.warning(
            f"{len(unscored)} cell(s) could not be scored: {', '.join(sorted(unscored))}"
        )


def probe(
    output_dir: str,
    model_name: str,
    source_checkpoints: Sequence[str],
    targets: Sequence[str] | None = None,
    data_dir: str | None = None,
    metric: str = DEFAULT_PROBE_METRIC,
    input_channels: int = PROBE_INPUT_CHANNELS,
    sampling_rate: int = PROBE_SAMPLING_RATE,
    random_init: bool = True,
    plots: bool = True,
    latex: bool = True,
    embedding_plots: str = KEEP_DIAGONAL,
    perplexity: float = 30.0,
    probe_c: float = 1.0,
    max_iter: int = 1000,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
    force_download: bool = False,
    logger: CLILogger | None = None,
) -> int:
    """Probe frozen encoders across datasets and write the transfer matrix.

    Args:
        output_dir: Directory to write the matrix, plots and table into
        model_name: Registered model name matching every checkpoint
        source_checkpoints: Repeated DATASET=PATH entries naming the encoders
        targets: Datasets to probe, defaulting to the four protocol corpora
        data_dir: Directory holding the datasets
        metric: Headline metric filling the matrix
        input_channels: Lead count the protocol runs at
        sampling_rate: Target sampling rate for every dataset, in Hz
        random_init: Include the untrained-encoder floor
        plots: Write the heatmap
        latex: Write the booktabs table
        embedding_plots: Which cells get an embedding plot
        perplexity: t-SNE perplexity
        probe_c: Inverse regularisation strength for the probe
        max_iter: Solver iteration ceiling for the probe
        val_split: Fraction held out for validation
        test_split: Fraction the probe reports on
        seed: Seed for the splits, the random-init encoder and the projection
        batch_size: Batch size for embedding extraction
        num_workers: Data loading workers
        accelerator: Accelerator to use
        force_download: Force re-download of the datasets
        logger: Where to report progress

    Returns:
        Zero on success, one on any handled failure.
    """
    logger = logger or CLILogger()

    try:
        sources = parse_source_checkpoints(source_checkpoints)
    except ValueError as error:
        logger.error(str(error))
        return 1

    resolved_targets = list(targets) if targets else list(PROBE_DATASETS)
    failure = _validate(model_name, sources, resolved_targets, metric, embedding_plots, logger)
    if failure is not None:
        return failure

    if not sources and not random_init:
        logger.error(
            "Nothing to probe: pass at least one --source-checkpoint, or drop --no-random-init."
        )
        return 1

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            f"Probing {len(sources) + int(random_init)} encoder(s) on "
            f"{len(resolved_targets)} target(s)..."
        )
        matrix = cross_dataset_transfer(
            dict(sources),
            model_name,
            targets=resolved_targets,
            input_channels=input_channels,
            metric=metric,
            include_random_init=random_init,
            sampling_rate=sampling_rate,
            data_dir=data_dir,
            force_download=force_download,
            batch_size=batch_size,
            num_workers=num_workers,
            device=accelerator,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            progress=not logger.quiet,
            verbose=logger.verbose,
            keep_embeddings=embedding_plots if plots else KEEP_NONE,
            c=probe_c,
            max_iter=max_iter,
            random_state=seed,
        )

        matrix.save(output_path / MATRIX_JSON)
        logger.info(f"Results saved to {output_path / MATRIX_JSON}")
        matrix.save(output_path / MATRIX_CSV)
        logger.info(f"Matrix saved to {output_path / MATRIX_CSV}")

        if latex:
            table = transfer_matrix_table(
                matrix.to_frame(),
                metric=matrix.metric,
                baseline_row=RANDOM_INIT if random_init else None,
            )
            (output_path / MATRIX_TABLE).write_text(table)
            logger.info(f"Table saved to {output_path / MATRIX_TABLE}")

        if plots:
            plot_transfer_matrix(
                matrix.values,
                matrix.sources,
                matrix.targets,
                metric=matrix.metric,
                baseline_rows=1 if random_init else 0,
                title=f"Cross-dataset transfer: {model_name}",
                save_path=str(output_path / MATRIX_PLOT),
            )
            logger.info(f"Heatmap saved to {output_path / MATRIX_PLOT}")
            _write_embedding_plots(matrix, output_path, perplexity, seed, logger)
            if embedding_plots == KEEP_DIAGONAL:
                logger.info(
                    "Embedding plots cover the in-domain cells and the random-init row; pass "
                    "--embedding-plots all for every cell."
                )

        _report(matrix, output_path, logger)
        return 0
    except (RuntimeError, ValueError) as error:
        message = str(error)
        if any(marker in message for marker in _LENGTH_MARKERS):
            logger.error(
                f"Probing failed: {message}\n"
                f"The protocol runs every dataset at {sampling_rate} Hz over 10 seconds, so "
                f"'{model_name}' must accept that length. AFModel needs at least 7971 samples and "
                f"cannot be probed under this protocol; the attention architectures cap at 5000."
            )
        elif _LEAD_MARKER in message:
            logger.error(
                f"Probing failed: {message}\n"
                f"The protocol is single-lead. Pass --input-channels matching the checkpoints, or "
                f"retrain the encoders on single-lead data."
            )
        else:
            logger.error(f"Probing failed: {message}")
        return 1
    except Exception as error:
        logger.error(f"Probing failed: {error}")
        logger.debug(traceback.format_exc())
        return 1


def add_parser(subparsers: Any) -> Any:
    """Register the probe subcommand's arguments."""
    parser = subparsers.add_parser(
        "probe", help="Probe frozen encoders across datasets and build a transfer matrix"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=MODEL_NAMES,
        help="Model architecture matching every checkpoint",
    )
    parser.add_argument(
        "--source-checkpoint",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Encoder trained on DATASET, repeatable once per source",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        choices=DATASET_NAMES,
        help=f"Dataset to probe, repeatable (default: {', '.join(PROBE_DATASETS)})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Directory for the matrix, plots and table",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the datasets",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_PROBE_METRIC,
        choices=list(PROBE_METRICS),
        help=f"Metric filling the matrix (default: {DEFAULT_PROBE_METRIC})",
    )
    parser.add_argument(
        "--input-channels",
        type=int,
        default=PROBE_INPUT_CHANNELS,
        help=f"Lead count the protocol runs at (default: {PROBE_INPUT_CHANNELS})",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=PROBE_SAMPLING_RATE,
        help=f"Sampling rate for every dataset in Hz (default: {PROBE_SAMPLING_RATE})",
    )
    parser.add_argument(
        "--no-random-init",
        action="store_true",
        help="Drop the untrained-encoder baseline row",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the heatmap and the embedding plots",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip the LaTeX table",
    )
    parser.add_argument(
        "--embedding-plots",
        type=str,
        default=KEEP_DIAGONAL,
        choices=list(KEEP_EMBEDDINGS_MODES),
        help=(
            f"Which cells get a t-SNE plot: none, all, or diagonal for the in-domain cells and "
            f"the random-init row (default: {KEEP_DIAGONAL})"
        ),
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity, clamped to what the sample count supports (default: 30)",
    )
    parser.add_argument(
        "--probe-c",
        type=float,
        default=1.0,
        help="Inverse regularisation strength for the probe (default: 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Solver iteration ceiling for the probe (default: 1000)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=defaults.VAL_SPLIT,
        help=f"Validation split fraction (default: {defaults.VAL_SPLIT})",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=defaults.TEST_SPLIT,
        help=f"Held-out split the probe reports on (default: {defaults.TEST_SPLIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.SEED,
        help=(
            f"Seed for the splits, the random encoder and the projection (default: {defaults.SEED})"
        ),
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
        "--force-download",
        action="store_true",
        help="Force re-download the datasets",
    )
    parser.set_defaults(handler="probe", to_kwargs=_to_kwargs)
    return parser


def _to_kwargs(args: Any) -> dict[str, Any]:
    return {
        "output_dir": args.output,
        "model_name": args.model,
        "source_checkpoints": args.source_checkpoint,
        "targets": args.target,
        "data_dir": args.data_dir,
        "metric": args.metric,
        "input_channels": args.input_channels,
        "sampling_rate": args.sampling_rate,
        "random_init": not args.no_random_init,
        "plots": not args.no_plots,
        "latex": not args.no_latex,
        "embedding_plots": args.embedding_plots,
        "perplexity": args.perplexity,
        "probe_c": args.probe_c,
        "max_iter": args.max_iter,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "force_download": args.force_download,
    }
