"""Explain command."""

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from deepecgkit.cli import defaults
from deepecgkit.cli._common import (
    LEAD_MARKER,
    LENGTH_MARKERS,
    build_data_module,
    check_pretrained_weights,
    dataset_kwargs_for,
    infer_input_channels_from,
    prepare_output_dir,
    require_dataset,
    require_file,
    require_model,
)
from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import DATASET_NAMES, MODEL_NAMES
from deepecgkit.datasets.io import as_model_input, read_ecg_recording
from deepecgkit.explainability import METHODS, ECGExplainer

ARRAY_FILE = "attribution.npz"
SUMMARY_FILE = "summary.json"
SIGNAL_PLOT = "signal_attribution.png"
LEAD_PLOT = "lead_importance.png"
BEAT_PLOT = "beat_profile.png"


def explain(
    output_dir: str,
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
    method: str = "saliency",
    target_class: str | None = None,
    sampling_rate: float = 500.0,
    max_samples: int = 8,
    ig_steps: int = 64,
    occlusion_window: int = 100,
    occlusion_stride: int = 50,
    detect_lead: int | None = None,
    plots: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    accelerator: str = "auto",
    input_channels: int | None = None,
    force_download: bool = False,
    logger: CLILogger | None = None,
) -> int:
    """Attribute a model's predictions back to the ECG and write them to disk."""
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

    if method not in METHODS:
        logger.error(f"Unknown method: {method}")
        logger.info(f"Available methods: {', '.join(METHODS)}")
        return 1

    if max_samples < 1:
        logger.error(f"--max-samples must be at least 1, got {max_samples}")
        return 1

    explain_all = target_class is not None and target_class.lower() == "all"
    fixed_target: int | None = None
    if target_class is not None and not explain_all:
        try:
            fixed_target = int(target_class)
        except ValueError:
            logger.error(f"--target-class must be an integer or 'all', got '{target_class}'.")
            return 1

    try:
        output_path = prepare_output_dir(output_dir)

        data, input_channels = _load_data(
            dataset_name=dataset_name,
            input_path=input_path,
            data_dir=data_dir,
            split=split,
            max_samples=max_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            input_channels=input_channels,
            force_download=force_download,
            logger=logger,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )

        explainer_kwargs: dict[str, Any] = {
            "method": method,
            "batch_size": batch_size,
            "device": accelerator,
            "sampling_rate": sampling_rate,
            "max_samples": max_samples,
            "ig_steps": ig_steps,
            "occlusion_window": occlusion_window,
            "occlusion_stride": occlusion_stride,
            "detect_lead": detect_lead,
            "num_workers": num_workers,
            "progress": not logger.quiet,
        }

        if checkpoint is not None:
            logger.info(f"Loading model from {checkpoint}...")
            explainer = ECGExplainer.from_checkpoint(
                checkpoint, model_name, input_channels=input_channels, **explainer_kwargs
            )
        else:
            logger.info(f"Loading pretrained weights '{weights}'...")
            explainer = ECGExplainer.from_pretrained(
                weights, model_name=model_name, **explainer_kwargs
            )

        logger.info(f"Attributing with {method}...")
        result = explainer.explain(data, target=fixed_target)

        per_class: list[np.ndarray] = []
        if explain_all:
            classes = _output_size(explainer, result)
            logger.info(f"Attributing each of the {classes} classes in turn...")
            for index in range(classes):
                per_class.append(explainer.explain(data, target=index).values)

        _write_arrays(output_path / ARRAY_FILE, result, per_class)
        logger.info(f"Attribution arrays saved to {output_path / ARRAY_FILE}")

        summary = result.summary()
        summary["dataset"] = dataset_name
        summary["input"] = input_path
        summary["split"] = split if dataset_name is not None else None
        summary["target_class"] = "all" if explain_all else target_class or "predicted"
        (output_path / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
        logger.info(f"Summary saved to {output_path / SUMMARY_FILE}")

        if plots:
            _write_plots(output_path, result, logger)

        _report(result, model_name, method, output_path, logger)
        return 0
    except (RuntimeError, ValueError) as error:
        message = str(error)
        if any(marker in message for marker in LENGTH_MARKERS):
            logger.error(
                f"Attribution failed: {message}\n"
                f"The signal length is incompatible with '{model_name}'. Recordings above an "
                f"architecture's ceiling are centre-cropped for you, so this is the opposite "
                f"problem: the recording is too short. Check the model's minimum length, for "
                f"example AFModel's recording_length."
            )
        elif LEAD_MARKER in message:
            logger.error(
                f"Attribution failed: {message}\n"
                f"Pass --input-channels to match the lead count the weights were built for."
            )
        else:
            logger.error(f"Attribution failed: {message}")
        return 1
    except Exception as error:
        logger.error(f"Attribution failed: {error}")
        logger.debug(traceback.format_exc())
        return 1


def _load_data(
    dataset_name: str | None,
    input_path: str | None,
    data_dir: str | None,
    split: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    input_channels: int | None,
    force_download: bool,
    logger: CLILogger,
    val_split: float = defaults.VAL_SPLIT,
    test_split: float = defaults.TEST_SPLIT,
    seed: int = defaults.SEED,
) -> tuple[Any, int | None]:
    """Resolve the data source to something the explainer accepts.

    Args:
        dataset_name: Registered dataset name, or None when a file is used
        input_path: Path to an ECG recording, or None when a dataset is used
        data_dir: Directory containing the dataset
        split: Which split to explain
        max_samples: Cap on the number of recordings
        batch_size: Loader batch size
        num_workers: Loader workers
        input_channels: Lead count, inferred from the dataset when omitted
        force_download: Re-download the dataset
        logger: Where progress is reported

    Returns:
        Tuple of the data and the resolved lead count.

    Raises:
        RuntimeError: If the requested split holds no dataset.
    """
    if dataset_name is None:
        array = as_model_input(read_ecg_recording(input_path), input_channels)
        return array[:max_samples], input_channels

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

    if input_channels is None:
        input_channels = infer_input_channels_from(data_module)

    subset = data_module.dataset if split == "all" else getattr(data_module, f"{split}_dataset")
    if subset is None:
        raise RuntimeError(f"Data module has no '{split}' split available.")

    return Subset(subset, range(min(max_samples, len(subset)))), input_channels


def _output_size(explainer: ECGExplainer, result: Any) -> int:
    """Count the classes the model predicts.

    Args:
        explainer: The explainer holding the model
        result: A completed attribution, used for a representative input

    Returns:
        The width of the model's classifier head.
    """
    with torch.no_grad():
        sample = torch.as_tensor(result.signal[:1], dtype=torch.float32)
        outputs = explainer.model(sample.to(next(explainer.model.parameters()).device))
    outputs = outputs.logits if hasattr(outputs, "logits") else outputs
    return int(outputs.shape[-1])


def _write_arrays(path: Path, result: Any, per_class: list[np.ndarray]) -> None:
    """Write the attribution arrays, adding a class axis when every class was run.

    Args:
        path: Destination .npz
        result: The predicted-class attribution
        per_class: One attribution array per class, or empty
    """
    result.save(path)
    if not per_class:
        return

    existing = dict(np.load(str(path), allow_pickle=False))
    existing["values_by_class"] = np.stack(per_class, axis=0)
    np.savez(str(path), **existing)


def _write_plots(output_path: Path, result: Any, logger: CLILogger) -> None:
    """Render the three figures into the output directory.

    Args:
        output_path: Directory receiving the figures
        result: The attribution being drawn
        logger: Where progress is reported
    """
    from deepecgkit.explainability import (
        plot_attribution,
        plot_beat_profile,
        plot_lead_importance,
    )

    offset = result.crop_offset / result.sampling_rate
    plot_attribution(
        result.signal[0],
        result.values[0],
        sampling_rate=result.sampling_rate,
        leads=result.lead_names,
        time_offset=offset,
        title=f"{result.model_name or 'model'} attribution ({result.method})",
        save_path=str(output_path / SIGNAL_PLOT),
    )
    logger.info(f"Signal attribution saved to {output_path / SIGNAL_PLOT}")

    plot_lead_importance(
        result.lead_importance,
        leads=result.lead_names,
        title=f"{result.model_name or 'model'} lead importance",
        save_path=str(output_path / LEAD_PLOT),
    )
    logger.info(f"Lead importance saved to {output_path / LEAD_PLOT}")

    if result.beat_profile is None:
        logger.warning("No complete beats were found, so beat_profile.png was not written.")
        return

    plot_beat_profile(
        result.beat_profile,
        result.beat_times_ms,
        leads=result.lead_names,
        title=f"{result.model_name or 'model'} beat-aligned attribution",
        save_path=str(output_path / BEAT_PLOT),
    )
    logger.info(f"Beat profile saved to {output_path / BEAT_PLOT}")


def _report(
    result: Any, model_name: str, method: str, output_path: Path, logger: CLILogger
) -> None:
    """Print the headline numbers the attribution produced.

    Args:
        result: The attribution being reported
        model_name: Registry name of the model
        method: Attribution method used
        output_path: Directory the artifacts were written to
        logger: Where the report is written
    """
    logger.info("\nAttribution summary")
    logger.info("-" * 40)
    logger.info(f"  model: {model_name}")
    logger.info(f"  method: {method}")
    logger.info(f"  samples: {len(result)}")
    logger.info(f"  leads: {result.num_leads}")
    logger.info(f"  timesteps: {result.values.shape[2]}")
    if result.crop_offset:
        logger.info(f"  crop offset: {result.crop_offset} samples")

    shares = result.mean_lead_importance()
    names = result.lead_names or [f"Lead {i + 1}" for i in range(result.num_leads)]
    ranked = sorted(zip(names, shares, strict=False), key=lambda pair: pair[1], reverse=True)[:3]
    logger.info(f"  top leads: {', '.join(f'{n} {v:.1%}' for n, v in ranked)}")

    if result.pqrst_fractions is None:
        logger.info("  beats: none detected, so no cardiac breakdown was produced")
    else:
        logger.info(f"  beats: {result.beat_count}")
        shares_text = ", ".join(
            f"{k}={v:.1%}" for k, v in result.pqrst_fractions.items() if k != "baseline"
        )
        logger.info(f"  attribution by region: {shares_text}")
        enrich_text = ", ".join(
            f"{k}={v:.2f}" for k, v in result.pqrst_enrichment.items() if k != "baseline"
        )
        logger.info(f"  enrichment against chance: {enrich_text}")

    logger.info(f"  saved to: {output_path}")


def add_parser(subparsers: Any) -> Any:
    """Register the explain subcommand's arguments."""
    parser = subparsers.add_parser(
        "explain", help="Attribute a model's prediction back to the ECG signal"
    )
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
        help="Dataset to explain predictions on",
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
        help="Directory to write the arrays, summary and plots into",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="saliency",
        choices=list(METHODS),
        help="Attribution method (default: saliency)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=None,
        metavar="CLASS",
        help="Class index to explain, or 'all' (default: the predicted class)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=500.0,
        help="Sampling rate in Hz, used to place the beat windows (default: 500)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum number of recordings to explain (default: 8)",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=64,
        help="Path steps for integrated gradients (default: 64)",
    )
    parser.add_argument(
        "--occlusion-window",
        type=int,
        default=100,
        help="Occlusion window in samples (default: 100)",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=50,
        help="Occlusion stride in samples (default: 50)",
    )
    parser.add_argument(
        "--detect-lead",
        type=int,
        default=None,
        help="Lead index R-peaks are detected on (default: the second lead)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the arrays and summary only, skipping the figures",
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
        help="Dataset split to explain (default: test)",
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
        default=8,
        help="Batch size (default: 8)",
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
    parser.set_defaults(handler="explain", to_kwargs=_to_kwargs)
    return parser


def _to_kwargs(args: Any) -> dict[str, Any]:
    return {
        "output_dir": args.output,
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
        "method": args.method,
        "target_class": args.target_class,
        "sampling_rate": args.sampling_rate,
        "max_samples": args.max_samples,
        "ig_steps": args.ig_steps,
        "occlusion_window": args.occlusion_window,
        "occlusion_stride": args.occlusion_stride,
        "detect_lead": args.detect_lead,
        "plots": not args.no_plots,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "input_channels": args.input_channels,
        "force_download": args.force_download,
    }
