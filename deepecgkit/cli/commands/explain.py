"""Explain command."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    MODEL_NAMES,
    _get_dataset_registry,
    _get_model_registry,
)
from deepecgkit.datasets import ECGDataModule
from deepecgkit.datasets.io import as_model_input, read_ecg_recording
from deepecgkit.explainability import METHODS, ECGExplainer
from deepecgkit.models.pretrained import model_name_for_weights

ARRAY_FILE = "attribution.npz"
SUMMARY_FILE = "summary.json"
SIGNAL_PLOT = "signal_attribution.png"
LEAD_PLOT = "lead_importance.png"
BEAT_PLOT = "beat_profile.png"

_LENGTH_MARKERS = ("Kernel size", "must match the size of tensor", "needs at least")
_LEAD_MARKER = "leads but received"


def explain(
    output_dir: str,
    model_name: str,
    checkpoint: Optional[str] = None,
    weights: Optional[str] = None,
    dataset_name: Optional[str] = None,
    input_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    split: str = "test",
    method: str = "saliency",
    target_class: Optional[str] = None,
    sampling_rate: float = 500.0,
    max_samples: int = 8,
    ig_steps: int = 64,
    occlusion_window: int = 100,
    occlusion_stride: int = 50,
    detect_lead: Optional[int] = None,
    plots: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    accelerator: str = "auto",
    input_channels: Optional[int] = None,
    force_download: bool = False,
    logger: Optional[CLILogger] = None,
) -> int:
    """Attribute a model's predictions back to the ECG and write them to disk."""
    logger = logger or CLILogger()

    if model_name not in _get_model_registry():
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODEL_NAMES)}")
        return 1

    if checkpoint is not None and not Path(checkpoint).exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    if input_path is not None and not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if dataset_name is not None and dataset_name not in _get_dataset_registry():
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
        return 1

    if weights is not None and not Path(weights).exists():
        target = model_name_for_weights(weights)
        if target is not None and target != model_name:
            logger.error(
                f"Pretrained weights '{weights}' belong to model '{target}', not '{model_name}'."
            )
            return 1

    if method not in METHODS:
        logger.error(f"Unknown method: {method}")
        logger.info(f"Available methods: {', '.join(METHODS)}")
        return 1

    if max_samples < 1:
        logger.error(f"--max-samples must be at least 1, got {max_samples}")
        return 1

    explain_all = target_class is not None and target_class.lower() == "all"
    fixed_target: Optional[int] = None
    if target_class is not None and not explain_all:
        try:
            fixed_target = int(target_class)
        except ValueError:
            logger.error(f"--target-class must be an integer or 'all', got '{target_class}'.")
            return 1

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

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
        )

        explainer_kwargs: Dict[str, Any] = {
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

        per_class: List[np.ndarray] = []
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
        if any(marker in message for marker in _LENGTH_MARKERS):
            logger.error(
                f"Attribution failed: {message}\n"
                f"The signal length is incompatible with '{model_name}'. Recordings above an "
                f"architecture's ceiling are centre-cropped for you, so this is the opposite "
                f"problem: the recording is too short. Check the model's minimum length, for "
                f"example AFModel's recording_length."
            )
        elif _LEAD_MARKER in message:
            logger.error(
                f"Attribution failed: {message}\n"
                f"Pass --input-channels to match the lead count the weights were built for."
            )
        else:
            logger.error(f"Attribution failed: {message}")
        return 1
    except Exception as error:
        logger.error(f"Attribution failed: {error}")
        return 1


def _load_data(
    dataset_name: Optional[str],
    input_path: Optional[str],
    data_dir: Optional[str],
    split: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    input_channels: Optional[int],
    force_download: bool,
    logger: CLILogger,
) -> Tuple[Any, Optional[int]]:
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

    dataset_kwargs: Dict[str, Any] = {"force_download": force_download}
    if data_dir:
        dataset_kwargs["data_dir"] = data_dir

    logger.info(f"Creating data module with {dataset_name} dataset...")
    data_module = ECGDataModule(
        dataset_class=_get_dataset_registry()[dataset_name],
        batch_size=batch_size,
        num_workers=num_workers,
        download=True,
        dataset_kwargs=dataset_kwargs,
    )
    data_module.setup()

    sample_signal, _ = data_module.dataset[0]
    if input_channels is None:
        input_channels = int(sample_signal.shape[0])

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


def _write_arrays(path: Path, result: Any, per_class: List[np.ndarray]) -> None:
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
    ranked = sorted(zip(names, shares), key=lambda pair: pair[1], reverse=True)[:3]
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
