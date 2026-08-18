"""Feature extraction command."""

from pathlib import Path
from typing import Any, Dict, Optional

from deepecgkit.cli.logger import CLILogger
from deepecgkit.cli.registry import (
    DATASET_NAMES,
    MODEL_NAMES,
    _get_dataset_registry,
    _get_model_registry,
)
from deepecgkit.datasets import ECGDataModule
from deepecgkit.features import FeatureExtractor
from deepecgkit.models.pretrained import model_name_for_weights


def extract_features(
    output_path: str,
    model_name: str,
    checkpoint: Optional[str] = None,
    weights: Optional[str] = None,
    dataset_name: Optional[str] = None,
    input_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
    dtype: str = "float32",
    input_channels: Optional[int] = None,
    force_download: bool = False,
    logger: Optional[CLILogger] = None,
) -> int:
    """Extract deep embeddings from ECG data and write them to disk."""
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

    try:
        data: Any = input_path
        if dataset_name is not None:
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
            data = data_module

            sample_signal, _ = data_module.dataset[0]
            if input_channels is None:
                input_channels = int(sample_signal.shape[0])

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
        message = str(error)
        if "Kernel size" in message or "must match the size of tensor" in message:
            logger.error(
                f"Feature extraction failed: {message}\n"
                f"The signal length is incompatible with '{model_name}'. Check the model's "
                f"length constraints, for example AFModel's recording_length, or the 5000 "
                f"sample ceiling on the attention architectures."
            )
        else:
            logger.error(f"Feature extraction failed: {message}")
        return 1
    except Exception as error:
        logger.error(f"Feature extraction failed: {error}")
        return 1
