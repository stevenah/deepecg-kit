"""Evaluate command."""

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
from deepecgkit.training import ECGTrainer


def evaluate(
    checkpoint: str,
    dataset_name: str,
    model_name: str,
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: str = "auto",
    devices: int = 1,
    force_download: bool = False,
    split: str = "test",
    logger: Optional[CLILogger] = None,
) -> int:
    """Evaluate a trained model on a dataset."""
    logger = logger or CLILogger()

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        return 1

    dataset_registry = _get_dataset_registry()
    model_registry = _get_model_registry()

    if dataset_name not in dataset_registry:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASET_NAMES)}")
        return 1

    if model_name not in model_registry:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODEL_NAMES)}")
        return 1

    try:
        logger.info(f"Loading model from {checkpoint}...")
        model_class = model_registry[model_name]
        model = model_class(input_channels=1, output_size=4)
        trainer = ECGTrainer.load_checkpoint(checkpoint, model=model, device=accelerator)

        dataset_class = dataset_registry[dataset_name]
        dataset_kwargs: Dict[str, Any] = {"force_download": force_download}
        if data_dir:
            dataset_kwargs["data_dir"] = data_dir

        logger.info(f"Creating data module with {dataset_name} dataset...")
        data_module = ECGDataModule(
            dataset_class=dataset_class,
            batch_size=batch_size,
            num_workers=num_workers,
            download=True,
            dataset_kwargs=dataset_kwargs,
        )

        logger.info(f"Running evaluation on {split} split...")
        if split == "test":
            results = trainer.test(data_module)
        elif split == "val":
            results = trainer.validate(data_module)
        else:
            logger.error(f"Invalid split: {split}. Use 'test' or 'val'.")
            return 1

        if results:
            logger.info("\nEvaluation Results:")
            logger.info("-" * 40)
            for key, value in results.items():
                logger.info(f"  {key}: {value:.4f}")

        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
