"""The run-directory contract: what a training run records, and how it is read.

A run directory carries two JSON files next to its metrics:

- ``run_metadata.json``: what produced the run (model, dataset, seed, split
  fractions, labelling mode, hyperparameters). The analysis pipeline uses the
  recorded seed to group runs; evaluate uses the split protocol to rebuild the
  exact training partition.
- ``split_info.json``: the realised split sizes and test groups, used to
  verify a rebuilt partition matches the one trained on.

Filenames are shared with deepecgkit.analysis.aggregation, the consumer.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepecgkit.analysis.aggregation import METADATA_FILENAME, SPLIT_INFO_FILENAME
from deepecgkit.cli.logger import CLILogger

RUN_METADATA_SCHEMA_VERSION = 1

__all__ = [
    "RunContext",
    "find_run_dir",
    "load_run_context",
    "verify_split",
    "write_run_metadata",
]


def write_run_metadata(output_dir: str | Path, payload: dict[str, Any]) -> None:
    """Merge payload into the run directory's run_metadata.json.

    Merging rather than overwriting lets the training command and the batch
    runner both record their keys in the same file.

    Args:
        output_dir: The run directory
        payload: Keys to record or update
    """
    path = Path(output_dir) / METADATA_FILENAME
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
        except (ValueError, OSError):
            pass
    existing.update(payload)
    existing.setdefault("schema_version", RUN_METADATA_SCHEMA_VERSION)
    path.write_text(json.dumps(existing, indent=2) + "\n")


def find_run_dir(checkpoint: str | Path) -> Path | None:
    """Walk up from a checkpoint file to the run directory that produced it.

    Checkpoints live in ``<run>/checkpoints/epoch=...pt``, so the metadata is
    at most two levels up.

    Args:
        checkpoint: Path to the checkpoint file

    Returns:
        The run directory, or None when no metadata is found nearby.
    """
    path = Path(checkpoint).resolve()
    for candidate in list(path.parents)[:2]:
        if (candidate / METADATA_FILENAME).exists() or (candidate / SPLIT_INFO_FILENAME).exists():
            return candidate
    return None


@dataclass(frozen=True)
class RunContext:
    """The split protocol recovered for a checkpoint.

    Attributes:
        run_dir: The run directory the checkpoint belongs to, when found
        seed: Recorded split seed
        val_split: Recorded validation fraction
        test_split: Recorded test fraction
        source: Where the values came from: "run_metadata", "split_info",
            or "none"
    """

    run_dir: Path | None
    seed: int | None = None
    val_split: float | None = None
    test_split: float | None = None
    source: str = "none"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_run_context(checkpoint: str | Path) -> RunContext:
    """Recover the training split protocol for a checkpoint.

    run_metadata.json wins; split_info.json supplies the seed as a fallback
    for runs recorded before full metadata existed.

    Args:
        checkpoint: Path to the checkpoint file

    Returns:
        The recovered context; ``source == "none"`` when nothing was found.
    """
    run_dir = find_run_dir(checkpoint)
    if run_dir is None:
        return RunContext(run_dir=None)

    metadata = _read_json(run_dir / METADATA_FILENAME)
    if metadata:
        seed = metadata.get("seed")
        return RunContext(
            run_dir=run_dir,
            seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
            val_split=metadata.get("val_split"),
            test_split=metadata.get("test_split"),
            source="run_metadata",
        )

    split_info = _read_json(run_dir / SPLIT_INFO_FILENAME)
    seed = split_info.get("seed")
    if isinstance(seed, int) and not isinstance(seed, bool):
        return RunContext(run_dir=run_dir, seed=seed, source="split_info")
    return RunContext(run_dir=run_dir)


def verify_split(data_module: Any, run_dir: str | Path, logger: CLILogger) -> None:
    """Warn when a rebuilt partition disagrees with the recorded one.

    Args:
        data_module: A data module whose setup has already run
        run_dir: The run directory holding split_info.json
        logger: CLI logger
    """
    recorded = _read_json(Path(run_dir) / SPLIT_INFO_FILENAME)
    if not recorded:
        return
    actual = {
        "n_train": len(data_module.train_dataset) if data_module.train_dataset is not None else 0,
        "n_val": len(data_module.val_dataset) if data_module.val_dataset is not None else 0,
        "n_test": len(data_module.test_dataset) if data_module.test_dataset is not None else 0,
    }
    for key, value in actual.items():
        if key in recorded and recorded[key] != value:
            logger.warning(
                f"Rebuilt split does not match the training run: {key} is {value} "
                f"but split_info.json records {recorded[key]}. Check --seed, "
                f"--val-split and --test-split against the training configuration."
            )
            return
