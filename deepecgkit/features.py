"""Extracting deep embeddings from trained ECG models.

Every registered architecture exposes ``extract_features`` and ``feature_dim``,
but using them on real data means loading a checkpoint, batching, moving
tensors to a device, and keeping labels aligned. This module does that once so
that obtaining embeddings for clustering, visualisation, transfer learning, or
a linear probe is a single call.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

from deepecgkit.datasets.io import as_model_input, read_ecg_recording
from deepecgkit.datasets.modules import ECGDataModule
from deepecgkit.models.factory import build_model_from_checkpoint, infer_input_channels
from deepecgkit.models.pretrained import from_pretrained as _load_pretrained
from deepecgkit.models.pretrained import model_name_for_weights

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = [
    "FeatureExtractor",
    "FeatureSet",
    "extract_features",
]

FeatureInput = Union[np.ndarray, torch.Tensor, str, Path, Dataset, DataLoader, ECGDataModule]

SPLITS = ("train", "val", "test", "all")

_DTYPES: Dict[str, torch.dtype] = {"float32": torch.float32, "float16": torch.float16}


def _resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device name, including the CLI's accelerator vocabulary.

    Args:
        device: One of auto, cpu, cuda, gpu, mps, or any string torch accepts

    Returns:
        The resolved device. "auto" prefers CUDA, then MPS, then CPU, matching
        ECGTrainer; "gpu" is accepted as an alias for CUDA.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "gpu":
        return torch.device("cuda")
    return torch.device(device)


@dataclass
class FeatureSet:
    """Embeddings extracted from a model, with any aligned labels.

    Attributes:
        features: Array of shape (num_samples, feature_dim)
        labels: Aligned labels of shape (num_samples,) for single-label data or
            (num_samples, num_classes) for multi-label data, or None when the
            input carried no labels
        model_name: Registry name of the model that produced the features
    """

    features: np.ndarray
    labels: Optional[np.ndarray] = None
    model_name: Optional[str] = field(default=None)

    @property
    def feature_dim(self) -> int:
        """Width of the embedding."""
        return int(self.features.shape[1])

    def __len__(self) -> int:
        """Number of samples."""
        return int(self.features.shape[0])

    def __array__(self, dtype: Optional[Any] = None) -> np.ndarray:
        """Expose the features to numpy so the object can stand in for an array."""
        if dtype is None:
            return self.features
        return self.features.astype(dtype)

    def save(self, path: Union[str, Path]) -> None:
        """Write the features to disk, choosing the format from the suffix.

        Args:
            path: Destination ending in .npz, .npy, or .csv. The .npz form
                keeps the labels and metadata; .npy stores features only; .csv
                writes one column per feature plus a label column.

        Raises:
            ValueError: If the suffix is not one of the three supported forms.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix == ".npy":
            np.save(str(out_path), self.features)
            return

        if out_path.suffix == ".npz":
            payload: Dict[str, Any] = {
                "features": self.features,
                "feature_dim": np.asarray(self.feature_dim),
            }
            if self.labels is not None:
                payload["labels"] = self.labels
            if self.model_name is not None:
                payload["model_name"] = np.asarray(self.model_name)
            np.savez(str(out_path), **payload)
            return

        if out_path.suffix == ".csv":
            import pandas as pd

            frame = pd.DataFrame(self.features, columns=[f"f{i}" for i in range(self.feature_dim)])
            if self.labels is not None:
                labels = self.labels
                if labels.ndim == 1:
                    frame["label"] = labels
                else:
                    for i in range(labels.shape[1]):
                        frame[f"label_{i}"] = labels[:, i]
            frame.to_csv(str(out_path), index=False)
            return

        raise ValueError(
            f"Unsupported output format: {out_path.suffix}. Supported formats: .npz, .npy, .csv"
        )


class FeatureExtractor(TransformerMixin, BaseEstimator):
    """Runs a model's feature path over data and returns the embeddings.

    The extractor is a scikit-learn transformer, so it drops straight into a
    Pipeline ahead of any downstream estimator. Note that sklearn's ``clone``
    deep-copies the wrapped model, which doubles its memory.

    Args:
        model: Any model exposing extract_features and feature_dim
        model_name: Registry name recorded on the results, for provenance
        batch_size: Samples per forward pass, ignored for a supplied DataLoader
        device: Device name, or auto to prefer CUDA then MPS then CPU
        dtype: Precision of the returned features, float32 or float16
        num_workers: Worker processes for loaders this extractor constructs
        progress: Show a progress bar when tqdm is installed

    Example:
        >>> extractor = FeatureExtractor.from_checkpoint("best.ckpt", "kanres")
        >>> result = extractor.extract(signals)
        >>> result.features.shape
        (100, 64)
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        device: str = "auto",
        dtype: str = "float32",
        num_workers: int = 0,
        progress: bool = False,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.num_workers = num_workers
        self.progress = progress

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        model_name: str,
        input_channels: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs: Any,
    ) -> "FeatureExtractor":
        """Build an extractor from a saved checkpoint.

        Args:
            checkpoint: Path to an ECGTrainer checkpoint, a bare state dict, or
                a legacy PyTorch Lightning checkpoint
            model_name: Registered model name matching the weights
            input_channels: Lead count, required for architectures whose weights
                do not encode it
            output_size: Classifier width, normally recovered from the weights
            **kwargs: Forwarded to the constructor

        Returns:
            An extractor wrapping the loaded model.
        """
        model, _ = build_model_from_checkpoint(
            checkpoint,
            name=model_name,
            input_channels=input_channels,
            output_size=output_size,
        )
        return cls(model, model_name=model_name, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        weights: Union[str, Path],
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "FeatureExtractor":
        """Build an extractor from registered pretrained weights or a weights file.

        Args:
            weights: Name of registered pretrained weights, or a path
            model_name: Registered model name; inferred from the weight registry
                when the weights are registered
            **kwargs: Forwarded to the constructor, except that recognised model
                constructor arguments are passed to the model instead

        Returns:
            An extractor wrapping the loaded model.

        Raises:
            ValueError: If the model name is neither given nor inferable.
        """
        resolved = model_name or model_name_for_weights(str(weights))
        if resolved is None:
            raise ValueError(
                f"Could not determine which model '{weights}' belongs to. Pass model_name "
                f"explicitly."
            )
        extractor_keys = {"batch_size", "device", "dtype", "num_workers", "progress"}
        extractor_kwargs = {k: v for k, v in kwargs.items() if k in extractor_keys}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in extractor_keys}
        model = _load_pretrained(resolved, weights, **model_kwargs)
        return cls(model, model_name=resolved, **extractor_kwargs)

    @property
    def feature_dim(self) -> int:
        """Width of the embedding the wrapped model produces.

        Raises:
            AttributeError: If the model does not declare feature_dim.
        """
        if not hasattr(self.model, "feature_dim"):
            raise AttributeError(
                f"{type(self.model).__name__} does not declare feature_dim, so it cannot be "
                f"used for feature extraction."
            )
        return int(self.model.feature_dim)

    @property
    def input_channels(self) -> Optional[int]:
        """Lead count recovered from the weights, or None when unrecoverable."""
        declared = getattr(self.model, "input_channels", None)
        if isinstance(declared, int):
            return declared
        return infer_input_channels(self.model.state_dict())

    def extract(self, data: FeatureInput, split: str = "test") -> FeatureSet:
        """Run the model's feature path over the data.

        Args:
            data: An array, tensor, file path, Dataset, DataLoader, or
                ECGDataModule
            split: Which split to use when data is an ECGDataModule, one of
                train, val, test, or all

        Returns:
            The embeddings, with labels when the input supplied them.

        Raises:
            ValueError: If the split or dtype is unknown.
            RuntimeError: If the model returns a non-2D representation.
        """
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}. Choose one of {', '.join(SPLITS)}.")
        if self.dtype not in _DTYPES:
            raise ValueError(
                f"Unknown dtype: {self.dtype}. Choose one of {', '.join(sorted(_DTYPES))}."
            )

        out_dtype = _DTYPES[self.dtype]
        device = _resolve_device(self.device)
        loader = self._as_dataloader(data, split)

        was_training = self.model.training
        self.model.eval()
        self.model.to(device)

        feature_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        try:
            iterator: Any = loader
            if self.progress and tqdm is not None:
                iterator = tqdm(loader, desc="Extracting features")
            with torch.no_grad():
                for batch in iterator:
                    signals, labels = _unpack_batch(batch)
                    signals = signals.to(device=device, dtype=torch.float32)
                    features = self.model.extract_features(signals)
                    if features.dim() != 2:
                        raise RuntimeError(
                            f"{type(self.model).__name__}.extract_features returned a "
                            f"{features.dim()}-D tensor; a 2-D (batch, feature_dim) tensor is "
                            f"required."
                        )
                    feature_chunks.append(features.detach().to(device="cpu", dtype=out_dtype))
                    if labels is not None:
                        label_chunks.append(labels.detach().cpu())
        finally:
            if was_training:
                self.model.train()

        if not feature_chunks:
            empty = np.empty((0, self.feature_dim), dtype=np.dtype(self.dtype))
            return FeatureSet(empty, None, self.model_name)

        features_array = torch.cat(feature_chunks, dim=0).numpy()
        labels_array = None
        if label_chunks and len(label_chunks) == len(feature_chunks):
            labels_array = torch.cat(label_chunks, dim=0).numpy()

        return FeatureSet(features_array, labels_array, self.model_name)

    def fit(self, X: Any = None, y: Any = None) -> "FeatureExtractor":  # noqa: N803
        """Return self; the extractor holds a trained model and learns nothing.

        Args:
            X: Ignored, present for scikit-learn compatibility
            y: Ignored, present for scikit-learn compatibility

        Returns:
            This extractor.
        """
        return self

    def transform(self, X: FeatureInput) -> np.ndarray:  # noqa: N803
        """Return the embeddings as a plain array, for use inside a Pipeline.

        Args:
            X: Any input accepted by extract

        Returns:
            Array of shape (num_samples, feature_dim).
        """
        return self.extract(X).features

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> np.ndarray:
        """Return names for the embedding dimensions.

        Args:
            input_features: Ignored, present for scikit-learn compatibility

        Returns:
            Array of feature names.
        """
        prefix = self.model_name or type(self.model).__name__.lower()
        return np.asarray([f"{prefix}_{i}" for i in range(self.feature_dim)], dtype=object)

    def _as_dataloader(self, data: FeatureInput, split: str) -> DataLoader:
        """Normalise any supported input to a non-shuffling DataLoader.

        Args:
            data: Any supported input
            split: Split name, used only for an ECGDataModule

        Returns:
            A loader yielding either signals alone or (signals, labels).

        Raises:
            TypeError: If the input type is not supported.
            RuntimeError: If a data module split is unavailable.
        """
        if isinstance(data, ECGDataModule):
            return self._loader_for_module(data, split)

        if isinstance(data, DataLoader):
            if isinstance(data.sampler, RandomSampler):
                warnings.warn(
                    "The supplied DataLoader shuffles, so extracted rows will not follow "
                    "dataset order and cannot be joined to external metadata. Pass "
                    "shuffle=False for a reproducible ordering.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return data

        if isinstance(data, Dataset):
            return self._build_loader(data)

        if isinstance(data, (str, Path)):
            array = as_model_input(read_ecg_recording(data), self.input_channels)
            return self._build_loader(TensorDataset(torch.as_tensor(array)))

        if isinstance(data, torch.Tensor):
            array = as_model_input(data.detach().cpu().numpy(), self.input_channels)
            return self._build_loader(TensorDataset(torch.as_tensor(array)))

        if isinstance(data, np.ndarray):
            array = as_model_input(data, self.input_channels)
            return self._build_loader(TensorDataset(torch.as_tensor(array)))

        raise TypeError(
            f"Unsupported input type: {type(data).__name__}. Pass an array, tensor, file path, "
            f"Dataset, DataLoader, or ECGDataModule."
        )

    def _loader_for_module(self, data: ECGDataModule, split: str) -> DataLoader:
        """Build a non-shuffling loader over one split of a data module.

        The module's own train_dataloader shuffles, which would scramble row
        order, so the loader is constructed here instead.

        Args:
            data: The data module
            split: One of train, val, test, or all

        Returns:
            A loader over the requested split.

        Raises:
            RuntimeError: If the split holds no dataset after setup.
        """
        if data.dataset is None or data.train_dataset is None:
            data.setup()

        subset = data.dataset if split == "all" else getattr(data, f"{split}_dataset")
        if subset is None:
            raise RuntimeError(f"Data module has no '{split}' split available.")
        return self._build_loader(subset)

    def _build_loader(self, dataset: Dataset) -> DataLoader:
        """Wrap a dataset in a loader that preserves ordering.

        Args:
            dataset: Any map-style dataset

        Returns:
            A DataLoader with shuffling disabled.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split a batch into signals and optional labels.

    Args:
        batch: A tensor, or a sequence whose first element is the signals

    Returns:
        Tuple of the signals and the labels, which are None when absent.
    """
    if isinstance(batch, (list, tuple)):
        signals = batch[0]
        labels = batch[1] if len(batch) > 1 else None
        return signals, labels
    return batch, None


def _iter_inputs(**sources: Any) -> Iterator[Tuple[str, Any]]:
    """Yield the weight sources that were actually supplied.

    Args:
        **sources: Candidate sources keyed by argument name

    Returns:
        Iterator over the non-empty (name, value) pairs.
    """
    return ((name, value) for name, value in sources.items() if value is not None)


def extract_features(
    data: FeatureInput,
    model: Optional[nn.Module] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    weights: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    input_channels: Optional[int] = None,
    output_size: Optional[int] = None,
    split: str = "test",
    **kwargs: Any,
) -> FeatureSet:
    """Extract deep embeddings in one call.

    Exactly one weight source must be given: an already-built model, a
    checkpoint path, or pretrained weights.

    Args:
        data: An array, tensor, file path, Dataset, DataLoader, or ECGDataModule
        model: An already-built model
        checkpoint: Path to a checkpoint, requiring model_name
        weights: Registered pretrained weight name, or a path to a weights file
        model_name: Registered model name
        input_channels: Lead count, when it cannot be recovered from the weights
        output_size: Classifier width, when it cannot be recovered
        split: Which split to use when data is an ECGDataModule
        **kwargs: Forwarded to FeatureExtractor

    Returns:
        The embeddings, with labels when the input supplied them.

    Raises:
        ValueError: If not exactly one weight source is given, or a checkpoint
            is given without a model name.

    Example:
        >>> result = extract_features(signals, checkpoint="best.ckpt", model_name="kanres")
        >>> result.features.shape
        (100, 64)
    """
    supplied = dict(_iter_inputs(model=model, checkpoint=checkpoint, weights=weights))
    if len(supplied) != 1:
        raise ValueError(
            f"Pass exactly one of model, checkpoint, or weights; got "
            f"{', '.join(sorted(supplied)) or 'none'}."
        )

    if model is not None:
        extractor = FeatureExtractor(model, model_name=model_name, **kwargs)
    elif checkpoint is not None:
        if model_name is None:
            raise ValueError("model_name is required when loading from a checkpoint.")
        extractor = FeatureExtractor.from_checkpoint(
            checkpoint,
            model_name,
            input_channels=input_channels,
            output_size=output_size,
            **kwargs,
        )
    else:
        extractor = FeatureExtractor.from_pretrained(weights, model_name=model_name, **kwargs)

    return extractor.extract(data, split=split)
