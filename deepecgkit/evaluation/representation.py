"""Measuring how much a frozen encoder's representation is actually worth.

A classifier's headline accuracy conflates two things: the quality of the
representation and the quality of the head fitted on top of it. Freezing the
encoder and fitting nothing but a logistic regression separates them, and
running that probe on a dataset the encoder was never trained on asks the
question the embeddings exist to answer, namely whether an architecture learns
transferable ECG structure or a shortcut peculiar to one corpus.

Two things make the numbers interpretable rather than decorative. Features are
standardised inside the probe pipeline, because embedding scales differ by
orders of magnitude across the registered architectures and an unstandardised
L2 penalty therefore means something different for each one. And an untrained
encoder is probed alongside the trained ones, because a probe score without its
own floor cannot be read at all: much of what a linear head recovers from a
random convolutional encoder is recoverable from any random convolutional
encoder.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deepecgkit.datasets.modules import ECGDataModule
from deepecgkit.evaluation.metrics import (
    BINARY_THRESHOLD,
    TASK_MULTILABEL,
    calculate_classification_metrics,
    resolve_task_shape,
)
from deepecgkit.features import FeatureExtractor, FeatureSet
from deepecgkit.models.factory import build_model
from deepecgkit.registry import get_dataset

__all__ = [
    "DEFAULT_PROBE_METRIC",
    "KEEP_EMBEDDINGS_MODES",
    "PROBE_DATASETS",
    "PROBE_INPUT_CHANNELS",
    "PROBE_METRICS",
    "PROBE_SAMPLING_RATE",
    "PROTOCOL_DATASET_KWARGS",
    "RANDOM_INIT",
    "ProbeResult",
    "TransferMatrix",
    "build_probe",
    "cross_dataset_transfer",
    "embed_2d",
    "linear_probe",
    "probe_encoder",
    "protocol_data_module",
]

RANDOM_INIT = "random-init"

PROBE_DATASETS = ("af-classification", "ptbxl", "mitbih-afdb", "ltafdb")

PROBE_SAMPLING_RATE = 300
PROBE_INPUT_CHANNELS = 1
PROBE_SEGMENT_SECONDS = 10.0

PROTOCOL_DATASET_KWARGS: Dict[str, Dict[str, Any]] = {"ptbxl": {"leads": ["II"]}}

RANDOM_INIT_OUTPUT_SIZE = 2

METRIC_AUROC_MACRO = "auroc_macro"
METRIC_ACCURACY = "accuracy"
METRIC_BALANCED_ACCURACY = "balanced_accuracy"
METRIC_F1_MACRO = "f1_macro"

PROBE_METRICS = (
    METRIC_AUROC_MACRO,
    METRIC_ACCURACY,
    METRIC_BALANCED_ACCURACY,
    METRIC_F1_MACRO,
)
DEFAULT_PROBE_METRIC = METRIC_AUROC_MACRO

EMBEDDING_METHODS = ("tsne", "pca")

MIN_EMBEDDING_SAMPLES = 3
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_PERPLEXITY = 30.0
PERPLEXITY_SAMPLE_DIVISOR = 3.0

TRANSFER_KEY_SEPARATOR = "->"

KEEP_NONE = "none"
KEEP_DIAGONAL = "diagonal"
KEEP_ALL = "all"
KEEP_EMBEDDINGS_MODES = (KEEP_NONE, KEEP_DIAGONAL, KEEP_ALL)


def _nan_metrics() -> Dict[str, float]:
    """Return the metric dictionary used when a cell could not be scored.

    Returns:
        Every protocol metric mapped to NaN.
    """
    return {name: float("nan") for name in PROBE_METRICS}


@dataclass
class ProbeResult:
    """The outcome of fitting one linear probe on one held-out split.

    Attributes:
        metrics: Every metric in PROBE_METRICS, with NaN where a metric was not
            defined for this label set
        metric: Name of the headline metric
        score: Value of the headline metric, a convenience view on metrics
        task: Resolved label shape, one of binary, multiclass or multilabel
        n_train: Samples the probe was fitted on
        n_test: Samples the probe was scored on
        feature_dim: Width of the embedding the probe consumed
        model_name: Registry name of the encoder that produced the embeddings
        source: Dataset the encoder was trained on, or RANDOM_INIT
        target: Dataset the probe was fitted and scored on
        note: Why a metric is missing, when one is
    """

    metrics: Dict[str, float]
    metric: str = DEFAULT_PROBE_METRIC
    task: Optional[str] = None
    n_train: int = 0
    n_test: int = 0
    feature_dim: int = 0
    model_name: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    note: Optional[str] = field(default=None)

    @property
    def score(self) -> float:
        """Value of the headline metric."""
        return float(self.metrics.get(self.metric, float("nan")))

    def to_dict(self) -> Dict[str, Any]:
        """Render the result as JSON-serialisable primitives.

        Returns:
            A dictionary whose float values are plain floats, so NaN survives
            json.dumps as the non-standard but universally parsed NaN literal.
        """
        return {
            "source": self.source,
            "target": self.target,
            "model_name": self.model_name,
            "metric": self.metric,
            "score": self.score,
            "task": self.task,
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "feature_dim": int(self.feature_dim),
            "metrics": {name: float(value) for name, value in self.metrics.items()},
            "note": self.note,
        }


def build_probe(
    c: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    multilabel: bool = False,
) -> Pipeline:
    """Build the protocol's probe: standardisation followed by logistic regression.

    The scaler is a pipeline step rather than a preprocessing call so that it is
    fitted on the training split alone and merely applied to the held-out split.
    Standardising the two together, or standardising the pooled features once
    before splitting, leaks the test distribution into the fit.

    Args:
        c: Inverse regularisation strength passed to LogisticRegression
        max_iter: Solver iteration ceiling
        random_state: Seed for the solver
        multilabel: Wrap the estimator in OneVsRestClassifier, for targets whose
            labels are a multi-hot matrix rather than a class index

    Returns:
        An unfitted pipeline of a StandardScaler and a logistic regression.
    """
    estimator: Any = LogisticRegression(C=c, max_iter=max_iter, random_state=random_state)
    if multilabel:
        estimator = OneVsRestClassifier(estimator)
    return Pipeline([("scaler", StandardScaler()), ("probe", estimator)])


def _as_arrays(data: Union[FeatureSet, Tuple[Any, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise a FeatureSet or a (features, labels) pair to two arrays.

    Args:
        data: A FeatureSet carrying labels, or a pair of arrays

    Returns:
        Tuple of the features and the labels.

    Raises:
        TypeError: If the input is neither of the two supported forms.
        ValueError: If a FeatureSet carries no labels.
    """
    if isinstance(data, FeatureSet):
        if data.labels is None:
            raise ValueError(
                "A linear probe needs labels, but this FeatureSet carries none. Extract from a "
                "Dataset, DataLoader or ECGDataModule rather than a bare array."
            )
        return np.asarray(data.features), np.asarray(data.labels)

    if isinstance(data, (tuple, list)) and len(data) == 2:
        return np.asarray(data[0]), np.asarray(data[1])

    raise TypeError(
        f"Unsupported probe input: {type(data).__name__}. Pass a FeatureSet or a "
        f"(features, labels) pair."
    )


def _is_multilabel(labels: np.ndarray) -> bool:
    """Report whether a label array is a multi-hot matrix.

    Args:
        labels: Label array of one or two dimensions

    Returns:
        True when the labels have a class axis wider than one.
    """
    return labels.ndim > 1 and labels.shape[-1] > 1


def _multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Score independent per-label sigmoids without collapsing them to one class.

    calculate_classification_metrics takes the argmax of any two-dimensional
    prediction, which is the right reading for a softmax and the wrong one here,
    so the multi-label case is computed directly. Accuracy is the label-wise
    mean rather than the exact-match ratio, and AUROC is averaged over the label
    columns that carry both classes.

    Args:
        y_true: Multi-hot targets of shape (n_samples, n_labels)
        y_prob: Independent per-label probabilities of the same shape

    Returns:
        Every metric in PROBE_METRICS, with NaN where none of the columns were
        scoreable.
    """
    y_pred = (y_prob >= BINARY_THRESHOLD).astype(int)
    metrics = _nan_metrics()
    metrics[METRIC_ACCURACY] = float((y_true == y_pred).mean())

    try:
        metrics[METRIC_F1_MACRO] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except ValueError:
        pass

    aurocs: List[float] = []
    balanced: List[float] = []
    for column in range(y_true.shape[1]):
        truth = y_true[:, column]
        if len(np.unique(truth)) < 2:
            continue
        aurocs.append(float(roc_auc_score(truth, y_prob[:, column])))
        balanced.append(float(balanced_accuracy_score(truth, y_pred[:, column])))

    if aurocs:
        metrics[METRIC_AUROC_MACRO] = float(np.mean(aurocs))
        metrics[METRIC_BALANCED_ACCURACY] = float(np.mean(balanced))
    return metrics


def _single_label_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Score a class-index target against a probability matrix.

    Args:
        y_true: Class indices of shape (n_samples,)
        y_prob: Probabilities of shape (n_samples, n_classes)

    Returns:
        Every metric in PROBE_METRICS, with NaN where one was not computable.
    """
    base = calculate_classification_metrics(
        y_true, y_prob, metrics=["accuracy", "f1", "auc"], average="macro"
    )
    metrics = _nan_metrics()
    metrics[METRIC_ACCURACY] = float(base.get("accuracy", float("nan")))
    metrics[METRIC_F1_MACRO] = float(base.get("f1", float("nan")))
    metrics[METRIC_AUROC_MACRO] = float(base.get("auc", float("nan")))
    try:
        metrics[METRIC_BALANCED_ACCURACY] = float(
            balanced_accuracy_score(y_true, np.argmax(y_prob, axis=1))
        )
    except ValueError:
        pass
    return metrics


def _probe_metrics(y_true: np.ndarray, y_prob: np.ndarray, task: str) -> Dict[str, float]:
    """Dispatch scoring on the resolved label shape.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        task: One of TASK_BINARY, TASK_MULTICLASS or TASK_MULTILABEL

    Returns:
        Every metric in PROBE_METRICS.
    """
    if task == TASK_MULTILABEL:
        return _multilabel_metrics(y_true, y_prob)
    return _single_label_metrics(y_true, y_prob)


def _probe_probabilities(pipeline: Pipeline, features: np.ndarray) -> np.ndarray:
    """Obtain probabilities from a fitted probe in a shape the scorers accept.

    Args:
        pipeline: A fitted probe pipeline
        features: Embeddings of shape (n_samples, feature_dim)

    Returns:
        Probabilities of shape (n_samples, n_classes) or (n_samples, n_labels).
    """
    probabilities = np.asarray(pipeline.predict_proba(features))
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    return probabilities


def linear_probe(
    train: Union[FeatureSet, Tuple[Any, Any]],
    test: Union[FeatureSet, Tuple[Any, Any]],
    metric: str = DEFAULT_PROBE_METRIC,
    model_name: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    **probe_kwargs: Any,
) -> ProbeResult:
    """Fit a standardised logistic probe on training embeddings and score it on held-out ones.

    Args:
        train: Embeddings and labels the probe is fitted on
        test: Embeddings and labels the probe is scored on, never seen by the fit
        metric: Headline metric, one of PROBE_METRICS
        model_name: Registry name of the encoder, recorded on the result
        source: Dataset the encoder was trained on, recorded on the result
        target: Dataset being probed, recorded on the result
        **probe_kwargs: Forwarded to build_probe

    Returns:
        The probe's scores. A target whose training split holds a single class
        cannot be fitted at all and comes back with NaN metrics and a note
        rather than an exception, so one degenerate cell does not abort a
        transfer matrix.

    Raises:
        ValueError: If the metric is unknown, the two splits disagree on
            embedding width, or either split carries no labels.
    """
    if metric not in PROBE_METRICS:
        raise ValueError(
            f"Unknown probe metric: {metric}. Choose one of {', '.join(PROBE_METRICS)}."
        )

    train_features, train_labels = _as_arrays(train)
    test_features, test_labels = _as_arrays(test)

    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError(
            f"Embedding width differs between the splits: {train_features.shape[1]} for train and "
            f"{test_features.shape[1]} for test. Both splits must come from the same encoder."
        )

    resolved_name = model_name
    if resolved_name is None and isinstance(train, FeatureSet):
        resolved_name = train.model_name

    multilabel = _is_multilabel(train_labels)
    result = ProbeResult(
        metrics=_nan_metrics(),
        metric=metric,
        task=TASK_MULTILABEL if multilabel else None,
        n_train=int(train_features.shape[0]),
        n_test=int(test_features.shape[0]),
        feature_dim=int(train_features.shape[1]),
        model_name=resolved_name,
        source=source,
        target=target,
    )

    pipeline = build_probe(multilabel=multilabel, **probe_kwargs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(train_features, train_labels)
            probabilities = _probe_probabilities(pipeline, test_features)
    except ValueError as error:
        result.note = f"probe could not be fitted: {error}"
        return result

    result.task = resolve_task_shape(test_labels, probabilities)
    result.metrics = _probe_metrics(test_labels, probabilities, result.task)
    if not np.isfinite(result.score):
        result.note = f"{metric} is undefined for this held-out split"
    return result


def probe_encoder(
    extractor: FeatureExtractor,
    data_module: ECGDataModule,
    train_split: str = "train",
    test_split: str = "test",
    metric: str = DEFAULT_PROBE_METRIC,
    source: Optional[str] = None,
    target: Optional[str] = None,
    **probe_kwargs: Any,
) -> ProbeResult:
    """Embed two splits of one data module with a frozen encoder and probe them.

    Args:
        extractor: A frozen encoder; nothing here updates its weights
        data_module: The data module supplying both splits
        train_split: Split the probe is fitted on
        test_split: Split the probe is scored on
        metric: Headline metric, one of PROBE_METRICS
        source: Dataset the encoder was trained on, recorded on the result
        target: Dataset being probed, recorded on the result
        **probe_kwargs: Forwarded to build_probe

    Returns:
        The probe's scores on the held-out split.
    """
    train_features = extractor.extract(data_module, split=train_split)
    test_features = extractor.extract(data_module, split=test_split)
    return linear_probe(
        train_features,
        test_features,
        metric=metric,
        model_name=extractor.model_name,
        source=source,
        target=target,
        **probe_kwargs,
    )


@dataclass
class TransferMatrix:
    """Probe performance for every (encoder source, probe target) pair.

    Rows are the datasets the encoders were trained on, columns are the datasets
    the probes were fitted and scored on. The diagonal is the in-domain probe.
    When the random-init floor is included it is the first row, and it is the
    floor for every target column; transpose to_frame() for a figure that wants
    sources along the top instead.

    Attributes:
        sources: Row labels, RANDOM_INIT first when the baseline is present
        targets: Column labels
        values: Headline metric per cell, of shape (len(sources), len(targets))
        metric: Name of the headline metric
        results: Full ProbeResult per cell, keyed "source->target"
        model_name: Registry name of the probed architecture
        protocol: The resolved protocol settings, recorded for provenance
        embeddings: Held-out embeddings retained for the cells requested by
            keep_embeddings, keyed "source->target", for plotting. Empty unless
            they were asked for, since keeping all of them costs memory
            proportional to the corpus
        class_names: Class names per target, for labelling an embedding plot
    """

    sources: List[str]
    targets: List[str]
    values: np.ndarray
    metric: str = DEFAULT_PROBE_METRIC
    results: Dict[str, ProbeResult] = field(default_factory=dict)
    model_name: Optional[str] = None
    protocol: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, FeatureSet] = field(default_factory=dict)
    class_names: Dict[str, List[str]] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Render the headline metric as a labelled table.

        Returns:
            A frame indexed by source with one column per target.
        """
        return pd.DataFrame(self.values, index=list(self.sources), columns=list(self.targets))

    def embedding_for(self, source: str, target: str) -> Optional[FeatureSet]:
        """Look up the retained held-out embeddings behind one cell.

        Args:
            source: Row label
            target: Column label

        Returns:
            The cell's embeddings, or None when they were not retained.
        """
        return self.embeddings.get(f"{source}{TRANSFER_KEY_SEPARATOR}{target}")

    def result_for(self, source: str, target: str) -> Optional[ProbeResult]:
        """Look up the full result behind one cell.

        Args:
            source: Row label
            target: Column label

        Returns:
            The cell's ProbeResult, or None when the pair was not run.
        """
        return self.results.get(f"{source}{TRANSFER_KEY_SEPARATOR}{target}")

    def to_dict(self) -> Dict[str, Any]:
        """Render the whole matrix as JSON-serialisable primitives.

        Returns:
            A dictionary holding the headline table, every cell's full metrics,
            and the protocol settings the run used.
        """
        return {
            "model_name": self.model_name,
            "metric": self.metric,
            "sources": list(self.sources),
            "targets": list(self.targets),
            "values": [[float(value) for value in row] for row in np.asarray(self.values)],
            "protocol": dict(self.protocol),
            "cells": {key: result.to_dict() for key, result in self.results.items()},
        }

    def save(self, path: Union[str, Path]) -> None:
        """Write the matrix to disk, choosing the format from the suffix.

        Args:
            path: Destination ending in .json or .csv. The .json form keeps
                every metric of every cell and the protocol settings; .csv holds
                the headline table alone.

        Raises:
            ValueError: If the suffix is neither of the two supported forms.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
            return

        if out_path.suffix == ".csv":
            self.to_frame().to_csv(str(out_path))
            return

        raise ValueError(
            f"Unsupported output format: {out_path.suffix}. Supported formats: .json, .csv"
        )


def protocol_data_module(
    dataset_name: str,
    data_dir: Optional[str] = None,
    sampling_rate: int = PROBE_SAMPLING_RATE,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    download: bool = True,
    force_download: bool = False,
    verbose: bool = False,
) -> ECGDataModule:
    """Build one dataset in the single-lead configuration the protocol requires.

    Transfer only means anything if every encoder sees the same input shape, so
    the protocol runs every dataset at one lead and one sampling rate. The three
    AF corpora already yield a single lead at their constructor defaults;
    PTB-XL is pinned to lead II and resampled to match, which its standardiser
    handles.

    Args:
        dataset_name: A registered dataset name
        data_dir: Directory holding the dataset, or None for its default
        sampling_rate: Target rate for every dataset, in Hz
        batch_size: Batch size for the loaders
        num_workers: Worker processes for the loaders
        val_split: Fraction held out for validation
        test_split: Fraction held out for the probe's report
        seed: Split seed
        download: Fetch the dataset when it is absent
        force_download: Re-download an existing dataset
        verbose: Print dataset construction progress

    Returns:
        A data module that has not yet been set up.
    """
    dataset_kwargs: Dict[str, Any] = {
        "force_download": force_download,
        "verbose": verbose,
    }
    dataset_kwargs.update(PROTOCOL_DATASET_KWARGS.get(dataset_name, {}))

    return ECGDataModule(
        dataset_class=get_dataset(dataset_name),
        data_dir=data_dir,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        download=download,
        verbose=verbose,
        dataset_kwargs=dataset_kwargs,
    )


def _random_init_extractor(
    model_name: str,
    input_channels: int,
    seed: int,
    **extractor_kwargs: Any,
) -> FeatureExtractor:
    """Build the untrained encoder that supplies the matrix's floor.

    One encoder is built and reused across every target, so the floor row
    describes a single random network rather than a different one per column.
    Its classifier head is never called, which is why its width is fixed.

    Args:
        model_name: Registered model name
        input_channels: Lead count the protocol runs at
        seed: Seed drawn before construction, making the floor reproducible
        **extractor_kwargs: Forwarded to FeatureExtractor

    Returns:
        An extractor wrapping an untrained model.
    """
    torch.manual_seed(seed)
    model = build_model(
        model_name,
        input_channels=input_channels,
        output_size=RANDOM_INIT_OUTPUT_SIZE,
    )
    return FeatureExtractor(model, model_name=model_name, **extractor_kwargs)


def _keeps_embedding(mode: str, source: str, target: str) -> bool:
    """Report whether one cell's held-out embeddings should be retained.

    Args:
        mode: One of the KEEP_EMBEDDINGS_MODES
        source: Row label
        target: Column label

    Returns:
        True when the cell is in the requested subset.
    """
    if mode == KEEP_ALL:
        return True
    if mode == KEEP_DIAGONAL:
        return source in (target, RANDOM_INIT)
    return False


def cross_dataset_transfer(
    checkpoints: Dict[str, Union[str, Path]],
    model_name: str,
    targets: Optional[Sequence[str]] = None,
    input_channels: int = PROBE_INPUT_CHANNELS,
    metric: str = DEFAULT_PROBE_METRIC,
    include_random_init: bool = True,
    data_modules: Optional[Dict[str, ECGDataModule]] = None,
    sampling_rate: int = PROBE_SAMPLING_RATE,
    data_dir: Optional[str] = None,
    download: bool = True,
    force_download: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto",
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    progress: bool = False,
    verbose: bool = False,
    keep_embeddings: str = KEEP_NONE,
    **probe_kwargs: Any,
) -> TransferMatrix:
    """Probe every source encoder on every target dataset.

    Each cell freezes the encoder trained on its source, embeds the target's
    training and held-out splits, fits a standardised logistic head on the
    training embeddings and reports on the held-out ones. The head is refitted
    per target because the label spaces differ and no head can be carried
    across them; what transfers, or fails to, is the representation.

    Args:
        checkpoints: Source dataset name mapped to the checkpoint of the encoder
            trained on it
        model_name: Registered model name matching every checkpoint
        targets: Datasets to probe, defaulting to the supplied data modules or
            else to PROBE_DATASETS
        input_channels: Lead count the protocol runs at
        metric: Headline metric filling the matrix, one of PROBE_METRICS
        include_random_init: Prepend the untrained-encoder floor as the first row
        data_modules: Pre-built modules keyed by target name, bypassing dataset
            construction entirely
        sampling_rate: Target rate for every dataset, in Hz
        data_dir: Directory holding the datasets, or None for their defaults
        download: Fetch a dataset when it is absent
        force_download: Re-download existing datasets
        batch_size: Batch size for embedding extraction
        num_workers: Worker processes for the loaders
        device: Device name, or auto to prefer CUDA then MPS then CPU
        val_split: Fraction held out for validation
        test_split: Fraction the probe reports on
        seed: Seed for the splits and for the random-init encoder
        progress: Show a progress bar during extraction
        verbose: Print dataset construction progress
        keep_embeddings: Which cells keep their held-out embeddings for later
            plotting: none, all, or diagonal for the in-domain cells together
            with the random-init row, which is the pairing worth looking at
        **probe_kwargs: Forwarded to build_probe

    Returns:
        The filled transfer matrix.

    Raises:
        ValueError: If no sources are given, or a name is not a registered
            dataset, or the metric is unknown.
    """
    if metric not in PROBE_METRICS:
        raise ValueError(
            f"Unknown probe metric: {metric}. Choose one of {', '.join(PROBE_METRICS)}."
        )

    if keep_embeddings not in KEEP_EMBEDDINGS_MODES:
        raise ValueError(
            f"Unknown keep_embeddings mode: {keep_embeddings}. Choose one of "
            f"{', '.join(KEEP_EMBEDDINGS_MODES)}."
        )

    if not checkpoints and not include_random_init:
        raise ValueError(
            "Nothing to probe: pass at least one source checkpoint, or keep the random-init floor."
        )

    if targets is None:
        targets = sorted(data_modules) if data_modules else list(PROBE_DATASETS)
    resolved_targets = list(targets)
    if not resolved_targets:
        raise ValueError("No probe targets were given.")

    extractor_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "device": device,
        "num_workers": num_workers,
        "progress": progress,
    }

    modules: Dict[str, ECGDataModule] = dict(data_modules or {})
    for name in resolved_targets:
        if name in modules:
            continue
        modules[name] = protocol_data_module(
            name,
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
            download=download,
            force_download=force_download,
            verbose=verbose,
        )

    sources: List[str] = []
    extractors: Dict[str, FeatureExtractor] = {}
    if include_random_init:
        sources.append(RANDOM_INIT)
        extractors[RANDOM_INIT] = _random_init_extractor(
            model_name, input_channels, seed, **extractor_kwargs
        )
    for name in sorted(checkpoints):
        sources.append(name)
        extractors[name] = FeatureExtractor.from_checkpoint(
            checkpoints[name],
            model_name,
            input_channels=input_channels,
            **extractor_kwargs,
        )

    values = np.full((len(sources), len(resolved_targets)), np.nan, dtype=float)
    results: Dict[str, ProbeResult] = {}
    retained: Dict[str, FeatureSet] = {}
    class_names: Dict[str, List[str]] = {}
    for row, source in enumerate(sources):
        for column, target in enumerate(resolved_targets):
            module = modules[target]
            train_features = extractors[source].extract(module, split="train")
            test_features = extractors[source].extract(module, split="test")
            result = linear_probe(
                train_features,
                test_features,
                metric=metric,
                model_name=model_name,
                source=source,
                target=target,
                **probe_kwargs,
            )
            key = f"{source}{TRANSFER_KEY_SEPARATOR}{target}"
            values[row, column] = result.score
            results[key] = result
            if _keeps_embedding(keep_embeddings, source, target):
                retained[key] = test_features
            if target not in class_names:
                names = getattr(module.dataset, "class_names", None)
                if names:
                    class_names[target] = list(names)

    protocol = {
        "input_channels": int(input_channels),
        "sampling_rate": int(sampling_rate),
        "val_split": float(val_split),
        "test_split": float(test_split),
        "seed": int(seed),
        "random_init": bool(include_random_init),
        "dataset_kwargs": {
            name: PROTOCOL_DATASET_KWARGS[name]
            for name in resolved_targets
            if name in PROTOCOL_DATASET_KWARGS
        },
    }

    return TransferMatrix(
        sources=sources,
        targets=resolved_targets,
        values=values,
        metric=metric,
        results=results,
        model_name=model_name,
        protocol=protocol,
        embeddings=retained,
        class_names=class_names,
    )


def embed_2d(
    features: Any,
    method: str = "tsne",
    perplexity: float = DEFAULT_PERPLEXITY,
    pca_components: Optional[int] = DEFAULT_PCA_COMPONENTS,
    random_state: int = 42,
) -> np.ndarray:
    """Project embeddings to two dimensions for plotting.

    Features are standardised first, for the same reason the probe standardises
    them: a handful of wide-range dimensions would otherwise dominate the
    distances t-SNE is built from. Wide embeddings are then reduced by PCA
    before t-SNE, which is both the conventional recipe and much faster.

    Args:
        features: Embeddings of shape (n_samples, feature_dim), or a FeatureSet
        method: One of tsne or pca
        perplexity: t-SNE perplexity, clamped to a value the sample count
            supports rather than letting scikit-learn reject it
        pca_components: Width to reduce to before t-SNE, or None to skip
        random_state: Seed, making the projection reproducible

    Returns:
        Coordinates of shape (n_samples, 2).

    Raises:
        ValueError: If the method is unknown, the input is not two-dimensional,
            or there are too few samples to project.
    """
    if method not in EMBEDDING_METHODS:
        raise ValueError(
            f"Unknown embedding method: {method}. Choose one of {', '.join(EMBEDDING_METHODS)}."
        )

    data = np.asarray(features, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"Embeddings must be two-dimensional (n_samples, feature_dim); got shape {data.shape}."
        )
    if data.shape[0] < MIN_EMBEDDING_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_EMBEDDING_SAMPLES} samples to project; got {data.shape[0]}."
        )

    scaled = StandardScaler().fit_transform(data)
    if method == "pca":
        return np.asarray(PCA(n_components=2, random_state=random_state).fit_transform(scaled))

    width = min(pca_components or 0, scaled.shape[0], scaled.shape[1])
    if pca_components and scaled.shape[1] > width > 2:
        scaled = PCA(n_components=width, random_state=random_state).fit_transform(scaled)

    ceiling = max(1.0, (scaled.shape[0] - 1) / PERPLEXITY_SAMPLE_DIVISOR)
    projection = TSNE(
        n_components=2,
        perplexity=float(min(perplexity, ceiling)),
        random_state=random_state,
        init="pca",
    ).fit_transform(scaled)
    return np.asarray(projection)
