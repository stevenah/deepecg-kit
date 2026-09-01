"""
Pytest configuration and fixtures for deepecgkit tests.
"""

import tempfile
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pytest
import scipy.io
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_ecg_signal() -> torch.Tensor:
    """Create a sample ECG signal for testing (6 seconds at 500 Hz = 3000 samples)."""

    length = 3000
    t = np.linspace(0, 6, length)

    signal = (
        0.1 * np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * np.sin(2 * np.pi * 72 * t) * np.exp(-((t % (60 / 72) - 0.2) ** 2) / 0.01)
        + 0.1 * np.random.normal(0, 0.05, length)
    )

    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)


@pytest.fixture
def sample_batch_ecg() -> torch.Tensor:
    """Create a batch of sample ECG signals for testing (6 seconds)."""
    batch_size = 4
    length = 3000
    signals = []

    for i in range(batch_size):
        t = np.linspace(0, 6, length)
        signal = (
            0.1 * np.sin(2 * np.pi * 1.2 * t)
            + 0.3
            * np.sin(2 * np.pi * (70 + i * 2) * t)
            * np.exp(-((t % (60 / (70 + i * 2)) - 0.2) ** 2) / 0.01)
            + 0.1 * np.random.normal(0, 0.05, length)
        )
        signals.append(signal)

    return torch.tensor(signals, dtype=torch.float32).unsqueeze(1)


@pytest.fixture
def sample_batch_ecg_10s() -> torch.Tensor:
    """Create a batch of sample ECG signals for testing (10 seconds)."""
    batch_size = 4
    length = 5000
    signals = []

    for i in range(batch_size):
        t = np.linspace(0, 10, length)
        signal = (
            0.1 * np.sin(2 * np.pi * 1.2 * t)
            + 0.3
            * np.sin(2 * np.pi * (70 + i * 2) * t)
            * np.exp(-((t % (60 / (70 + i * 2)) - 0.2) ** 2) / 0.01)
            + 0.1 * np.random.normal(0, 0.05, length)
        )
        signals.append(signal)

    return torch.tensor(signals, dtype=torch.float32).unsqueeze(1)


@pytest.fixture
def sample_batch_ecg_30s() -> torch.Tensor:
    """Create a batch of sample ECG signals for testing (30 seconds)."""
    batch_size = 4
    length = 15000
    signals = []

    for i in range(batch_size):
        t = np.linspace(0, 30, length)
        signal = (
            0.1 * np.sin(2 * np.pi * 1.2 * t)
            + 0.3
            * np.sin(2 * np.pi * (70 + i * 2) * t)
            * np.exp(-((t % (60 / (70 + i * 2)) - 0.2) ** 2) / 0.01)
            + 0.1 * np.random.normal(0, 0.05, length)
        )
        signals.append(signal)

    return torch.tensor(signals, dtype=torch.float32).unsqueeze(1)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Create sample labels for classification."""
    return torch.tensor([0, 1, 2, 3], dtype=torch.long)


@pytest.fixture
def sample_probabilities() -> torch.Tensor:
    """Create sample probability distributions."""
    probs = torch.tensor(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.1],
            [0.05, 0.15, 0.6, 0.2],
            [0.1, 0.1, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    return probs


@pytest.fixture
def sample_regression_targets() -> torch.Tensor:
    """Create sample regression targets."""
    return torch.tensor([1.5, 2.3, 0.8, 3.1], dtype=torch.float32)


@pytest.fixture
def mock_dataset_files(temp_dir: Path):
    """Create mock dataset files for testing."""

    training_dir = temp_dir / "training2017"
    training_dir.mkdir()

    sample_dir = temp_dir / "sample2017"
    sample_dir.mkdir()

    reference_file = temp_dir / "REFERENCE-v3.csv"
    reference_file.write_text("A00001,N\nA00002,A\nA00003,O\nA00004,~\n")

    (temp_dir / "REFERENCE.csv").write_text("A00001,N\nA00002,A\nA00003,O\nA00004,~\n")
    (temp_dir / "REFERENCE-v2.csv").write_text("A00001,N\nA00002,A\nA00003,O\nA00004,~\n")
    (temp_dir / "REFERENCE-v1.csv").write_text("A00001,N\nA00002,A\nA00003,O\nA00004,~\n")

    for i, _label in enumerate(["N", "A", "O", "~"], 1):
        mat_file = training_dir / f"A0000{i}.mat"

        ecg_data = np.random.randn(1, 3000)
        scipy.io.savemat(str(mat_file), {"val": ecg_data})

        hea_file = training_dir / f"A0000{i}.hea"
        hea_file.write_text(f"A0000{i} 1 500 3000\nA0000{i}.mat 16 200 11 0 0 0 0 ECG\n")

        mat_file_root = temp_dir / f"A0000{i}.mat"
        scipy.io.savemat(str(mat_file_root), {"val": ecg_data})

        hea_file_root = temp_dir / f"A0000{i}.hea"
        hea_file_root.write_text(f"A0000{i} 1 500 3000\nA0000{i}.mat 16 200 11 0 0 0 0 ECG\n")

    return temp_dir


CALIBRATION_SAMPLES = 8000
CALIBRATION_SEED = 20240
CALIBRATION_LOGIT_SCALE = 1.2
OVERCONFIDENCE_FACTOR = 3.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax of a logit matrix."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Elementwise logistic function."""
    return 1.0 / (1.0 + np.exp(-logits))


def _sample_from_rows(probabilities: np.ndarray, rng) -> np.ndarray:
    """Draw one class index per row, distributed as that row."""
    cumulative = probabilities.cumsum(axis=1)
    draws = rng.random((probabilities.shape[0], 1))
    return (draws > cumulative).sum(axis=1)


@pytest.fixture
def calibrated_multiclass_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four-class predictions whose labels were drawn from the predictions themselves.

    A model whose labels come from its own predicted distribution is calibrated
    by construction, so its expected calibration error is sampling noise alone.
    """
    rng = np.random.default_rng(CALIBRATION_SEED)
    logits = rng.normal(0.0, CALIBRATION_LOGIT_SCALE, size=(CALIBRATION_SAMPLES, 4))
    probabilities = _softmax(logits)
    labels = _sample_from_rows(probabilities, rng)
    return labels, probabilities, logits


@pytest.fixture
def overconfident_multiclass_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The calibrated four-class predictions sharpened by a known factor.

    The labels are unchanged and only the logits are multiplied, so the
    temperature that undoes the damage is exactly OVERCONFIDENCE_FACTOR.
    """
    rng = np.random.default_rng(CALIBRATION_SEED)
    logits = rng.normal(0.0, CALIBRATION_LOGIT_SCALE, size=(CALIBRATION_SAMPLES, 4))
    labels = _sample_from_rows(_softmax(logits), rng)
    sharpened = logits * OVERCONFIDENCE_FACTOR
    return labels, _softmax(sharpened), sharpened


@pytest.fixture
def calibrated_binary_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-logit binary predictions stored as the two columns [1 - p, p].

    This is the layout ECGTrainer produces for the binary_classification flag,
    where the model head emits one logit rather than one per class.
    """
    rng = np.random.default_rng(CALIBRATION_SEED + 1)
    logits = rng.normal(0.0, CALIBRATION_LOGIT_SCALE, size=CALIBRATION_SAMPLES)
    positive = _sigmoid(logits)
    labels = (rng.random(CALIBRATION_SAMPLES) < positive).astype(int)
    return labels, np.column_stack([1.0 - positive, positive]), logits


@pytest.fixture
def overconfident_binary_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The calibrated binary predictions sharpened by a known factor."""
    rng = np.random.default_rng(CALIBRATION_SEED + 1)
    logits = rng.normal(0.0, CALIBRATION_LOGIT_SCALE, size=CALIBRATION_SAMPLES)
    labels = (rng.random(CALIBRATION_SAMPLES) < _sigmoid(logits)).astype(int)
    sharpened = logits * OVERCONFIDENCE_FACTOR
    positive = _sigmoid(sharpened)
    return labels, np.column_stack([1.0 - positive, positive]), sharpened


@pytest.fixture
def calibrated_multilabel_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five independent sigmoid outputs with multi-hot targets, as PTB-XL produces."""
    rng = np.random.default_rng(CALIBRATION_SEED + 2)
    logits = rng.normal(0.0, CALIBRATION_LOGIT_SCALE, size=(CALIBRATION_SAMPLES, 5))
    probabilities = _sigmoid(logits)
    labels = (rng.random((CALIBRATION_SAMPLES, 5)) < probabilities).astype(int)
    return labels, probabilities, logits


@pytest.fixture
def overconfident_multilabel_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The calibrated multi-label predictions sharpened by a known factor."""
    rng = np.random.default_rng(CALIBRATION_SEED + 2)
    logits = rng.normal(0.0, CALIBRATION_LOGIT_SCALE, size=(CALIBRATION_SAMPLES, 5))
    labels = (rng.random((CALIBRATION_SAMPLES, 5)) < _sigmoid(logits)).astype(int)
    sharpened = logits * OVERCONFIDENCE_FACTOR
    return labels, _sigmoid(sharpened), sharpened


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
