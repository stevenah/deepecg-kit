"""
Tests for deepecgkit.prediction: one call from a checkpoint and a raw recording to a label.
"""

import numpy as np
import pytest
import torch

from deepecgkit import ECGPredictor, Prediction, predict
from deepecgkit.datasets import AFClassificationDataset, ECGDataModule, prepare_recording
from deepecgkit.models import SimpleCNN
from deepecgkit.training import ECGTrainer

AF_CLASSES = ["Normal", "AF", "Other", "Noisy"]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A run directory holding one epoch of SimpleCNN on the synthetic AF dataset."""
    from tests.conftest import write_af_dataset

    root = tmp_path_factory.mktemp("prediction")
    data = ECGDataModule(
        AFClassificationDataset,
        data_dir=str(write_af_dataset(root / "data")),
        batch_size=16,
        verbose=False,
        dataset_kwargs={"verbose": False},
    )
    trainer = ECGTrainer(SimpleCNN(1, 4), device="cpu")
    trainer.fit(data, epochs=1, checkpoint_dir=root / "run" / "checkpoints", progress_bar=False)
    return root / "run"


def _recording(seconds: float, rate: float) -> np.ndarray:
    time = np.arange(int(seconds * rate)) / rate
    return 300.0 * np.sin(2 * np.pi * 1.2 * time)


def _save_checkpoint(path, model, data_config, train_config=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"name": "simple-cnn", "input_channels": 1},
            "data_config": data_config,
            "train_config": train_config or {},
        },
        path,
    )
    return path


class TestPredict:
    """The one-call path a newcomer takes."""

    def test_the_result_names_the_classes_and_sums_to_one(self, run_dir):
        result = predict(_recording(30, 300), checkpoint=run_dir)

        assert isinstance(result, Prediction)
        assert list(result.probabilities) == AF_CLASSES
        assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert result.label in AF_CLASSES
        assert result.labels == [result.label]
        assert result.confidence == max(result.probabilities.values())

    def test_probabilities_are_the_softmax_of_the_model_averaged_over_segments(self, run_dir):
        recording = _recording(30, 300)
        predictor = ECGPredictor.from_checkpoint(run_dir, device="cpu")
        prepared = prepare_recording(recording[np.newaxis], predictor.preprocessing, 300)
        with torch.no_grad():
            expected = torch.softmax(predictor.model(torch.from_numpy(prepared.segments)), dim=1)

        result = predictor.predict(recording, sampling_rate=300)

        np.testing.assert_allclose(result.segment_probabilities, expected.numpy(), atol=1e-6)
        np.testing.assert_allclose(
            list(result.probabilities.values()), expected.mean(dim=0).numpy(), atol=1e-6
        )

    def test_a_recording_at_another_rate_is_resampled_and_cut_into_segments(self, run_dir):
        result = predict(_recording(30, 250), checkpoint=run_dir, sampling_rate=250)

        assert result.num_segments == 3
        np.testing.assert_allclose(result.segment_starts, [0.0, 10.0, 20.0])
        assert any("resampled from 250 Hz to 300 Hz" in note for note in result.notes)

    def test_a_file_predicts_as_its_array_does(self, run_dir, tmp_path):
        recording = _recording(20, 300)
        np.save(tmp_path / "ecg.npy", recording)

        from_file = predict(tmp_path / "ecg.npy", checkpoint=run_dir)
        from_array = predict(recording, checkpoint=run_dir)

        assert from_file.probabilities == pytest.approx(from_array.probabilities)

    def test_a_run_directory_stands_for_its_best_checkpoint(self, run_dir):
        (checkpoint,) = (run_dir / "checkpoints").glob("*.pt")
        recording = _recording(10, 300)

        assert predict(recording, run_dir).probabilities == pytest.approx(
            predict(recording, checkpoint).probabilities
        )

    def test_a_tensor_is_accepted(self, run_dir):
        recording = _recording(10, 300)

        assert predict(torch.tensor(recording), run_dir).probabilities == pytest.approx(
            predict(recording, run_dir).probabilities
        )


class TestSeveralRecordings:
    def test_predict_refuses_input_holding_several_recordings(self, run_dir):
        rows = np.stack([_recording(10, 300), _recording(10, 300)])

        with pytest.raises(ValueError, match="holds 2 recordings; predict_all"):
            predict(rows, run_dir)

    def test_predict_all_gives_one_prediction_per_row(self, run_dir):
        rows = np.stack([_recording(10, 300), -_recording(10, 300), _recording(10, 300)])

        results = ECGPredictor.from_checkpoint(run_dir, device="cpu").predict_all(rows)

        assert len(results) == 3
        assert results[0].probabilities == pytest.approx(results[2].probabilities)


class TestHeads:
    """Each kind of head is read the way it was trained."""

    def test_a_multi_label_head_gets_a_sigmoid_per_class(self, tmp_path):
        model = SimpleCNN(1, 3).eval()
        path = _save_checkpoint(
            tmp_path / "multi.pt",
            model,
            {"class_names": ["NORM", "MI", "STTC"], "multi_label": True, "preprocessing": None},
            {"multi_label": True},
        )
        recording = _recording(10, 300)
        with torch.no_grad():
            logits = model(torch.tensor(recording, dtype=torch.float32).reshape(1, 1, -1))

        result = predict(recording, path, device="cpu")

        np.testing.assert_allclose(
            list(result.probabilities.values()), torch.sigmoid(logits)[0].numpy(), atol=1e-6
        )
        assert result.multi_label
        present = [name for name, p in result.probabilities.items() if p > 0.5]
        assert result.labels == present
        assert result.label == (", ".join(present) or "none")

    def test_a_single_logit_head_gives_both_classes(self, tmp_path):
        model = SimpleCNN(1, 1).eval()
        path = _save_checkpoint(
            tmp_path / "binary.pt",
            model,
            {"class_names": ["Non-AF", "AF"], "preprocessing": None},
            {"binary_classification": True},
        )
        recording = _recording(10, 300)
        with torch.no_grad():
            positive = torch.sigmoid(
                model(torch.tensor(recording, dtype=torch.float32).reshape(1, 1, -1))
            ).item()

        result = predict(recording, path, device="cpu")

        assert result.probabilities == pytest.approx({"Non-AF": 1 - positive, "AF": positive})

    def test_unnamed_classes_are_numbered(self, tmp_path):
        path = _save_checkpoint(tmp_path / "unnamed.pt", SimpleCNN(1, 2), {})

        result = predict(_recording(5, 300), path, device="cpu")

        assert list(result.probabilities) == ["class 0", "class 1"]
        assert result.num_segments == 1
