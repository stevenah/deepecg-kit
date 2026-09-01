"""Tests for the single shared device resolver."""

import pytest
import torch

from deepecgkit.utils.device import resolve_device


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_torch_device_passthrough(self):
        dev = torch.device("cpu")
        assert resolve_device(dev) is dev

    def test_auto_prefers_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == torch.device("cuda")

    def test_auto_prefers_mps_over_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
            assert resolve_device("auto") == torch.device("mps")

    def test_auto_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device("auto") == torch.device("cpu")

    def test_gpu_alias_resolves_to_cuda(self, monkeypatch):
        """Regression: --accelerator gpu used to crash train/evaluate with
        torch.device('gpu')."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("gpu") == torch.device("cuda")

    def test_explicit_cuda_unavailable_raises_clearly(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")

    def test_explicit_gpu_unavailable_raises_clearly(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("gpu")

    def test_invalid_device_string_raises(self):
        with pytest.raises(RuntimeError):
            resolve_device("not-a-device")


class TestResolverConsumers:
    def test_trainer_accepts_gpu_alias(self, monkeypatch):
        """ECGTrainer previously passed 'gpu' verbatim to torch.device."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        from deepecgkit.models import SimpleCNN
        from deepecgkit.training import ECGTrainer

        with pytest.raises(RuntimeError, match="CUDA was requested"):
            ECGTrainer(
                model=SimpleCNN(input_channels=1, output_size=4),
                train_config={
                    "learning_rate": 1e-3,
                    "scheduler": {"factor": 0.5, "patience": 5},
                },
                device="gpu",
            )

    def test_evaluator_uses_shared_resolver(self, monkeypatch):
        """ECGEvaluator previously ignored MPS entirely (silent CPU on Apple
        Silicon)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
            from deepecgkit.evaluation.evaluator import ECGEvaluator

            assert ECGEvaluator(device="auto").device == torch.device("mps")
