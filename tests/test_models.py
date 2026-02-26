"""
Tests for deepecgkit.models module.
"""

import pytest
import torch
from torch import nn

from deepecgkit.models import AFModel, KanResWideX
from deepecgkit.models.af_classifier import ConvBlock
from deepecgkit.models.kanres_x import ConvBlock as KanResConvBlock
from deepecgkit.models.kanres_x import KanResModule


class TestConvBlock:
    """Test the ConvBlock component."""

    def test_conv_block_init(self):
        """Test ConvBlock initialization."""
        block = ConvBlock(1, 64, 8, dropout_rate=0.2, use_dropout=True)

        assert isinstance(block.conv, nn.Conv1d)
        assert isinstance(block.batch_norm, nn.BatchNorm1d)
        assert block.use_dropout is True
        assert hasattr(block, "dropout")

    def test_conv_block_forward(self, sample_ecg_signal):
        """Test ConvBlock forward pass."""
        block = ConvBlock(1, 64, 8)
        output = block(sample_ecg_signal.unsqueeze(0))

        assert output.shape[0] == 1
        assert output.shape[1] == 64
        assert output.shape[2] < sample_ecg_signal.shape[1]

    def test_conv_block_no_dropout(self, sample_ecg_signal):
        """Test ConvBlock without dropout."""
        block = ConvBlock(1, 32, 8, use_dropout=False)
        output = block(sample_ecg_signal.unsqueeze(0))

        assert output.shape[1] == 32


class TestAFModel:
    """Test the AFModel for atrial fibrillation classification."""

    @pytest.mark.parametrize("recording_length", [6, 10, 30])
    def test_af_model_init(self, recording_length):
        """Test AFModel initialization with different recording lengths."""
        model = AFModel(recording_length=recording_length)

        assert model.recording_length == recording_length
        assert hasattr(model, "conv_layers")
        assert hasattr(model, "output")
        assert isinstance(model.output, nn.Linear)

    def test_af_model_invalid_recording_length(self):
        """Test AFModel with invalid recording length."""
        with pytest.raises(AssertionError, match="Recording length must be 6, 10, or 30"):
            AFModel(recording_length=15)

    @pytest.mark.parametrize(
        "recording_length,expected_input_size", [(6, 3000), (10, 5000), (30, 15000)]
    )
    def test_af_model_forward(self, recording_length, expected_input_size):
        """Test AFModel forward pass."""
        model = AFModel(recording_length=recording_length)

        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, expected_input_size)

        output = model(input_tensor)

        assert output.shape == (batch_size, 4)
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_af_model_get_features(self):
        """Test AFModel feature extraction."""
        model = AFModel(recording_length=10)

        # AFModel doesn't have get_features method, using get_feature_size instead
        feature_size = model.get_feature_size(5000)

        assert isinstance(feature_size, int)
        assert feature_size > 0

    def test_af_model_weight_initialization(self):
        """Test that model weights are properly initialized."""
        model = AFModel(recording_length=10)

        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
            elif isinstance(module, nn.Linear):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))


class TestKanResModule:
    """Test the KanResModule component."""

    def test_kanres_module_init(self):
        """Test KanResModule initialization."""
        module = KanResModule(channels=32)

        assert isinstance(module.conv1, KanResConvBlock)
        assert isinstance(module.conv2, KanResConvBlock)

    def test_kanres_module_forward(self):
        """Test KanResModule forward pass with residual connection."""
        module = KanResModule(channels=32)

        input_tensor = torch.randn(2, 32, 100)
        output = module(input_tensor)

        assert output.shape == input_tensor.shape

        assert not torch.allclose(output, input_tensor)


class TestKanResWideX:
    """Test the KanResWideX model."""

    def test_kanres_widex_init(self):
        """Test KanResWideX initialization."""
        model = KanResWideX(input_channels=1, output_size=4, base_channels=64)

        assert isinstance(model.input_layer, KanResConvBlock)
        assert isinstance(model.res_modules, nn.Sequential)
        assert isinstance(model.classifier, nn.Linear)
        assert isinstance(model.global_pool, nn.AdaptiveAvgPool1d)

    def test_kanres_widex_forward(self, sample_batch_ecg):
        """Test KanResWideX forward pass."""
        model = KanResWideX(input_channels=1, output_size=4)

        output = model(sample_batch_ecg)

        assert output.shape == (sample_batch_ecg.shape[0], 4)
        assert output.dtype == torch.float32

    def test_kanres_widex_different_configs(self):
        """Test KanResWideX with different configurations."""
        configs = [
            {"input_channels": 12, "output_size": 2, "base_channels": 32},
            {"input_channels": 1, "output_size": 1, "base_channels": 16},
            {"input_channels": 3, "output_size": 10, "base_channels": 128},
        ]

        for config in configs:
            model = KanResWideX(**config)

            batch_size = 2
            seq_length = 1000
            input_tensor = torch.randn(batch_size, config["input_channels"], seq_length)

            output = model(input_tensor)
            assert output.shape == (batch_size, config["output_size"])

    def test_kanres_widex_adaptive_pooling(self):
        """Test that adaptive pooling works with different input lengths."""
        model = KanResWideX(input_channels=1, output_size=4)

        for seq_length in [500, 1000, 3000, 5000]:
            input_tensor = torch.randn(2, 1, seq_length)
            output = model(input_tensor)
            assert output.shape == (2, 4)

    def test_kanres_widex_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = KanResWideX(input_channels=1, output_size=4)
        input_tensor = torch.randn(2, 1, 1000, requires_grad=True)

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelIntegration:
    """Integration tests for model components."""

    def test_models_can_be_imported(self):
        """Test that all models can be imported from the package."""
        assert AFModel is not None
        assert KanResWideX is not None

    def test_models_work_together(self, sample_batch_ecg):
        """Test that different models can process the same input."""

        af_input = torch.randn(sample_batch_ecg.shape[0], 1, 5000)

        af_model = AFModel(recording_length=10)
        kanres_model = KanResWideX(input_channels=1, output_size=4)

        af_output = af_model(af_input)
        kanres_output = kanres_model(sample_batch_ecg)

        assert af_output.shape == kanres_output.shape
        assert af_output.shape == (sample_batch_ecg.shape[0], 4)

    def test_model_device_compatibility(self):
        """Test model compatibility with different devices."""
        model = KanResWideX(input_channels=1, output_size=4)

        input_cpu = torch.randn(2, 1, 1000)
        output_cpu = model(input_cpu)
        assert output_cpu.device.type == "cpu"

        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_gpu = input_cpu.cuda()
            output_gpu = model_gpu(input_gpu)
            assert output_gpu.device.type == "cuda"

    def test_model_eval_mode(self, sample_batch_ecg):
        """Test that models work correctly in evaluation mode."""
        model = AFModel(recording_length=10)

        af_input = torch.randn(sample_batch_ecg.shape[0], 1, 5000)

        model.train()
        output_train = model(af_input)

        model.eval()
        output_eval = model(af_input)

        assert output_train.shape == output_eval.shape
