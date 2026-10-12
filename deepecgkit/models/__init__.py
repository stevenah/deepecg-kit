"""
Deep learning models for ECG analysis.

This module contains various neural network architectures optimized for ECG signal processing.
"""

from deepecgkit.models.af_classifier import AFModel
from deepecgkit.models.base import ECGModelBase
from deepecgkit.models.convnext_v2_1d import ConvNeXtV21D
from deepecgkit.models.crnn import CRNN
from deepecgkit.models.deep_res_cnn import DeepResCNN
from deepecgkit.models.ecg_dualnet import ECGDualNet
from deepecgkit.models.factory import (
    build_model,
    build_model_from_checkpoint,
    infer_input_channels,
    infer_output_size,
)
from deepecgkit.models.fcn_wang import FCNWang
from deepecgkit.models.gru_model import GRUECG
from deepecgkit.models.inception_time import InceptionTime1D
from deepecgkit.models.kanres_deep_x import KanResDeepX
from deepecgkit.models.kanres_wide_x import KanResWideX
from deepecgkit.models.lstm_model import LSTMECG
from deepecgkit.models.mamba1d import Mamba1D
from deepecgkit.models.medformer import Medformer
from deepecgkit.models.mlp_wang import MLPWang
from deepecgkit.models.pretrained import PretrainedMixin, from_pretrained, model_name_for_weights
from deepecgkit.models.resnet1d import ResNet1D
from deepecgkit.models.resnet1d_wang import ResNetWang
from deepecgkit.models.resnet_hannun import ResNetHannun
from deepecgkit.models.rocket import ROCKET
from deepecgkit.models.se_resnet1d import SEResNet1D
from deepecgkit.models.simple_cnn import SimpleCNN
from deepecgkit.models.spectrogram_cnn import SpectrogramCNN
from deepecgkit.models.tcn import TCN
from deepecgkit.models.transformer_ecg import TransformerECG
from deepecgkit.models.xresnet1d import XResNet1D
from deepecgkit.models.xresnet1d101_benchmark import XResNet1d101Benchmark
from deepecgkit.models.xresnet1d_benchmark import XResNet1dBenchmark

__all__ = [
    "CRNN",
    "GRUECG",
    "LSTMECG",
    "ROCKET",
    "TCN",
    "AFModel",
    "ConvNeXtV21D",
    "DeepResCNN",
    "ECGDualNet",
    "ECGModelBase",
    "FCNWang",
    "InceptionTime1D",
    "KanResDeepX",
    "KanResWideX",
    "MLPWang",
    "Mamba1D",
    "Medformer",
    "PretrainedMixin",
    "ResNet1D",
    "ResNetHannun",
    "ResNetWang",
    "SEResNet1D",
    "SimpleCNN",
    "SpectrogramCNN",
    "TransformerECG",
    "XResNet1D",
    "XResNet1d101Benchmark",
    "XResNet1dBenchmark",
    "build_model",
    "build_model_from_checkpoint",
    "from_pretrained",
    "infer_input_channels",
    "infer_output_size",
    "model_name_for_weights",
]
