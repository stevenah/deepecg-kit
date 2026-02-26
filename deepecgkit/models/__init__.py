"""
Deep learning models for ECG analysis.

This module contains various neural network architectures optimized for ECG signal processing.
"""

from deepecgkit.models.af_classifier import AFModel
from deepecgkit.models.convnext_v2_1d import ConvNeXtV21D
from deepecgkit.models.crnn import CRNN
from deepecgkit.models.deep_res_cnn import DeepResCNN
from deepecgkit.models.ecg_dualnet import ECGDualNet
from deepecgkit.models.fcn_wang import FCNWang
from deepecgkit.models.gru_model import GRUECG
from deepecgkit.models.inception_time import InceptionTime1D
from deepecgkit.models.kanres_wide_x import KanResDeepX
from deepecgkit.models.kanres_x import KanResWideX
from deepecgkit.models.lstm_model import LSTMECG
from deepecgkit.models.mamba1d import Mamba1D
from deepecgkit.models.medformer import Medformer
from deepecgkit.models.resnet1d import ResNet1D
from deepecgkit.models.resnet1d_wang import ResNetWang
from deepecgkit.models.se_resnet1d import SEResNet1D
from deepecgkit.models.simple_cnn import SimpleCNN
from deepecgkit.models.tcn import TCN
from deepecgkit.models.transformer_ecg import TransformerECG
from deepecgkit.models.xresnet1d import XResNet1D
from deepecgkit.models.xresnet1d_benchmark import XResNet1dBenchmark

__all__ = [
    "CRNN",
    "GRUECG",
    "LSTMECG",
    "TCN",
    "AFModel",
    "ConvNeXtV21D",
    "DeepResCNN",
    "ECGDualNet",
    "FCNWang",
    "InceptionTime1D",
    "KanResDeepX",
    "KanResWideX",
    "Mamba1D",
    "Medformer",
    "ResNet1D",
    "ResNetWang",
    "SEResNet1D",
    "SimpleCNN",
    "TransformerECG",
    "XResNet1D",
    "XResNet1dBenchmark",
]
