# deepecgkit.models

Neural network architectures for ECG signal processing. All models are `nn.Module` subclasses
with a consistent interface: `__init__(input_channels, output_size, **kwargs)`.

## CNN Architectures

::: deepecgkit.models.simple_cnn.SimpleCNN

::: deepecgkit.models.fcn_wang.FCNWang

::: deepecgkit.models.af_classifier.AFModel

## Residual Networks

::: deepecgkit.models.resnet1d.ResNet1D

::: deepecgkit.models.resnet1d_wang.ResNetWang

::: deepecgkit.models.se_resnet1d.SEResNet1D

::: deepecgkit.models.xresnet1d.XResNet1D

::: deepecgkit.models.xresnet1d_benchmark.XResNet1dBenchmark

::: deepecgkit.models.kanres_x.KanResWideX

::: deepecgkit.models.kanres_wide_x.KanResDeepX

::: deepecgkit.models.deep_res_cnn.DeepResCNN

## Modern CNN

::: deepecgkit.models.convnext_v2_1d.ConvNeXtV21D

::: deepecgkit.models.inception_time.InceptionTime1D

::: deepecgkit.models.tcn.TCN

## Recurrent Networks

::: deepecgkit.models.crnn.CRNN

::: deepecgkit.models.gru_model.GRUECG

::: deepecgkit.models.lstm_model.LSTMECG

## Transformers & Attention

::: deepecgkit.models.transformer_ecg.TransformerECG

::: deepecgkit.models.medformer.Medformer

## Hybrid & State Space

::: deepecgkit.models.ecg_dualnet.ECGDualNet

::: deepecgkit.models.mamba1d.Mamba1D
