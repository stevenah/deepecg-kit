# deepecgkit.prediction

Predictions on raw recordings from a trained checkpoint. The recording is resampled, segmented
and normalised the way the checkpoint's training data was, and the model's outputs are read the
way its head was trained: a softmax over classes, one sigmoid for a single-logit binary head, or a
sigmoid per class for a multi-label one.

::: deepecgkit.prediction.predict

::: deepecgkit.prediction.ECGPredictor

::: deepecgkit.prediction.Prediction
