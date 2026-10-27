# deepecgkit.evaluation

Model evaluation, metrics computation, and visualization tools.

## Evaluator

::: deepecgkit.evaluation.evaluator.ECGEvaluator

## Metrics

::: deepecgkit.evaluation.metrics.calculate_classification_metrics

::: deepecgkit.evaluation.metrics.calculate_regression_metrics

::: deepecgkit.evaluation.metrics.confusion_matrix_analysis

::: deepecgkit.evaluation.metrics.resolve_average

## Calibration

Calibration measures whether predicted probabilities match observed frequencies. Temperature
scaling corrects them without changing which class is predicted, and per-label Platt scaling
corrects label-specific offsets of sigmoid outputs.

::: deepecgkit.evaluation.metrics.calculate_calibration_metrics

::: deepecgkit.evaluation.metrics.expected_calibration_error

::: deepecgkit.evaluation.metrics.adaptive_calibration_error

::: deepecgkit.evaluation.metrics.maximum_calibration_error

::: deepecgkit.evaluation.metrics.brier_score

::: deepecgkit.evaluation.metrics.resolve_task_shape

## Recalibration

::: deepecgkit.evaluation.calibration.TemperatureScaler

::: deepecgkit.evaluation.calibration.fit_temperature

::: deepecgkit.evaluation.calibration.apply_temperature

::: deepecgkit.evaluation.calibration.temperature_nll

::: deepecgkit.evaluation.calibration.fit_platt

::: deepecgkit.evaluation.calibration.apply_platt

::: deepecgkit.evaluation.calibration.probabilities_to_logits

::: deepecgkit.evaluation.calibration.calibration_report

## Representation quality

::: deepecgkit.evaluation.representation.linear_probe

::: deepecgkit.evaluation.representation.probe_encoder

::: deepecgkit.evaluation.representation.in_domain_probe

::: deepecgkit.evaluation.representation.build_probe

::: deepecgkit.evaluation.representation.cross_dataset_transfer

::: deepecgkit.evaluation.representation.TransferMatrix

::: deepecgkit.evaluation.representation.ProbeResult

::: deepecgkit.evaluation.representation.protocol_data_module

::: deepecgkit.evaluation.representation.embed_2d

## Visualization

::: deepecgkit.evaluation.visualization.plot_confusion_matrix

::: deepecgkit.evaluation.visualization.plot_roc_curve

::: deepecgkit.evaluation.visualization.plot_training_curves

::: deepecgkit.evaluation.visualization.plot_ecg_signals

::: deepecgkit.evaluation.visualization.plot_predictions

::: deepecgkit.evaluation.visualization.plot_calibration_curve

::: deepecgkit.evaluation.visualization.plot_transfer_matrix

::: deepecgkit.evaluation.visualization.plot_embedding
