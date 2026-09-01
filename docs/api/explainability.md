# deepecgkit.explainability

Attribution for trained ECG models. Each method assigns every sample of every lead a contribution to
one output class, and the surrounding helpers reduce that map to a distribution over leads and to a
beat-aligned profile with its P, QRS and T shares.

The package re-exports every symbol below, so `from deepecgkit.explainability import ECGExplainer`
works whichever submodule a name is defined in. The plotting helpers are imported on first use, so
importing the package does not pull matplotlib into the path of every command.

## Explainer

::: deepecgkit.explainability.attribution.explain

::: deepecgkit.explainability.attribution.ECGExplainer

::: deepecgkit.explainability.attribution.Attribution

## Attribution methods

::: deepecgkit.explainability.attribution.saliency

::: deepecgkit.explainability.attribution.integrated_gradients

::: deepecgkit.explainability.attribution.occlusion

::: deepecgkit.explainability.attribution.lead_importance

## Beat alignment

::: deepecgkit.explainability.beats.detect_r_peaks

::: deepecgkit.explainability.beats.beat_aligned_attribution

::: deepecgkit.explainability.beats.pqrst_fractions

::: deepecgkit.explainability.beats.pqrst_enrichment

::: deepecgkit.explainability.beats.region_duration_fractions

## Model constraints

::: deepecgkit.explainability.constraints.length_bounds

::: deepecgkit.explainability.constraints.fit_to_constraints

## Visualisation

::: deepecgkit.explainability.visualization.plot_attribution

::: deepecgkit.explainability.visualization.plot_lead_importance

::: deepecgkit.explainability.visualization.plot_beat_profile
