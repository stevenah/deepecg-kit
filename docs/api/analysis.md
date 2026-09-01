# deepecgkit.analysis

Multi-seed analysis over finished training runs: discovery and aggregation of run directories,
ranking stability, the statistics behind both, and LaTeX table emitters. These modules back the
`deepecg aggregate` and `deepecg rank` commands.

## Aggregation

Discovers run directories (any directory holding `classification_metrics.csv`), groups them by
model, dataset and seed, and summarises every metric across seeds.

::: deepecgkit.analysis.aggregation

## Ranking

Rank stability and cross-dataset flip tests across random seeds: per-seed ranks, rank
distributions, Kendall's tau between seeds, and paired comparisons between models.

::: deepecgkit.analysis.ranking

## Statistics

Summary statistics and paired tests over per-seed scalar metrics, including Student-t intervals
and the Nadeau-Bengio correction.

::: deepecgkit.analysis.statistics

## LaTeX

booktabs table emitters for the seed-variance analysis.

::: deepecgkit.analysis.latex
