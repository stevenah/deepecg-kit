# Changelog

Notable changes to DeepECG-Kit. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Feature extraction (`deepecg extract-features`, `FeatureExtractor`).
- Explainability: attribution methods, beat-level analysis, visualizations
  (`deepecg explain`, `ECGExplainer`).
- Probability calibration (temperature scaling) in evaluation.
- Representation-quality probing (`deepecg probe`).
- Multi-seed experiment support with aggregation and ranking-stability
  analysis, now first-class commands: `deepecg aggregate` and `deepecg rank`
  (the `scripts/` entry points remain as shims).
- `mamba` and `medformer` architectures.
- Typed package: `py.typed` shipped, all functions annotated.
- Run-directory contract: `deepecg train` writes `run_metadata.json`, its
  default output directory gains a `-seed{N}` suffix, and the analysis
  pipeline consumes CLI runs directly.
- `deepecg evaluate` auto-adopts the recorded split configuration from the
  run directory, verifies rebuilt split sizes, and writes the full artifact
  set with `-o/--output`.
- `extract-features` and `explain` accept `--seed/--val-split/--test-split`
  so their split matches the training partition.
- Datasets: `download_to()` classmethod, `strict=True` to re-raise record
  load errors, and loud warnings whenever source records are skipped or
  stratification is silently downgraded.
- Expanded public API: `build_model`, `build_model_from_checkpoint`,
  `ECGEvaluator`, registry functions, `resolve_device`, `DataSplitter`,
  `PretrainedMixin` are importable from their natural packages.

### Changed
- Python 3.10+ required (the package already used 3.10-only syntax at runtime).
- Dropped unused dependencies `pyEDFlib` and `PyWavelets`.
- Library diagnostics go through `logging` under the `deepecgkit.*`
  hierarchy; the CLI's `-v`/`-q` now govern all output. Enable
  `logging.basicConfig(level=logging.INFO)` to see dataset progress from
  library code.
- `ECGEvaluator` uses the shared device resolver and now selects MPS on
  Apple Silicon under `device="auto"` (previously CPU).
- LTAFDB subsampling (`max_segments_per_record`) is seeded
  (`subsample_seed`, default 0) and part of the cache key; previously it was
  irreproducible and frozen into the cache.

### Fixed
- `deepecgkit/analysis/` was accidentally gitignored and missing from the
  repository history.
- `--config` files were silently ignored for every key with a non-None CLI
  default (epochs, batch size, learning rate, ...); config now applies to
  every command with documented precedence.
- `--accelerator gpu` crashed train/evaluate; `predict` ignored
  `auto`/`mps` and always ran on CPU.
- `download=True` was ignored when the dataset directory existed but was
  empty.
- Growing the LTAFDB cache memmap silently kept its original capacity,
  dropping records past the initial estimate.

### Removed
- `deepecg resume` (a stub that always failed since introduction) and the
  broken `examples/cli_*.sh` scripts.

## [0.2.0] - earlier

Initial tracked baseline (training, evaluation, 19 architectures, 5 datasets,
CLI, documentation site).
