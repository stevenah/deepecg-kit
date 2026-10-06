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
- Self-describing checkpoints: `ECGTrainer` records the architecture
  (`model_config`) and the data it was trained on (`data_config`: dataset,
  sampling rate, segment length, normalisation, class names, split). `--model`
  is therefore optional for `evaluate`, `predict`, `extract-features` and
  `explain`; `ECGTrainer.load_checkpoint(path)` and
  `build_model_from_checkpoint(path)` rebuild the model without being told
  which one it is; a contradicting model name is reported in one line.
- `deepecg predict`, `extract-features --input` and `explain --input` put a
  recording through the checkpoint's preprocessing (resample, segment,
  normalise) before it reaches the model, report class names, and take
  `--sampling-rate` for the recording's own rate (`--no-preprocess` opts
  out). `deepecgkit.datasets.prepare_recording` does the same in Python.
- `--checkpoint` accepts a run directory, standing for its best checkpoint;
  `deepecg train` prints that checkpoint and the next command when it ends.
- `deepecg train --segment-duration`, and `--sampling-rate` /
  `--normalization` for every dataset rather than PTB-XL alone.
- `--verbose`, `--quiet` and `--config` work after the command as well as
  before it; a config file may supply required arguments such as `model` and
  `dataset`; `deepecg --version`.
- `deepecg --import MODULE` (and `DEEPECGKIT_PLUGINS`) makes models and
  datasets registered outside the package usable from every command.
  `ECGModelBase` is exported from `deepecgkit.models`.
- `DEEPECGKIT_HOME` relocates downloaded datasets and cached weights.
- Downloads resume after an interruption, are written to `.part` files until
  complete, use PhysioNet's open-data mirror with physionet.org as the
  fallback, and are serialised across processes. `Dataset.is_downloaded()`
  reports whether a usable copy is present.
- `deepecg list-datasets` shows classes, signal shape, accepted options,
  download size and local status; `deepecg list-models` shows input-length
  limits.
- `ECGDataModule.describe()` and `BaseECGDataset.get_preprocessing()`.
- A progress bar over the batches of the current epoch on interactive
  terminals.

### Changed
- **Datasets are used at their own sampling rate.** `ECGDataModule` no longer
  defaults `sampling_rate` to 500, which silently resampled every dataset
  that passed through it (and so every CLI run): the AF datasets now train
  at their default 300 Hz, giving 3000-sample segments instead of 5000.
  PTB-XL is unchanged at 500 Hz. Pass `--sampling-rate 500` to reproduce
  earlier runs. Checkpoints that predate recorded preprocessing are assumed
  to be 500 Hz when evaluated, with a warning.
- `run_metadata.json` (schema version 2) records the preprocessing that was
  in effect — `sampling_rate`, `normalization`, `segment_duration_seconds`,
  `input_length` — instead of the flags as typed, and the run's outcome
  (`returncode`, `error`, `best_checkpoint`). The run directory is created
  once the data has loaded and the model has been checked against it.
- A flag the dataset cannot honour (`--binary-classification` or
  `--multi-label` on a dataset without that mode, `--segment-duration` on
  PTB-XL) is now an error. Previously it was accepted, ignored, and recorded.
- `ECGTrainer`'s `train_config` is optional and every key has a default.
  `ECGTrainer.validate()` returns `val_loss` / `val_acc` as documented
  (it returned `test_*` keys).
- `num_workers` defaults to 0 for `ECGDataModule` and every command: the
  built-in datasets hold their signals in memory, so workers only add
  start-up cost, most of all under the spawn start method.
- `af-classification` downloads the three files its loader reads (about
  100 MB) instead of the whole 1.4 GB project archive; MIT-BIH AFDB, LTAFDB
  and PTB-XL are fetched record by record from the mirror.
- `deepecg evaluate` prints the full metric set whether or not `-o` is given,
  and `--calibration-bins` applies in both cases.
- `deepecg predict --batch-size` defaults to 32 and counts segments per
  forward pass, now that a recording is cut into segments.
- Signals of `af-classification` and `mitbih-afdb` are stored as float32,
  halving memory.
- `list_pretrained_weights()` and `deepecg list-models` list only weights
  that can be downloaded; none are published yet.
- `mitbih-afdb` and `ltafdb` are registered with one input channel, which is
  what their default configuration yields.
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
- **LTAFDB was labelled almost entirely Normal.** Its annotation files hold
  an annotation per heartbeat between the rhythm changes, and every beat
  ended the rhythm episode before it, so an AF episode lasted one beat: none
  of the 705,816 ten-second segments was labelled AF, where the annotations
  put 52.6% of them in AF. Rhythm labels now run from one rhythm change to
  the next. This changes every `ltafdb` result and, through LTAFDB's share
  of it, every `unified-af` result; MIT-BIH AFDB, whose files hold rhythm
  changes only, is unaffected. The LTAFDB segment cache is versioned so
  mislabelled caches are not read back; they are reported so they can be
  deleted.
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
- An interrupted dataset download left a truncated archive that every later
  run failed on (`File is not a zip file`), with no path or remedy in the
  message. Incomplete downloads are now detected and completed, and a missing
  dataset names its directory and how to fetch it.
- `AFClassificationDataset(subset="validation")` loaded no recordings: it
  looked beside the `validation/` folder the sample archive unpacks into.
- `UnifiedAFDataset` ignored its `normalization` argument.
- `extract-features` and `explain` built their dataset in its default
  labelling mode instead of the one the checkpoint was trained in.
- CSV recordings without a header row lost their first sample.

### Deprecated
- `--devices` has no effect and will be removed; it now says so when used.

### Removed
- `deepecg resume` (a stub that always failed since introduction) and the
  broken `examples/cli_*.sh` scripts.

## [0.2.0] - earlier

Initial tracked baseline (training, evaluation, 19 architectures, 5 datasets,
CLI, documentation site).
