# Runs and known limitations

This page describes what a training run stores, how settings are resolved, how grids of runs are
launched, and which behaviours are known to bear on how a reported value should be read. The
analysis commands (`deepecg aggregate`, `deepecg rank`) read nothing but the run directories
described here.

## Environment

The package requires Python 3.10 or later and declares its runtime dependencies with lower version
bounds only. An installation therefore resolves to whatever versions are current, and the exact
versions behind a result are recoverable only from the provenance recorded with it; a lockfile
pins the development environment instead.

## Configuration

Each command wraps a function that is also importable from Python, so the terminal and Python
paths run the same code. Training, evaluation, feature extraction and attribution build their data
modules through one shared routine.

Evaluation, feature extraction and attribution rebuild the dataset in the labelling mode and with
the preprocessing recorded in the checkpoint, so their label-stratified partitions match those of
the training run. Probing builds its own data modules.

Given a checkpoint in a run directory, evaluation adopts the recorded seed and split fractions in
place of its own defaults; a value set explicitly is kept instead, with a warning. It also warns
when the rebuilt partition sizes differ from those recorded for the run. The identities of the
groups themselves are not compared, so a rebuilt partition of the right size need not hold the
same recordings.

Settings may be given directly or read from a configuration file. Settings given directly take
precedence over file values, and both take precedence over the built-in defaults (see
[Configuration files](../user-guide/cli.md#configuration-files)).

## Stored outputs of a run

Every training invocation writes one self-contained run directory. It records:

- the run identity (architecture, dataset and seed), the validation and test split fractions, the
  labelling flags (multi-label and binary classification), the preprocessing in effect (sampling
  rate, normalisation and segment length), the training settings (epochs, batch size and learning
  rate) and a schema version, currently 3;
- the realised split: the seed, the training, validation and test window counts, whether the
  split was grouped, and the number and sorted identifiers of the test groups;
- at most three checkpoints, ranked by validation loss, each storing the model and optimiser
  states together with the training configuration (including the scheduler settings and any
  positive weights) and the model configuration;
- the per-epoch loss and accuracy;
- one row of test discrimination and calibration metrics, which for single-label tasks also
  states whether those metrics were averaged in positive-class or macro mode;
- the true label, the prediction and the probabilities of every test window;
- training-curve, confusion-matrix and reliability plots;
- the package version, the git commit of the checkout and whether it was clean, the host,
  platform, Python and library versions and the resolved device, and the dataset's download URL
  with a fingerprint of the dataset files on disk (their sorted relative paths and sizes).

The early-stopping patience survives only inside the command that the batch runner stores,
together with the exit status and duration it adds.

## Batch runner

Grids are run by `scripts/run_experiments.py`, which reads a YAML experiment file with a defaults
section and a list of experiments, either of which may carry its own seed list. A seed list given
at launch overrides a per-experiment list, which overrides the default list; failing all three,
the single configured seed is used. Seeds must be distinct integers from 0 to 4,294,967,295.

Each pairing of an experiment with a seed runs as a separate training subprocess with its own run
directory. Two experiments that would otherwise collide are held apart by an eight-character
hexadecimal fingerprint (MD5) of their remaining options.

A training invocation marks itself successful only after the test outputs have been written, so a
run that failed carries no such marker; the runner then records the subprocess exit code in its
place, together with the run identity, the fingerprint, the wall-clock duration and the command.
A run counts as complete when its metrics exist and its exit code is zero. Completion is
re-checked on disk when a grid is resumed, and a complete run is overwritten only on explicit
request.

## Known limitations

These behaviours remain in the current code and bear on how a reported value should be read.

- **Calibration bins.** Reported calibration metrics use the bin count recorded with the run (15
  by default), and the per-run plot follows it. Reliability curves recomputed afterwards from the
  stored probabilities are limited by those probabilities being rounded to six decimals.
- **Determinism.** Deterministic cuDNN algorithms are not enabled, so bitwise reproducibility
  across hardware is not guaranteed.
- **Devices.** The device count accepted for training and evaluation has no effect, with a
  warning; training and evaluation always use a single device.
- **Pooled attribution targets.** Explaining every class at once attributes each class in turn
  from the same evaluation-mode model, but a pooled summary of the default, predicted-class
  target can still mix classes.
- **Bimodality gap.** The gap of 0.10 that marks a comparison as bimodal is a global default
  without task-specific calibration.
- **Probe cells.** Each cell of a transfer matrix is a single fit without an interval.

The guides for [explainability](../user-guide/explainability.md#known-issues) and
[feature extraction](../user-guide/feature-extraction.md#known-issues) list further limitations
of those components.

Issues that have been resolved, including those found while the journal article was checked
against the code, are recorded under *Fixed* and *Added* in the
[changelog](https://github.com/stevenah/deepecg-kit/blob/main/CHANGELOG.md).
