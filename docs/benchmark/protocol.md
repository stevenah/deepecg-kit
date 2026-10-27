# Analysis protocol

This page states the protocol behind every benchmark result: the grid of training runs, the
analyses computed from the completed runs, which of those analyses need no further training,
which need additional runs, and the order in which they are carried out. It stays at the level of
the protocol. The commands that realise each stage are documented in
[CLI Usage](../user-guide/cli.md), so that they can be corrected there without changing the
description given here.

## The grid

The grid trains every architecture of the benchmark set on two tasks, PTB-XL and PhysioNet 2017,
each pairing once with seed 42, under one shared configuration; with the current 24 architectures
that is 48 runs. Every run uses the same optimiser, learning rate, batch size, epoch cap,
early-stopping patience and input sampling rate (500 Hz). Only the labelling mode follows the task:
multi-label for PTB-XL and multi-class for PhysioNet 2017. PTB-XL is split along its recommended
folds and labelled with the rule of the PTB-XL benchmark; PhysioNet 2017 is split by recording with
the seeded 70/20/10 split. Every architecture of a task is therefore evaluated on the same test
partition.

The long-term AF tasks (MIT-BIH AFDB, LTAFDB and Unified AF) are trained from
`experiments-longterm.yaml` for the benchmark website and are not part of the analyses described
here.

## Order of the stages

1. Obtaining the software and the datasets.
2. Readiness checks and provenance capture.
3. The training grid.
4. Aggregation of the per-run metrics, including calibration.
5. The bootstrap of the test recordings.
6. The reliability bins from stored predictions.
7. In-domain linear probing of every encoder beside an untrained one.
8. Attribution summaries, for the trained models and with randomized weights.
9. Validation of the R-peak detector against an annotated database.
10. The figures and the summary of numbers.

Stages 4 to 6 read only what the completed runs store and need neither training nor new inference.
Stages 7 and 8 run inference on the stored checkpoints, and on untrained models of the same
architectures, but train nothing. Stage 9 uses no run at all.

## Installation and data

The software is obtained from the public repository and installed either with its pinned
development environment or as a package. The datasets are downloaded on first use; Unified AF
always reads its three sources where they were downloaded. The only integrity check applied to a
download is the internal checksum test of the archive, so we recommend archiving the downloaded
archives together with their checksums alongside the run directories. Stage 9 also uses the MIT-BIH
Arrhythmia Database (version 1.0.0), which is fetched from PhysioNet's open-data mirror for that
stage only.

## Readiness checks and provenance capture

Two checks precede the grid. First, each architecture–task pairing is built and fed a single batch
at the input shape its task uses (`scripts/preflight_experiments.py`), which catches shape and
head mismatches before any training time is spent. Second, the two architectures that attend over
all 5000 samples of an input are each trained for a single epoch, one on a 12-lead task and one on
a single-lead task, to confirm that they fit in memory and complete an epoch.

The readiness check builds the shapes the command-line interface builds: 12 leads for PTB-XL, one
lead otherwise, and one logit for the binary tasks of the long-term grid. The machine-readable
description of the grid holds exactly the benchmarked pairings.

The provenance behind a result (the commit of the software, the package and library versions, the
operating system and platform, and a fingerprint of the dataset files) is recorded in each run
directory from schema 3 onwards (see [Runs and known limitations](runs.md)). It is also captured
once for the whole grid when the grid is launched and deposited with the results, together with
the checksums of the downloaded archives.

## Training grid

The grid is described once, in machine-readable form (`experiments.yaml`), as the
architecture–task pairings with the labelling mode of each task and the seed. Each run trains in
its own directory and counts as complete when it has written its metrics and recorded a successful
exit. The aggregation stages set aside runs whose recorded exit status is non-zero, incomplete runs
are listed before any aggregation, and an interrupted grid is resumed without repeating the runs
that already completed.

## Aggregation and calibration

The aggregation collects, for every task and architecture, the discrimination metrics (F1,
precision, recall, AUROC, MCC for PhysioNet 2017, and exact-match and label accuracy for PTB-XL)
and the calibration metrics (expected calibration error, its adaptive variant, maximum calibration
error and Brier score, before recalibration, after temperature scaling and, for PTB-XL, after
per-label Platt scaling, together with the fitted temperature). With one run per pairing, each value
is that run's value; the same aggregation reports means and Student-t intervals when a grid has
several seeds. The summary across architectures takes the median and interquartile range over the
architectures. Aggregation is set to refuse a duplicate, so that one task, architecture and seed
occurring in more than one run directory stops the analysis instead of silently keeping the most
recently modified run.

## Bootstrap of the test recordings

The sampling variation of each test set is estimated from the stored predictions
(`deepecg bootstrap`). The test recordings are resampled with replacement 1,000 times, every window
of a drawn recording is kept with it, F1 and AUROC are recomputed on every resample with the
definitions of the stored test metrics, and the 2.5th and 97.5th percentiles form a 95% interval.
For every task, the architecture with the highest F1 is the reference, and the difference of every
other architecture from it is bootstrapped on the same resampled recordings. These intervals
describe the test set; the variation between training runs is not measured, because each pairing
is trained once.

## Representation probing

Every trained encoder is probed on its own task (`deepecg probe --in-domain`). The dataset is
rebuilt from the checkpoint with the preprocessing, leads, labels and split of the training run,
and the rebuilt partition sizes are checked against those recorded. A linear probe (embeddings
standardised with statistics of the training partition alone, L2-regularised logistic regression
with inverse regularisation strength 1, at most 5000 solver iterations) is fitted on the
penultimate-layer embeddings of the training partition and scored as macro AUROC on the test
partition. An untrained encoder of the same architecture, built under seed 42, is probed
identically on the same split. Whether each fit converged is recorded with its result. The probe
fits run with the BLAS and OpenMP libraries limited to four threads, which made them several times
faster beside other jobs.

The cross-dataset transfer matrix (`deepecg probe` without `--in-domain`) remains available in the
package but is not part of this protocol.

## Attribution

For each task, 64 test inputs (whole records for PTB-XL, 10-second windows for PhysioNet 2017) are
drawn once with seed 42, stratified by reference label (for PTB-XL, by the first positive
superclass), from the test partition of the ResNet1D run, which every architecture of the task
shares. The draw is made by a script that rebuilds the partition from the checkpoint and verifies
it against the recorded one. Every architecture is explained on these inputs with three methods
(saliency, integrated gradients with 64 steps, and occlusion with a 100-sample window advanced in
steps of 50 samples), using the checkpoint with the lowest validation loss. The explained class is
fixed rather than taken from the prediction: AF for PhysioNet 2017, and each PTB-XL superclass in
turn, with the summaries averaged over the five.

Every explanation is repeated for an untrained model of the same architecture and output size,
built under seed 42 (`deepecg explain --randomize-weights`), with the same inputs, methods and
classes: the model-randomisation check. The attribution sampling rate equals the training rate,
500 Hz, because the beat windows of the summary are placed in samples, and R-peaks are detected on
lead II for PTB-XL and on the only lead otherwise.

## Validation of the R-peak detector

The detector of the beat summaries is run on the 48 records of the MIT-BIH Arrhythmia Database as
attribution runs it: each record resampled to 500 Hz with the Fourier method, cut into
non-overlapping 10-second windows from the first sample, each window z-scored, on the MLII lead
(the first lead for the two records without one). Detections are matched one to one to the
reference beat annotations within 150 ms. Over 109,150 annotated beats, the sensitivity was 0.924
and the positive predictive value 0.998 (per-record medians 0.980 and 0.999), with the lowest
sensitivity on records dominated by paced or ventricular ectopic beats (104, 203, 208 and 228).

## Model size

Parameter counts are read from each model as the benchmark builds it, at the two input
configurations the benchmark uses: 12 leads with a five-output head, and one lead with a
four-output head (8,739,973 parameters for ResNet1D at the 12-lead configuration).

## Reliability bins and other analyses of stored predictions

The reliability bins are recomputed from the stored probabilities at the same 15 bins as the
tabulated calibration errors. Reliability curves after recalibration are rebuilt from the stored
probabilities, through logits recovered from them, with the recorded temperature or the recorded
per-label scales and biases. Stored probabilities are rounded to six decimals, which saturates the
most confident bin, so the rebuilt curves are checked against the recorded calibration errors.

Two further analyses are computed from what the runs store and published on the
[benchmark website](https://stevenah.github.io/deepecg-kit-bench/) only: the sweep of the fixed
decision threshold and the training-dynamics and stopping summary. The reliability diagrams of
every architecture and task are published there too.

## Repeated runs and ranking

The package also analyses grids in which every pairing is trained with several seeds: means with
Student-t intervals over seeds, ranks within every seed, Kendall's tau between the rankings of
every pair of seeds, paired comparisons with the Nadeau–Bengio correction, and flip tests between
tasks (`deepecg aggregate`, `deepecg rank`). The present benchmark trains each pairing once, so none
of these is part of its protocol.

## Figures

Each figure is drawn by its own plotting code and reads only the outputs named above. No plotting
code writes into a working copy of the repository, and none reuses the framework's own table
output, whose captions and sort order follow the framework and not the article. The plotting code
is part of the replication package deposited with the results.

## Open items

These are recorded when the grid is released and are not yet filled in:

- the release tag of the software and the commit from which the grid was run;
- the archive location of the downloaded datasets or of their checksums;
- the commit that produced each figure.
