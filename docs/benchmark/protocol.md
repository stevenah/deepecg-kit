# Analysis protocol

This page states the protocol behind every benchmark result: the grid of training runs, the
analyses computed from the completed runs, which of those analyses need no further training,
which need additional runs, and the order in which they are carried out. It stays at the level of
the protocol. The commands that realise each stage are documented in
[CLI Usage](../user-guide/cli.md), so that they can be corrected there without changing the
description given here.

## The grid

The grid trains every architecture of the benchmark set on five tasks with five repetitions of
each pairing under one shared configuration; with the current 24 architectures that is 120
pairings and 600 runs. Every run uses the same optimiser, learning rate, batch size, epoch cap,
early-stopping patience, split fractions and input sampling rate. Only the labelling mode follows
the task: multi-label for PTB-XL, binary for the two long-term datasets and multi-class otherwise.
Repetitions differ only in the random seed, which redraws the grouped partition, the
initialisation and the data order; the seeds are 42 to 46 throughout.

A planning pass over the grid, made without training, enumerates exactly these runs, and the
aggregation and ranking stages were exercised on synthetic run directories to confirm the outputs
they produce.

## Order of the stages

1. Obtaining the software and the datasets.
2. Readiness checks and provenance capture.
3. The seeded training grid.
4. Aggregation across repetitions, which gives the benchmark scores and their intervals.
5. The calibration aggregation and its summary across architectures.
6. Rank stability, the flip tests and the claim re-test.
7. Linear probing, which gives the transfer matrices.
8. Attribution summaries.
9. The post hoc analyses from stored predictions.
10. The zero-shot cross-dataset evaluation.
11. The figures.

Stages 4 to 6, 8 and 9 reuse the completed runs and need neither further training nor new
inference. Two stages add runs beyond the grid: the single-lead PTB-XL encoders that the transfer
matrices of stage 7 need as sources, which are trained, and the cross-dataset evaluations of
stage 10, which train nothing but score existing checkpoints on the other dataset.

## Installation and data

The software is obtained from the public repository and installed either with its pinned
development environment or as a package. The datasets are downloaded on first use; Unified AF
always reads its three sources where they were downloaded. The only integrity check applied to a
download is the internal checksum test of the archive, so we recommend archiving the downloaded
archives together with their checksums alongside the run directories.

## Readiness checks and provenance capture

Two checks precede the grid. First, each architecture–task pairing is built and fed a single batch
at the input shape its task uses (`scripts/preflight_experiments.py`), which catches shape and
head mismatches before any training time is spent. Second, the two architectures that attend over
all 5000 samples of an input are each trained for a single epoch, one on a 12-lead task and one on
a single-lead task, to confirm that they fit in memory and complete an epoch.

The readiness check builds the shapes the command-line interface builds: 12 leads for PTB-XL, one
lead otherwise, and one logit for the binary tasks. The machine-readable description of the grid
holds exactly the benchmarked pairings.

The provenance behind a result (the commit of the software, the package and library versions, the
operating system and platform, and a fingerprint of the dataset files) is recorded in each run
directory from schema 3 onwards (see [Runs and known limitations](runs.md)). It is also captured
once for the whole grid when the grid is launched and deposited with the results, together with
the checksums of the downloaded archives.

## Seeded training grid

The grid is described once, in machine-readable form (`experiments.yaml`), as the
architecture–task pairings with the labelling mode of each task and the five seeds. The pairings
are also enumerated explicitly when the grid is launched, as a safeguard against running pairings
that are not part of the benchmark. Each run trains in its own directory and counts as complete
when it has written its metrics and recorded a successful exit. The aggregation stages set aside
runs whose recorded exit status is non-zero, incomplete runs are listed before any aggregation,
and an interrupted grid is resumed without repeating the runs that already completed.

## Aggregation and calibration

Two aggregations are computed over the completed runs: one over the discrimination metrics (F1,
precision, recall, AUROC, MCC, accuracy, and exact-match and label accuracy for PTB-XL) and one
over the calibration metrics (expected calibration error, its adaptive variant, maximum
calibration error, Brier score, their counterparts after temperature scaling, and the fitted
temperature). Each yields, for every task, architecture and metric, the mean over the
contributing repetitions, its standard deviation, the half-width of its two-sided 95% interval
and the number of repetitions that contributed.

The summary across architectures takes the median and interquartile range, over the
architectures, of each architecture's mean over repetitions. Aggregation is set to refuse a
duplicate, so that one task, architecture and repetition occurring in more than one run directory
stops the analysis instead of silently keeping the most recently modified run.

## Ranking stability and flip tests

The ranking stage runs once over the completed runs across all benchmarked architectures, with
test F1, the Nadeau–Bengio correction, a 95% level and the requirement of at least ten
independent test units for a comparison to be testable. No claim was specified in advance; every
comparison is exploratory.

## Representation probing

For each architecture a transfer matrix is computed. For every pair of a source and a target
dataset, a linear probe is fitted on penultimate-layer embeddings of the target's training split,
extracted by a frozen encoder trained on the source dataset, and scored as macro AUROC on the
target's test split. The row of untrained, seed-fixed encoders gives each column its floor.

Every dataset is presented as a single lead at one sampling rate, which must equal the rate at
which the source encoders were trained, because a different rate rescales time in their input;
PTB-XL is restricted to lead II. The probe settings (inverse regularisation strength 1, at most
1000 iterations, embeddings standardised with statistics from the target's training split alone)
and the split fractions are the same for every cell, and each cell is a single fit without an
interval. Each matrix also yields a heat map and embedding plots for the in-domain and
random-initialisation cells. The PTB-XL source row uses the lead-II encoders described next; the
other source rows use the seed-42 benchmark checkpoints.

### Single-lead PTB-XL encoders

Because the probe protocol presents every dataset as a single lead, the 12-lead PTB-XL
checkpoints of the grid cannot serve as sources, and the command-line path offers no way to
restrict PTB-XL to one lead. One additional encoder per architecture is therefore trained on lead
II of PTB-XL through the Python interface, at the benchmark's input sampling rate (500 Hz) and
seed and otherwise under the grid's shared configuration. These runs are a separate sub-grid, not
part of the seeded grid, and are reported as such. The probe is run at the same rate.

## Attribution

Attribution is computed from the checkpoint with the lowest validation loss of selected runs, for
three methods (saliency, integrated gradients with 64 steps, and occlusion with a 100-sample
window advanced in steps of 50 samples), on inputs of the test split (whole records for PTB-XL,
10-second windows otherwise). The sampling rate used for attribution must equal the rate the
model was trained at, 500 Hz for every task of the grid, because the beat windows of the summary
are placed in samples. R-peaks are detected on lead II for PTB-XL and on the only lead otherwise.

This stage rebuilds the dataset in the checkpoint's labelling mode, and the explained inputs are
drawn from the run's test partition by a script that verifies the rebuilt partition against the
recorded one.

## Model size

Parameter counts are read from each model as the benchmark builds it, at the two input
configurations the benchmark uses: 12 leads with a five-output head, and one lead with a
four-output head (8,739,973 parameters for ResNet1D at the 12-lead configuration).

## Post hoc analyses from stored predictions

Four analyses are computed from what the completed runs already store (the per-run predictions
with their probabilities, the per-run metrics and the per-epoch training logs) and need neither
training nor new inference:

- the sweep of the fixed decision threshold;
- the training-dynamics and stopping summary;
- the resampling over subsets of the five repetitions;
- the reliability bins, recomputed at the same 15 bins as the tabulated calibration errors.

Stored probabilities are rounded to six decimals, which saturates the most confident bin, so no
analysis reconstructs logits from them, and reliability curves after temperature scaling are not
drawn from the stored files.

The article reports none of these four analyses as figures of its own. The reliability diagrams
for every architecture and task, and the other three analyses, are published with the results on
the [benchmark website](https://stevenah.github.io/deepecg-kit-bench/).

## Zero-shot cross-dataset evaluation

Every grid checkpoint trained on one of the two long-term datasets is scored on the other
dataset's test partition, in both directions and for all five repetitions. This adds evaluation
runs but trains nothing. The two datasets share a binary label space, one lead and the input
sampling rate, so no label remapping is needed, and the temperature fit on the target dataset is
skipped, which keeps the evaluation zero-shot. The resulting outputs carry no run metadata and are
therefore read directly and not through the aggregation stage. A split-size mismatch is reported
whenever a checkpoint meets the other dataset; it is expected and not a failure.

Every other pairing is excluded: PhysioNet 2017 and Unified AF have four classes but not the same
four, and Unified AF pools both long-term datasets. A drop in score is a result about dataset
shift, not about model quality.

## Figures

Each figure is drawn by its own plotting code and reads only the outputs named above. No plotting
code writes into a working copy of the repository, and none reuses the framework's own table
output, whose captions and sort order follow the framework and not the article. The plotting code
is part of the replication package deposited with the results.

## Open items

These are recorded when the grid is run and are not yet filled in:

- the hardware used for the grid (GPU model, driver, CUDA and PyTorch versions, operating system);
- the release version of the software and the commit from which the grid was launched;
- the archive location of the downloaded datasets or of their checksums;
- the archived copy of the grid description, with confirmation that it holds exactly the
  benchmarked pairings;
- the probe sampling rate and the training rate and seed of the lead-II sub-grid, as run;
- the architectures, seeds and number of inputs explained in the attribution stage;
- confirmation that the zero-shot stage rebuilds the target dataset in its binary labelling mode,
  which it does not inherit from the training run, and that no recording occurs in both long-term
  datasets;
- the commit that produced each figure.
