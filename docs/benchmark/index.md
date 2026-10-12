# Benchmark

DeepECG-Kit is demonstrated on a seeded benchmark: every architecture of the benchmark set is
trained on every task, five times, under one shared configuration. This section holds the
material that describes that benchmark and changes as the package develops. It is kept here, and
not in the journal article, so that it can be corrected and extended without a new publication.

| Page | Contents |
|------|----------|
| This page | The grid and the five tasks as the benchmark builds them |
| [Architectures](architectures.md) | Design of every benchmarked architecture |
| [Analysis protocol](protocol.md) | The stages behind every reported result, in order |
| [Runs and known limitations](runs.md) | What a run stores, how grids are launched, and what is known not to work |

Per-architecture results for every recorded metric are published on the
[benchmark website](https://stevenah.github.io/deepecg-kit-bench/), which also carries the
architecture catalogue. The article shows figures only; the values behind them are part of the
deposited results.

## The grid

The benchmark currently trains 24 architectures on five tasks with five seeds each (42 to 46),
600 runs in all. The 24 architectures are a starting point, not a closed selection: architectures
are added to the package continuously, and the grid is extended as they are. The grid is defined
in `experiments.yaml` at the root of the repository, and the stages that turn it into results are
described in the [analysis protocol](protocol.md).

Two registered architectures are outside the grid. `afmodel` is built for 30-second recordings and
needs at least 7971 samples per lead, more than the 5000 samples of a 10-second input at 500 Hz.
`mamba` is not part of the journal benchmark.

## Tasks

The table summarises how the benchmark builds each of the five tasks. The benchmark trains PTB-XL
in its multi-label mode and MIT-BIH AFDB and LTAFDB in their binary mode, and leaves the other two
tasks multi-class. Every recording is resampled to 500 Hz, and every model input (a 10-second
window, or a whole PTB-XL record) is z-scored per lead. These are the benchmark's settings; the
dataset defaults documented in the [Datasets guide](../user-guide/datasets.md) differ.

| | PTB-XL | PhysioNet 2017 | MIT-BIH AFDB | LTAFDB | Unified AF |
|---|---|---|---|---|---|
| Registry name | `ptbxl` | `af-classification` | `mitbih-afdb` | `ltafdb` | `unified-af` |
| Source, version | PTB-XL 1.0.3 | PhysioNet/CinC Challenge 2017 training set 1.0.0 | MIT-BIH Atrial Fibrillation Database 1.0.0 | Long Term AF Database 1.0.0 | Pooled from the three AF datasets |
| Recordings | 21,799 (18,869 patients) | 8,528 (9–61 s) | 25 (about 10 h); 23 loaded | 84 (typically 24–25 h) | PhysioNet 2017 Normal and AF; both long-term datasets |
| Leads, native / used | 12 / 12 | 1 / 1 | 2 (unnamed) / first | 2 (unnamed) / first | 1 / 1 |
| Rate, native / input | 500 / 500 Hz | 300 / 500 Hz | 250 / 500 Hz | 128 / 500 Hz | 128–300 / 500 Hz |
| Model input | 12 leads, 5000 samples | 1 lead, 5000 samples | 1 lead, 5000 samples | 1 lead, 5000 samples | 1 lead, 5000 samples |
| Input unit | 10 s record | 10 s window | 10 s window | 10 s window | 10 s window |
| Classes | NORM, MI, STTC, CD, HYP | Normal, AF, Other, Noisy | AF, non-AF | AF, non-AF | Normal, AF, AFL, J |
| Input label | Superclasses of statements with likelihood at least 50 | Recording label | Majority of per-sample labels | Majority of per-sample labels | Recording label (PhysioNet 2017, without Other and Noisy) or majority of per-sample labels |
| Task, head, loss | Multi-label, 5 logits, BCE with positive weights | Multi-class, 4 logits, cross-entropy | Binary, 1 logit, BCE | Binary, 1 logit, BCE | Multi-class, 4 logits, cross-entropy |
| Precision, recall, F1 | Macro over labels | Macro over classes | AF class | AF class | Macro over classes |
| Split groups | Patients | Recordings | Recordings | Recordings | Source-prefixed recordings |
| Test groups per seed | 10% of patients | 10% of recordings | 2 | 8 if all 84 records load | 10% of recordings |

NORM, normal ECG; MI, myocardial infarction; STTC, ST/T change; CD, conduction disturbance; HYP,
hypertrophy; AF, atrial fibrillation; AFL, atrial flutter; J, junctional rhythm; BCE, binary
cross-entropy with logits. Recording counts are taken from the PhysioNet dataset descriptions and
were not recounted.

In Unified AF, Normal means normal rhythm for PhysioNet 2017 recordings and any rhythm other than
AF, AFL and J for the two long-term datasets, so the meaning of the class depends on the source.

### Splits

Partitions are drawn over the split groups in the table and redrawn for every seed. The validation
partition takes 20% of the groups and the test partition 10%, each rounded down and never fewer
than one group, and the training partition takes the rest. The 23 loaded MIT-BIH AFDB records
therefore always split into 17, 4 and 2 records, and the 84 LTAFDB records, if all load, into 60,
16 and 8. Both test partitions are below the ten independent test units that a flip test requires,
so `deepecg rank` reports every flip test involving either task as untestable.

The realised partition sizes and the identifiers of the test groups are written to
`split_info.json` in every run directory.
