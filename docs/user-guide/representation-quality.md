# Representation Quality

A model's headline accuracy conflates two things: how good its representation is, and how good the
head fitted on top of it is. Freezing the encoder and fitting nothing but a logistic regression
separates them. Running that probe on a corpus the encoder never saw asks the question the
embeddings exist to answer — whether an architecture learns transferable ECG structure, or a
shortcut peculiar to one dataset.

```python
from deepecgkit.evaluation import cross_dataset_transfer

matrix = cross_dataset_transfer(
    {"af-classification": "runs/af/best.pt", "ptbxl": "runs/ptbxl/best.pt"},
    model_name="kanres",
)
print(matrix.to_frame())
```

```bash
deepecg probe -m kanres -o probe/ \
    --source-checkpoint af-classification=runs/af/best.pt \
    --source-checkpoint ptbxl=runs/ptbxl/best.pt
```

## Standardisation is part of the protocol

Probe features are standardised inside the estimator pipeline, not before it:

```python
from deepecgkit.evaluation import build_probe

build_probe()
# Pipeline([("scaler", StandardScaler()), ("probe", LogisticRegression())])
```

**Results without this step are not comparable and should not be reported.** Embedding scales differ
by orders of magnitude across the registered architectures: most CNNs return non-negative
post-ReLU activations, `convnext-v2` and `mamba` are LayerNorm'd, `crnn`, `gru` and `lstm` are
`tanh`-bounded, and `xresnet` sits near zero at initialisation. Logistic regression applies one L2
penalty to every coefficient, so on unstandardised features that penalty means something different
for each architecture, and the solver may not converge at all on the wide-range ones. The comparison
then measures embedding scale as much as embedding quality.

Keeping the scaler as a pipeline step, rather than standardising up front, is what makes it fit on
the training split alone and merely apply to the held-out split. Standardising the two together
leaks the held-out distribution into the fit. `embed_2d` standardises for the same reason before
projecting.

!!! warning "Do not compare probe scores fitted without the scaler"
    A probe on unstandardised embeddings is not a weaker version of this protocol, it is a
    different measurement. Numbers from the two cannot be placed in the same table.

## The linear probe

`linear_probe` fits on one split's embeddings and reports on another's. It never touches the
encoder's weights.

```python
from deepecgkit import FeatureExtractor
from deepecgkit.evaluation import probe_encoder

extractor = FeatureExtractor.from_checkpoint("runs/af/best.pt", model_name="kanres")
result = probe_encoder(extractor, data_module, target="af-classification")

result.score            # the headline metric
result.metrics          # every metric, keyed by name
result.task             # binary, multiclass or multilabel
result.n_train, result.n_test
```

Four metrics are computed for every probe: `auroc_macro` (the default headline), `accuracy`,
`balanced_accuracy` and `f1_macro`. Macro AUROC leads because it is the only one defined identically
for all three label shapes in this library — binary, multiclass one-vs-rest, and PTB-XL's
multi-label — and because it is threshold-free, so it does not reward a probe for predicting the
majority class.

Multi-label targets are fitted one-vs-rest and scored per label. They are deliberately **not** put
through `calculate_classification_metrics`, which takes the argmax of any two-dimensional
prediction; that is the right reading for a softmax and the wrong one for independent sigmoids.

A degenerate split — one class in the training half, or a label column with no positives — yields
`NaN` for the affected metric and a `note` on the result rather than an exception, so one bad cell
does not abort a whole matrix.

## The cross-dataset transfer matrix

Rows are the corpora the frozen encoders were trained on, columns are the corpora they were probed
on. The diagonal is the in-domain probe; everything off it is transfer.

```python
matrix.to_frame()
#                   af-classification  ptbxl  mitbih-afdb  ltafdb
# random-init                   0.512  0.601        0.498   0.520
# af-classification             0.884  0.702        0.771   0.755
# ptbxl                         0.690  0.912        0.664   0.671
```

The probe head is refitted for every column. It has to be: the label spaces differ — four classes
for PhysioNet 2017, five multi-label diagnostic superclasses for PTB-XL, four for MIT-BIH AFDB and
LTAFDB — so no head can be carried across them. What transfers, or fails to, is the representation.

The protocol does not train anything. It consumes checkpoints that `deepecg train` already produced,
one per source corpus.

### Every corpus is run single-lead

Transfer only means anything if every encoder sees the same input, so the protocol pins all four
corpora to one lead at one sampling rate:

| Dataset | Native leads | Under this protocol |
|---|---|---|
| `af-classification` | 1 | 1, unchanged |
| `mitbih-afdb` | 2 available | 1, its constructor default |
| `ltafdb` | 2 available | 1, its constructor default |
| `ptbxl` | 12 | 1, pinned to lead II and resampled |

Three of the four already yield a single lead at their defaults, so only PTB-XL is constrained, via
`PROTOCOL_DATASET_KWARGS`. Every cell then feeds an identical `1 x 3000` tensor and
`input_channels=1` holds throughout.

!!! note "`afmodel` cannot be probed under this protocol"
    At 300 Hz over 10 seconds every architecture accepts the input except `afmodel`, which needs at
    least 7971 samples. No sampling rate satisfies both `afmodel` and the attention architectures,
    which cap at 5000, so the two cannot share a protocol. See the length table in
    [Feature Extraction](feature-extraction.md).

Splits come from `ECGDataModule`, so its group-aware splitting keeps a subject's records out of both
halves of the probe.

## The random-init baseline

The first row probes an untrained encoder built by `build_model` under the run's seed. It is the
floor for every column, and it is not optional decoration.

A probe score means nothing on its own. Much of what a linear head recovers from a trained
convolutional encoder it also recovers from a random one, because a random convolutional stack is
already a bank of filters and a global pool. A cell at 0.88 next to a floor of 0.51 is evidence; the
same 0.88 next to a floor of 0.86 is not. Reporting probe numbers without this row is the most
common way the measurement gets misread, and it is usually omitted.

```python
matrix.to_frame().loc["random-init"]     # the floor for each target
matrix.to_frame().T                      # transpose for sources across the top
```

Pass `include_random_init=False` (or `--no-random-init`) to drop it, and `seed=` to choose which
random network the floor describes. The same seed reproduces the same floor exactly.

## Embedding visualisation

`embed_2d` standardises, reduces wide embeddings by PCA, then projects with scikit-learn's t-SNE.
`method="pca"` is the cheap alternative. There is no UMAP dependency.

```python
from deepecgkit.evaluation import embed_2d, plot_embedding

features = matrix.embedding_for("af-classification", "ptbxl")
plot_embedding(
    embed_2d(features.features),
    features.labels,
    class_names=matrix.class_names["ptbxl"],
    save_path="embedding.png",
)
```

t-SNE requires a perplexity below the sample count; `embed_2d` clamps it rather than letting
scikit-learn reject a small split. Multi-hot labels are coloured by their first positive label.

Held-out embeddings are kept only when asked for, since keeping all of them costs memory
proportional to the corpus:

```python
matrix = cross_dataset_transfer(checkpoints, "kanres", keep_embeddings="diagonal")
```

`diagonal` retains the in-domain cells together with the random-init row, which is the pairing worth
looking at: the same target seen through a trained encoder and through an untrained one. `all`
retains every cell, `none` is the default.

## Output

`TransferMatrix.save` dispatches on the suffix. The `.json` form keeps every metric of every cell
plus the resolved protocol settings; `.csv` holds the headline table alone.

```python
matrix.save("probe/transfer_matrix.json")
matrix.save("probe/transfer_matrix.csv")
```

For the manuscript, `deepecgkit.analysis.latex.transfer_matrix_table` emits a booktabs table in the
same style as the other result tables. Nothing is bolded: a single probe score carries no interval,
so marking a best cell would assert a comparison the numbers cannot support.

```python
from deepecgkit.analysis.latex import transfer_matrix_table

transfer_matrix_table(matrix.to_frame(), metric=matrix.metric, baseline_row="random-init")
```

## Lead counts that cannot be recovered

`deep-res-cnn`, `gru`, `lstm` and `transformer` do not record their lead count anywhere a state dict
can be read from, so `input_channels` must be passed explicitly. The protocol's default of `1` is
already correct for the single-lead configuration; pass `--input-channels` if you deviate from it.
See [Feature Extraction](feature-extraction.md) for the full table.
