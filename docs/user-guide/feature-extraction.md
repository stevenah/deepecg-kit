# Feature Extraction

Every model in DeepECG-Kit exposes the representation its classifier head sees. Extracting those
embeddings turns a trained classifier into a general-purpose ECG encoder, useful for clustering,
t-SNE or UMAP visualisation, similarity search, and transfer learning to a task you have few labels
for.

```python
from deepecgkit import FeatureExtractor

extractor = FeatureExtractor.from_checkpoint("runs/best.ckpt", model_name="kanres")
result = extractor.extract(signals)

result.features        # (num_samples, 64) numpy array
result.labels          # aligned labels, when the input carried them
result.feature_dim     # 64
```

```bash
deepecg extract-features --checkpoint runs/best.ckpt -m kanres \
    --dataset af-classification --split test --output features.npz
```

## Where the embedding comes from

Each architecture splits into a feature path and a classifier head. `extract_features` runs the
feature path and stops:

```python
def forward(self, x):
    x = self.extract_features(x)
    return self.classifier(x)
```

The result is always a two-dimensional `(batch, feature_dim)` tensor, obtained by global pooling
over time, so a 10-second and a 30-second recording produce the same width.

Several models place dropout between the features and the head. Because extraction always runs
under `eval()` and `torch.no_grad()`, dropout is the identity, so the returned vector is exactly
what the trained classifier consumes at inference time.

## Input kinds

One entry point accepts everything the rest of the library produces.

```python
extractor.extract(numpy_array)                       # (N, leads, time), (leads, time), or (time,)
extractor.extract(torch_tensor)
extractor.extract("recording.npy")                   # .npy, .csv, or WFDB .dat/.hea
extractor.extract(dataset)                           # any Dataset yielding (signal, label)
extractor.extract(dataloader)                        # an existing DataLoader
extractor.extract(data_module, split="test")         # train, val, test, or all
```

Datasets, loaders and data modules yield `(signal, label)` pairs, so labels come back aligned in
`result.labels`. Arrays and files carry no labels, and `result.labels` is `None`.

!!! note "Row order is stable"
    `ECGDataModule.train_dataloader()` shuffles. Extraction never uses it; it builds its own
    non-shuffling loader over the split, so rows follow dataset order and can be joined to external
    metadata. Passing your own shuffling `DataLoader` warns, because that ordering is not
    reproducible.

## Loading weights

```python
FeatureExtractor.from_checkpoint("runs/best.ckpt", model_name="kanres")
FeatureExtractor.from_pretrained("weights.pt", model_name="kanres")
FeatureExtractor(model, model_name="kanres")
```

`from_checkpoint` accepts checkpoints written by `ECGTrainer.save_checkpoint`, bare state dicts, and
legacy PyTorch Lightning checkpoints keyed on `state_dict`.

!!! warning "No pretrained weights are published yet"
    The entries in `WEIGHTS_REGISTRY` are placeholders and their URLs do not resolve. Requesting
    one raises a `RuntimeError` explaining this. The working sources today are a local weights file
    and a checkpoint you trained. To distribute your own, call `register_weights()` with a URL you
    host.

## Linear probing with scikit-learn

`FeatureExtractor` is a scikit-learn transformer, so a linear probe is a pipeline:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

probe = Pipeline([
    ("ecg", FeatureExtractor.from_checkpoint("runs/best.ckpt", model_name="kanres")),
    ("clf", LogisticRegression(max_iter=1000)),
])
probe.fit(train_signals, train_labels)
probe.score(test_signals, test_labels)
```

`fit` is a no-op, since the encoder is already trained. Note that `sklearn.base.clone`, used by
`cross_val_score` and `GridSearchCV`, deep-copies the wrapped model and so doubles its memory.

## Saving

```python
result.save("features.npz")   # features, labels, model name, feature dim
result.save("features.npy")   # features only
result.save("features.csv")   # columns f0..fN plus label
```

## Model constraints

Verified by running every architecture. Feed a signal outside these bounds and you get a shape
error from inside the network, so check here first.

| Model | `feature_dim` | Min samples | Max samples | Leads recoverable |
|---|---|---|---|---|
| `afmodel` | 196 | 7971 | unbounded | yes |
| `convnext-v2` | 512 | 29 | unbounded | yes |
| `crnn` | 256 | 16 | unbounded | yes |
| `deep-res-cnn` | 128 | 127 | unbounded | **no** |
| `dualnet` | 384 | 8 | 5000 | yes |
| `fcn-wang` | 128 | 2 | unbounded | yes |
| `gru` | 768 | 1 | unbounded | **no** |
| `inception-time` | 128 | 1 | unbounded | yes |
| `kanres` | 64 | 1 | unbounded | yes |
| `kanres-deep` | 32 | 11 | unbounded | yes |
| `lstm` | 256 | 1 | unbounded | **no** |
| `mamba` | 128 | 50 | 25049 | yes |
| `medformer` | 384 | 50 | 5009 | yes |
| `resnet` | 512 | 1 | unbounded | yes |
| `resnet-wang` | 128 | 1 | unbounded | yes |
| `se-resnet` | 512 | 1 | unbounded | yes |
| `simple-cnn` | 256 | 8 | unbounded | yes |
| `tcn` | 256 | 1 | unbounded | yes |
| `transformer` | 128 | 1 | 5000 | **no** |
| `xresnet` | 512 | 17 | unbounded | yes |
| `xresnet1d-benchmark` | 128 | 1 | unbounded | yes |

Where leads are not recoverable from the weights, `from_checkpoint` cannot infer the lead count.
Pass `input_channels=` explicitly, or use `--dataset` on the CLI, which reads it from the data.

`afmodel` is the sharpest edge: its `recording_length` of 6, 10 or 30 selects a kernel size but is
never compared against your data, and `recording_length=6` and `10` build identical networks. It now
raises with the required length rather than failing deep inside the convolution stack.

## Embedding semantics

Values are **not comparable across architectures**. Five regimes appear:

| Regime | Models |
|---|---|
| Non-negative, post-ReLU global average pooling | most convolutional models |
| Normalised inside `extract_features` by a `LayerNorm` | `convnext-v2`, `mamba` |
| Bounded by `tanh`, being a recurrent hidden state | `crnn`, `gru`, `lstm` |
| Unbounded pooled attention output | `dualnet`, `medformer`, `transformer` |
| Near zero at initialisation, from zero-init residuals | `xresnet` |

Clustering or nearest-neighbour search over embeddings from different models will be dominated by
these scale differences. Standardise per model first.

For `gru` and `xresnet1d-benchmark` the head begins with a `BatchNorm1d` over the embedding, so the
trained classifier never saw the raw vector that `extract_features` returns.

## Known issues

- `Medformer` constructs a `cross_channel_attn` block, 5.5% of its parameters, that `forward` never
  invokes. It receives no gradient but is present in every checkpoint, and the model's description
  advertises cross-channel behaviour the embedding does not have.
- `AFModel.get_feature_size` returns the temporal length after the convolution stack, not the
  feature dimension its name suggests, and raises on any multi-lead model. Use `feature_dim`.
