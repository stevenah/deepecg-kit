# Explainability

A trained classifier gives you a class and a probability, and nothing about which part of the
recording moved it there. Attribution answers that second question: it assigns every sample of every
lead a contribution to one output class. That much is generic. What makes it useful for ECG is what
comes next — collapsing the map into a distribution over the twelve leads, and averaging it over
R-peak-aligned beats so the result can be read against the cardiac cycle. A heatmap over a
5000-sample twelve-lead tensor is not a claim anyone can check. "This model draws 10% of its
attribution from the P-wave region, against the 25% that region's width would earn by chance" is.

```python
from deepecgkit.explainability import ECGExplainer

explainer = ECGExplainer.from_checkpoint("runs/my-run/checkpoints/best-epoch.pt", sampling_rate=500.0)
result = explainer.explain(signals)

result.values            # (samples, leads, time) attribution, the input's own shape
result.signal            # the signal the model saw, same shape
result.lead_importance   # (samples, leads), each row sums to one
result.target            # (samples,) the class each row explains
result.beat_profile      # (leads, window) averaged over detected beats
result.pqrst_fractions   # {"P": 0.10, "QRS": 0.52, "T": 0.36, "baseline": 0.02}
result.pqrst_enrichment  # {"P": 0.41, "QRS": 4.18, "T": 0.96, "baseline": 0.08}
result.crop_offset       # samples dropped from the front to fit the model
```

```bash
deepecg explain --checkpoint runs/<run-directory> -d ptbxl \
    --method integrated-gradients -o explanations/
```

From the command line the data is prepared the way the checkpoint was trained, and the beat
windows are placed at the sampling rate the model ends up seeing, so neither needs stating; see
[`deepecg explain`](cli.md#deepecg-explain) for what `--sampling-rate` means with a recording
file.

## Where the attribution comes from

The gradient is taken with respect to the input, holding the weights fixed. Attribution therefore
comes back in the input's own layout, per lead and per timestep, which is what every summary below
depends on.

```python
inputs = signals.clone().detach().requires_grad_(True)
scores = model(inputs).gather(1, target.view(-1, 1)).squeeze(1)
gradients = torch.autograd.grad(scores.sum(), inputs)[0]
```

Two consequences worth knowing. Attribution runs the model in evaluation mode but **not** under
`torch.no_grad()`, unlike [feature extraction](feature-extraction.md) — it enables gradients itself,
so calling it from inside an evaluation loop that has already entered `no_grad` works. And it uses
`torch.autograd.grad` rather than `backward()`, so no parameter gradient is ever populated and a
live optimiser cannot be corrupted by explaining a model mid-training.

## Choosing a method

| Method | What it answers | Cost per recording |
|---|---|---|
| `saliency` | How much would this logit move if this sample moved slightly? | one forward, one backward |
| `integrated-gradients` | How much of the logit gap against a silent baseline does this sample own? | `ig_steps` forwards and backwards |
| `occlusion` | How much does the logit fall if this window is blanked out? | one forward per window per lead |

Saliency is the default because it is the only one cheap enough to run over a whole split.
Integrated gradients is the one to quote, because it satisfies the completeness axiom: its
attributions sum to the difference between the model's score on the signal and on the baseline, so
each value is a share of a real quantity rather than a relative heat. Occlusion asks a different
question entirely and does not depend on the gradient at all, which makes it a useful second opinion
when a gradient result looks suspicious.

Saliency returns magnitude by default; integrated gradients and occlusion return signed values,
where positive means the sample supported the prediction.

## Which lead drives the prediction

Each lead is scored by its mean absolute attribution and the scores are normalised to sum to one, so
`lead_importance` is a share rather than a magnitude. Magnitude rather than sign, because evidence
against a class is still evidence the lead was read; a signed sum would let a lead with large
opposing contributions cancel to near zero and read as ignored.

```python
from deepecgkit.explainability import plot_lead_importance

plot_lead_importance(result.lead_importance, leads=result.lead_names, save_path="leads.png")
```

Twelve-lead data is labelled with the standard PTB-XL ordering automatically. A single-lead model
returns `[1.0]`, which is correct rather than degenerate — with one lead there is nowhere else for
the evidence to be — but it tells you nothing, so do not report it.

## Where in the beat the model looks

R-peaks are detected with a Pan-Tompkins style energy detector, one window is cut per beat, and the
windows are averaged. Detection runs on the **signal**, never on the attribution: the cardiac cycle
is a property of the recording, so anchoring to the attribution would make the window positions
depend on the method under test.

The default beat window runs from 300 ms before the R-peak to 500 ms after it, 800 ms in all, and
the regions are those a cardiologist would name:

| Region | Window relative to R | Share of the beat window |
|---|---|---|
| P | −250 to −50 ms | 25% |
| QRS | −50 to +50 ms | 12.5% |
| T | +100 to +400 ms | 37.5% |
| baseline | the remainder | 25% |

`pqrst_fractions` reports each region's share of the total attribution mass. Those shares are taken
against the whole beat, not against the three named regions, so they do not sum to one — the
remainder is the isoelectric baseline, and renormalising it away would make a model that attends
mostly to the flat line look like one that attends to the P wave.

`pqrst_enrichment` divides each share by the region's share of the window's duration. One means the
region received exactly the attention its width would earn by chance, above one means the model
concentrates there, below one means it avoids it. This is the number the P-wave claim turns on: a P
share of 0.10 sounds small until it is read against 0.25, at which point it says the model attends to
the P-wave region at roughly 40% of chance.

    !!! note "Attribution explains the model, not the patient"
        A high QRS enrichment means this network's decision depended on QRS samples. It is not
        evidence that the QRS complex is diagnostic, and it is not a clinical finding. Two models
        trained on the same data will give different attributions and both will be faithful to
        themselves.

    !!! warning "Set the sampling rate"
        `sampling_rate` is what places the P, QRS and T windows around each detected R-peak. In
        Python nothing can check it against an array you pass, so giving 500 for a 100 Hz
        recording puts the QRS window where the T wave is and every fraction it reports will be
        wrong. The `deepecg explain` command takes the rate from the checkpoint and the dataset
        instead.

### How well the R-peaks are found

The detector is a filtered-derivative energy detector: a zero-phase 5–15 Hz band-pass, the squared
derivative averaged over 150 ms, and peaks above the mean plus one standard deviation that are at
least 300 ms apart, without adaptive thresholds or search-back. It was checked against the
reference beat annotations of the 48 records of the MIT-BIH Arrhythmia Database, run as attribution
runs it: on z-scored 10-second windows resampled to 500 Hz, on the MLII lead (the first lead for the
two records without one), counting a detection within 150 ms of an annotated beat as a match.
Over 109,150 annotated beats it found 92.4% (sensitivity 0.924) with a positive predictive value
of 0.998; the per-record medians were 0.980 and 0.999. Sensitivity was lowest on records dominated
by paced or ventricular ectopic beats (records 104, 203, 208 and 228), where the fixed threshold and
the 300 ms refractory period miss beats. A detected peak is almost always a real beat, so the beat
profile rests on true beats, but ectopic beats are under-represented in it.

`match_r_peaks(detected, reference, tolerance)` scores detections against annotations the same way,
one to one and nearest pairs first, and returns the true positives, false positives and false
negatives:

```python
from deepecgkit.explainability import detect_r_peaks, match_r_peaks

peaks = detect_r_peaks(signals[0, 0].numpy(), sampling_rate=300.0)
annotated_beats = peaks + 6  # stand-in for reference beat positions, in samples
# 150 ms at 300 Hz is 45 samples
true_positives, false_positives, false_negatives = match_r_peaks(peaks, annotated_beats, tolerance=45)
```

## Model-randomisation check

Attribution methods can produce plausible maps for a model that has learned nothing (Adebayo et
al., 2018). The check is to explain the same inputs, with the same method and the same target
classes, through a model of the same architecture whose weights were never trained. A summary that
changes little under randomisation reflects the input signal and the architecture rather than
what the model learned; one that changes is at least sensitive to the training. Passing the check
does not make an attribution faithful.

```python
untrained = ECGExplainer.from_random_init(
    "kanres", input_channels=1, output_size=4, seed=42, sampling_rate=500.0
)
randomized = untrained.explain(signals)
result.pqrst_enrichment, randomized.pqrst_enrichment
```

`from_random_init` builds the model exactly as the untrained encoder of
[in-domain probing](representation-quality.md#in-domain-probing) is built, under the given seed.
Give it the trained model's lead count and output size, so the same output index can be
explained. From the command line, add `--randomize-weights` to the `deepecg explain` call you
compare against: the checkpoint still decides the data, the preprocessing and the output size,
and `summary.json` records `"weights": "randomized"` and the `randomization_seed`.

!!! note "Architectures whose untrained attribution is degenerate"
    An untrained `xresnet` attributes only through its identity path (see the known issues below),
    so its randomised summaries describe that path rather than a random network.

## Visualisation

```python
from deepecgkit.explainability import plot_attribution, plot_beat_profile

plot_attribution(result.signal[0], result.values[0], sampling_rate=500.0,
                 leads=result.lead_names, save_path="signal.png")
plot_beat_profile(result.beat_profile, result.beat_times_ms, save_path="beats.png")
```

These take `(leads, time)`, matching the model input and the attribution tensor. The older
`plot_ecg_signals` in [Evaluation](evaluation.md) was written for `(time, leads)` and accepts either
layout; the new plots take `(leads, time)` only, the convention used everywhere else in the package.
Omitting `save_path` calls `plt.show()`, which will block in a script.

## Model constraints

The same bounds that govern feature extraction govern attribution, because attribution runs the same
forward pass. One thing differs: a recording above an architecture's ceiling is now centre-cropped
for you and a `RuntimeWarning` says so, rather than failing inside the network. A recording below the
minimum still raises, because cropping cannot make a signal longer.

The authoritative bounds live in the model registry, declared next to each architecture's
`@register_model` decorator, and can be queried as
`deepecgkit.registry.get_model_info(name)["min_input_length"]` /
`["max_input_length"]` (`None` meaning no floor beyond 1, or no ceiling). The table below is a
convenience copy.

| Model | Min samples | Max samples | Centre-cropped above max |
|---|---|---|---|
| `afmodel` | 7971 at `recording_length=30`, 1393 at 6 or 10 | unbounded | n/a |
| `convnext-v2` | 29 | unbounded | n/a |
| `crnn` | 16 | unbounded | n/a |
| `deep-res-cnn` | 127 | unbounded | n/a |
| `dualnet` | 8 | 5000 | yes |
| `fcn-wang` | 2 | unbounded | n/a |
| `gru` | 1 | unbounded | n/a |
| `inception-time` | 1 | unbounded | n/a |
| `kanres` | 1 | unbounded | n/a |
| `kanres-deep` | 11 | unbounded | n/a |
| `lstm` | 1 | unbounded | n/a |
| `mamba` | 50 | 25049 | yes |
| `medformer` | 50 | 5009 | yes |
| `resnet` | 1 | unbounded | n/a |
| `resnet-wang` | 1 | unbounded | n/a |
| `se-resnet` | 1 | unbounded | n/a |
| `simple-cnn` | 8 | unbounded | n/a |
| `tcn` | 1 | unbounded | n/a |
| `transformer` | 1 | 5000 | yes |
| `xresnet` | 17 | unbounded | n/a |
| `xresnet1d-benchmark` | 1 | unbounded | n/a |

After a crop, `result.values` is shorter than the signal you passed and `crop_offset` gives the
number of samples removed from the front, so sample `i` of the attribution is sample
`i + crop_offset` of your recording. The discarded head and tail were not attributed zero; they were
not attributed at all. Pass `time_offset=result.crop_offset / sampling_rate` to `plot_attribution`
to keep the time axis anchored to the original recording.

Every model bakes its lead count into its first layer, so a lead-count mismatch raises before the
forward pass rather than producing a plausible but meaningless result.

## Cost

Measured on `resnet` at twelve leads and 5000 samples, on CPU: one forward and backward pass is
about 74 ms. Saliency costs one of those. Integrated gradients at 64 steps costs about 4.7 s per
recording. Occlusion at `window=100, stride=50` is 1188 forward passes, about 13.5 s. Use
`--max-samples` deliberately; it defaults to 8 for this reason.

`mamba` is far worse than the table suggests, because its selective scan runs a Python loop over the
sequence — roughly 4000 sequential autograd nodes at its maximum length. Integrated gradients on
`mamba` at full length is impractical.

## Known issues

- Gradient saliency measures sensitivity at a single point. It is noisy, it saturates wherever the
  network does, and it reacts to sharp local structure such as the QRS complex whether or not the
  decision depended on it.
- Occlusion attribution bleeds one window width past any sharp boundary in the model's receptive
  field, because a window straddling the boundary perturbs samples on both sides and the single
  resulting score change is spread across all of it. Narrow the window to sharpen the result.
- The zero baseline is a modelling assumption, not a neutral fact. It is the right choice for
  z-scored input, where zero is the per-lead mean, and a poor one for raw millivolts, where blanking
  to zero injects a step the model reacts to.
- R-peak detection is a filtered-derivative threshold detector, not a clinically validated one. On
  the MIT-BIH Arrhythmia Database it finds 92.4% of the annotated beats with a positive predictive
  value of 0.998 ([How well the R-peaks are found](#how-well-the-r-peaks-are-found)), and it misses
  most beats on records dominated by paced or ventricular ectopic beats. On noisy or low-amplitude
  leads it misses beats too, and the profile is then an average over fewer beats than you assume.
  `beat_count` reports how many survived.
- Atrial fibrillation is the case where detection is hardest and the claim matters most: RR
  intervals are irregular and a false peak on a T wave misaligns every window. Check `beat_count`
  against the recording length before quoting a P-wave fraction.
- Fixed windows around each R-peak do not follow heart rate. Ectopic beats, wide QRS complexes and
  marked rate variation will all put real waveform outside the window it belongs to.
- An untrained `xresnet` attributes only through its identity path, because `xresnet1d.py` zero-inits
  the residual BatchNorm weight. Attributions from a randomly initialised model of that architecture
  are degenerate rather than merely uninformative.
- Values are not comparable across methods or across architectures. Never put a saliency map and an
  occlusion map on the same colour scale.
