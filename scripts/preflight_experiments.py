"""Check every pairing in an experiments config before spending GPU time on it.

A sweep of 500 runs that dies on run 340 because one architecture cannot accept one
dataset's sequence length has wasted most of a day. Every failure this catches is one
that would otherwise surface hours in, and all of them are cheap to find: build the
model at the shape the dataset actually produces and push one batch through it.

It found a real one. AFModel defaults to recording_length=30 and demands at least 7971
samples per lead, while these datasets emit 3000 (300 Hz x 10 s) or 5000 (PTB-XL,
500 Hz x 10 s), so all five of its pairings raise ValueError at build time.

Checked here, deliberately:
  * the model builds at (input_channels, sequence_length) for that dataset,
  * a forward pass returns exactly (batch, output_size).

NOT checked, because none of it is knowable without the data on disk and the training
loop running: whether the dataset downloads, whether the split is sane, whether the run
converges. By default this is a shape and arity check, nothing more.

Checked on request: whether a training step fits in memory. --train-step runs one
forward and backward pass per pairing, which is what an out-of-memory failure hours
into a sweep comes down to. Run it on the machine and device the sweep will use, at the
sweep's batch size. It is meant for a GPU with memory of its own, where a pairing that
does not fit fails at once; on a machine whose GPU shares the system's memory (Apple
Silicon) the attention models will push it into swap instead, so do not run it there.

    python scripts/preflight_experiments.py                    # default experiments.yaml
    python scripts/preflight_experiments.py --config my.yaml
    python scripts/preflight_experiments.py --batch-size 4
    python scripts/preflight_experiments.py --device cuda --batch-size 32 --train-step

Exit status is 1 if any pairing fails, so it can gate a sweep:

    python scripts/preflight_experiments.py && python scripts/run_experiments.py --resume
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import yaml  # type: ignore[import-untyped]

# Sequence length and channel count each dataset actually emits, and the head size each
# task needs. Derived from the dataset classes' own defaults: sampling_rate x
# segment_duration_seconds, with output_size following the flags in the config
# (multi_label / binary_classification) rather than the registry's raw num_classes.
DATASET_SHAPES: dict[str, tuple[int, int]] = {
    "ptbxl": (12, 5000),  # 500 Hz x 10 s
    "af-classification": (1, 3000),  # 300 Hz x 10 s
    "mitbih-afdb": (1, 3000),  # 300 Hz x 10 s
    "ltafdb": (1, 3000),  # 300 Hz x 10 s
    "unified-af": (1, 3000),  # 300 Hz x 10 s
}

# Head size per dataset, keyed by whether the config asked for binary classification.
# `deepecg train` gives a binary run a single-logit head. Only the three AF
# datasets below accept --binary-classification; the CLI rejects it elsewhere.
DATASET_OUTPUTS: dict[str, dict[bool, int]] = {
    "ptbxl": {False: 5, True: 2},
    "af-classification": {False: 4, True: 2},
    "mitbih-afdb": {False: 4, True: 1},
    "ltafdb": {False: 4, True: 1},
    "unified-af": {False: 4, True: 1},
}


def pairings(config_path: Path) -> list[tuple[str, str, bool]]:
    """Read (model, dataset, binary) triples from an experiments config."""
    config = yaml.safe_load(config_path.read_text())
    seen: list[tuple[str, str, bool]] = []
    for exp in config.get("experiments", []) or []:
        model, dataset = exp.get("model"), exp.get("dataset")
        if not model or not dataset:
            continue
        triple = (model, dataset, bool(exp.get("binary_classification", False)))
        if triple not in seen:
            seen.append(triple)
    return seen


def check(
    model: str,
    dataset: str,
    binary: bool,
    batch: int,
    device: str = "cpu",
    train_step: bool = False,
) -> str | None:
    """Return None when the pairing is fine, else a one-line reason."""
    import torch

    from deepecgkit.models import build_model
    from deepecgkit.utils.device import resolve_device

    if dataset not in DATASET_SHAPES:
        return f"unknown dataset (known: {', '.join(sorted(DATASET_SHAPES))})"
    channels, length = DATASET_SHAPES[dataset]
    outputs = DATASET_OUTPUTS[dataset][binary]
    try:
        target = resolve_device(device)
        net = build_model(model, input_channels=channels, output_size=outputs).to(target)
        signals = torch.randn(batch, channels, length, device=target)
        if train_step:
            # Gradients for every weight are held at once, as in training; which
            # loss they come from makes no difference to the memory they take.
            net.train()
            got = net(signals)
            got.sum().backward()
        else:
            net.eval()
            with torch.no_grad():
                got = net(signals)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if tuple(got.shape) != (batch, outputs):
        return f"forward returned {tuple(got.shape)}, expected {(batch, outputs)}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="experiments.yaml", type=Path)
    parser.add_argument(
        "--batch-size", type=int, default=2, help="batch for the probe forward pass (default: 2)"
    )
    parser.add_argument(
        "--device", default="cpu", help="device to run the probe on: cpu, cuda, mps or auto"
    )
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="run a forward and backward pass, to find pairings that do not fit in memory",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 1

    warnings.filterwarnings("ignore")
    todo = pairings(args.config)
    print(f"preflight: {len(todo)} pairings from {args.config}\n")

    failures: list[tuple[str, str, str]] = []
    for model, dataset, binary in todo:
        reason = check(model, dataset, binary, args.batch_size, args.device, args.train_step)
        mark = "ok  " if reason is None else "FAIL"
        suffix = " (binary)" if binary else ""
        print(f"  [{mark}] {model:22} x {dataset}{suffix}")
        if reason is not None:
            failures.append((model, dataset, reason))

    if failures:
        print(f"\n{len(failures)} pairing(s) cannot run:\n", file=sys.stderr)
        for model, dataset, reason in failures:
            print(f"  {model} x {dataset}\n      {reason}\n", file=sys.stderr)
        print(
            "Remove them from the config, or change the segmentation so the sequence "
            "length suits them. Running the sweep as-is will fail on each of them.",
            file=sys.stderr,
        )
        return 1

    print(f"\npreflight: OK\n  all {len(todo)} pairings build and return the right head size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
