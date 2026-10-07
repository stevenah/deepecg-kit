# Installation

## Install from PyPI

```bash
pip install deepecgkit
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add deepecgkit
```

## Development Install

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/stevenah/deepecg-kit.git
cd deepecg-kit
uv sync --group dev
```

Or, without uv:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with testing and linting tools (pytest, ruff).

## Building Documentation Locally

To build and preview the documentation site:

```bash
uv sync --group docs
make docs-serve
```

This starts a local server at `http://127.0.0.1:8000` with live-reloading.

## Where Data Is Stored

Datasets are downloaded on first use into `~/.deepecgkit/datasets`, one folder per dataset.
`deepecg list-datasets` shows each dataset's download size and whether a local copy exists.

To keep them elsewhere — a scratch disk, a shared project folder on a cluster — set
`DEEPECGKIT_HOME`:

```bash
export DEEPECGKIT_HOME=/scratch/$USER/deepecgkit
```

Datasets then live in `$DEEPECGKIT_HOME/datasets` and cached weights in
`$DEEPECGKIT_HOME/weights`. A single dataset can also be pointed at a specific folder with
`--data-dir` (CLI) or `data_dir=` (Python).

Downloads are fetched from PhysioNet's open-data mirror, fall back to physionet.org when the
mirror cannot be reached, and resume from where they stopped if interrupted.

## Hardware

DeepECG-Kit automatically detects available hardware:

| Device | Detection |
|--------|-----------|
| **CUDA GPU** | Used automatically if `torch.cuda.is_available()` |
| **Apple MPS** | Used automatically if `torch.backends.mps.is_available()` |
| **CPU** | Fallback when no GPU is available |

You can override device selection with `--accelerator cpu|gpu|mps` (CLI) or `device="cpu"` (Python API).

## Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- See [pyproject.toml](https://github.com/stevenah/deepecg-kit/blob/main/pyproject.toml) for the full dependency list
