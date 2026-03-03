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

This installs the package in editable mode along with testing and linting tools (pytest, ruff).

## Building Documentation Locally

To build and preview the documentation site:

```bash
uv sync --group docs
make docs-serve
```

This starts a local server at `http://127.0.0.1:8000` with live-reloading.

## Hardware

DeepECG-Kit automatically detects available hardware:

| Device | Detection |
|--------|-----------|
| **CUDA GPU** | Used automatically if `torch.cuda.is_available()` |
| **Apple MPS** | Used automatically if `torch.backends.mps.is_available()` |
| **CPU** | Fallback when no GPU is available |

You can override device selection with `--accelerator cpu|gpu|mps` (CLI) or `device="cpu"` (Python API).

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- See [pyproject.toml](https://github.com/stevenah/deepecg-kit/blob/main/pyproject.toml) for the full dependency list
