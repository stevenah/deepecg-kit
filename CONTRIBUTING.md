# Contributing to DeepECG-Kit

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync --group dev     # or: make install
```

Non-uv users can use an editable pip install instead:

```bash
pip install -e ".[dev]"   # or: make install-pip
```

Python 3.10+ is required.

## Development workflow

All common tasks go through the Makefile (run bare `make` for the full list):

```bash
make test        # run the test suite
make test-cov    # tests with coverage report
make check       # read-only gate: ruff lint + format check + tests (mirrors CI)
make fix         # auto-fix lint issues and format code
make typecheck   # mypy (advisory)
make docs-serve  # live-reload docs at http://127.0.0.1:8000
make build       # build sdist + wheel
```

Every tool runs through `uv run`, so the pinned toolchain from `uv.lock` is
used regardless of what is on your PATH.

`make check` never modifies files; if it complains about style, run `make fix`.

## Dependencies

`[dependency-groups]` in `pyproject.toml` is the canonical source of dev/docs
dependencies (what `uv sync` and CI install). The `[project.optional-dependencies]`
extras mirror it for pip users — keep the two in sync when editing either.

## Tests

- The suite is offline and uses synthetic fixtures — no dataset downloads.
- `tests/test_state_dict_compat.py` snapshots every registered model's
  state_dict keys. If it fails, your change renamed a module attribute and
  would break loading of existing checkpoints; only regenerate the fixture
  for a deliberate breaking release (instructions in that file's docstring).

## Releasing

```bash
make release-patch    # or release-minor / release-major
```

This runs `make check`, bumps the version in `pyproject.toml`, refreshes
`uv.lock`, commits, tags `vX.Y.Z`, and pushes. The `release.yml` workflow then
creates the GitHub release and publishes to PyPI via trusted publishing
(one-time setup: a PyPI trusted publisher for this repo/workflow and a `pypi`
environment in the GitHub repository settings).

Record notable changes in `CHANGELOG.md` under `[Unreleased]` as you go; the
release commit moves them under the new version heading.
