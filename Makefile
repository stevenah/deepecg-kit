.PHONY: help install install-pip test test-cov lint typecheck check fix dev clean clean-all \
	build version bump-patch bump-minor bump-major release-patch release-minor release-major \
	docs-serve docs-build docs-clean

.DEFAULT_GOAL := help

PACKAGE := deepecgkit

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_ -]+:.*?## / {sub(/ .*/, "", $$1); printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package + dev tools (uv)
	uv sync --group dev

install-pip: ## Editable install for non-uv users
	pip install -e ".[dev]"

test: ## Run test suite
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov --cov-report=term-missing

lint: ## Ruff lint + format check (read-only)
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## mypy (advisory until annotations land)
	uv run mypy $(PACKAGE)

check: lint test ## Read-only gate, mirrors CI

fix: ## Auto-fix lint issues and format code
	uv run ruff check . --fix
	uv run ruff format .

dev: install fix test ## Setup + fix + test

clean: ## Remove build artifacts and caches (skips .venv)
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage .coverage.* htmlcov/ .mypy_cache/ .ruff_cache/
	find . -name .venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
	find . -name .venv -prune -o -name "*.pyc" -type f -print0 | xargs -0 rm -f

clean-all: clean docs-clean ## Also remove stale Lightning-era checkpoints/ (not runs/ or data)
	rm -rf checkpoints/

build: ## Build sdist + wheel
	uv build

version: ## Print current version
	@grep -m1 '^version = ' pyproject.toml | cut -d'"' -f2

bump-patch bump-minor bump-major: bump-%: ## Bump version and refresh lockfile
	@V=$$(grep -m1 '^version = ' pyproject.toml | cut -d'"' -f2) && \
	MAJOR=$$(echo $$V | cut -d. -f1) && MINOR=$$(echo $$V | cut -d. -f2) && PATCH=$$(echo $$V | cut -d. -f3) && \
	case "$*" in \
	  patch) NEW="$$MAJOR.$$MINOR.$$((PATCH + 1))";; \
	  minor) NEW="$$MAJOR.$$((MINOR + 1)).0";; \
	  major) NEW="$$((MAJOR + 1)).0.0";; \
	esac && \
	sed -i.bak "s/^version = \"$$V\"/version = \"$$NEW\"/" pyproject.toml && rm pyproject.toml.bak && \
	uv lock && echo "$$V -> $$NEW"

release-patch release-minor release-major: release-%: check bump-% ## Tag and push a release (CI publishes)
	@V=$$(grep -m1 '^version = ' pyproject.toml | cut -d'"' -f2) && \
	git add pyproject.toml uv.lock && \
	git commit -m "Release v$$V" && \
	git tag -a "v$$V" -m "Release v$$V" && \
	git push && git push --tags && \
	echo "Released v$$V — release workflow publishes the GitHub release and PyPI package"

docs-serve: ## Serve docs locally with live reload
	uv run --group docs mkdocs serve

docs-build: ## Build docs (strict)
	uv run --group docs mkdocs build --strict

docs-clean: ## Remove built docs
	rm -rf site/
