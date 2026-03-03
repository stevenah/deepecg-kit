


.PHONY: help install install-dev test test-cov lint format clean build check version bump-patch bump-minor bump-major release-patch release-minor release-major publish publish-test docs-install docs-serve docs-build docs-clean

.DEFAULT_GOAL := help

PYTHON := python
PACKAGE := deepecgkit

help: 
	@awk 'BEGIN {FS = ":.*?

install: 
	pip install -e .

install-dev: 
	pip install -e ".[dev]"

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=$(PACKAGE) --cov-report=term-missing

lint: 
	ruff check . --fix

format: 
	ruff format .

check: lint format

clean: 
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

build: clean
	$(PYTHON) -m build

publish: build ## Upload to PyPI
	twine upload dist/*

publish-test: build ## Upload to TestPyPI
	twine upload --repository testpypi dist/*

dev: install-dev check test

docs-install:
	uv sync --group docs

docs-serve: docs-install
	uv run mkdocs serve

docs-build: docs-install
	uv run mkdocs build

docs-clean:
	rm -rf site/

version:
	@grep -m1 'version = ' pyproject.toml | cut -d'"' -f2

bump-patch:
	@V=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2) && \
	MAJOR=$$(echo $$V | cut -d. -f1) && \
	MINOR=$$(echo $$V | cut -d. -f2) && \
	PATCH=$$(echo $$V | cut -d. -f3) && \
	NEW="$$MAJOR.$$MINOR.$$((PATCH + 1))" && \
	sed -i.bak "s/version = \"$$V\"/version = \"$$NEW\"/" pyproject.toml && rm pyproject.toml.bak && \
	echo "$$V → $$NEW"

bump-minor:
	@V=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2) && \
	MAJOR=$$(echo $$V | cut -d. -f1) && \
	MINOR=$$(echo $$V | cut -d. -f2) && \
	NEW="$$MAJOR.$$((MINOR + 1)).0" && \
	sed -i.bak "s/version = \"$$V\"/version = \"$$NEW\"/" pyproject.toml && rm pyproject.toml.bak && \
	echo "$$V → $$NEW"

bump-major:
	@V=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2) && \
	MAJOR=$$(echo $$V | cut -d. -f1) && \
	NEW="$$((MAJOR + 1)).0.0" && \
	sed -i.bak "s/version = \"$$V\"/version = \"$$NEW\"/" pyproject.toml && rm pyproject.toml.bak && \
	echo "$$V → $$NEW"

release-patch: check test bump-patch
	@V=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2) && \
	git add pyproject.toml && \
	git commit -m "Release v$$V" && \
	git tag -a "v$$V" -m "Release v$$V" && \
	git push && git push --tags && \
	echo "Released v$$V"

release-minor: check test bump-minor
	@V=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2) && \
	git add pyproject.toml && \
	git commit -m "Release v$$V" && \
	git tag -a "v$$V" -m "Release v$$V" && \
	git push && git push --tags && \
	echo "Released v$$V"

release-major: check test bump-major
	@V=$$(grep -m1 'version = ' pyproject.toml | cut -d'"' -f2) && \
	git add pyproject.toml && \
	git commit -m "Release v$$V" && \
	git tag -a "v$$V" -m "Release v$$V" && \
	git push && git push --tags && \
	echo "Released v$$V"
