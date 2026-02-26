


.PHONY: help install install-dev test test-cov lint format clean build check version bump-patch bump-minor bump-major release-patch release-minor release-major

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
	pytest tests/ -v

test-cov: 
	pytest tests/ -v --cov=$(PACKAGE) --cov-report=term-missing

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

dev: install-dev check test

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
