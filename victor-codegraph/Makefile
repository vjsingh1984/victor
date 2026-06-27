.PHONY: help install dev test lint format build clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Editable install (Python-only path; no native grammars)
	pip install -e .

dev:  ## Editable install with dev + tree-sitter grammars (local victor-contracts first)
	pip install -e ../victor-contracts || true
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest tests -v

lint:  ## Lint with ruff
	ruff check victor_codegraph tests

format:  ## Auto-fix with ruff
	ruff check --fix victor_codegraph tests

build:  ## Build sdist + wheel
	python -m build

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
