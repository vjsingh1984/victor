.PHONY: help install test lint format build clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e ".[dev]"

test:  ## Run unit tests
	pytest tests/unit -v

test-all:  ## Run all tests
	pytest tests/ -v

lint:  ## Run linting (ruff + mypy)
	ruff check victor_sdk/
	mypy victor_sdk/

format:  ## Format code (black + ruff --fix)
	black victor_sdk/ tests/
	ruff check --fix victor_sdk/ tests/

build:  ## Build distribution packages
	python -m build

clean:  ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
