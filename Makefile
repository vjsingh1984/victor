# Victor Makefile
# Common development and distribution commands
#
# Usage:
#   make install      # Install for development
#   make test         # Run tests
#   make build        # Build distribution packages
#   make release      # Create a release (requires version)

.PHONY: help install install-dev test test-definition-boundaries lint check-repo-hygiene format clean build build-binary docker release sync-version check-version

# Default target
help:
	@echo "Victor Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install for development"
	@echo "  make install-dev   Install with all dev dependencies"
	@echo "  make test          Run unit tests"
	@echo "  make test-all      Run all tests including integration"
	@echo "  make test-definition-boundaries  Run SDK-definition import guardrails"
	@echo "  make lint          Run linters"
	@echo "  make check-repo-hygiene  Validate workflow/link/metadata drift guards"
	@echo "  make format        Format code"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Distribution:"
	@echo "  make build         Build Python packages (sdist + wheel)"
	@echo "  make build-binary  Build standalone binary"
	@echo "  make docker        Build Docker image"
	@echo "  make release       Create a release (VERSION=x.y.z required)"
	@echo ""
	@echo "Versioning:"
	@echo "  make check-version Verify victor-ai/victor-sdk versions are in sync"
	@echo "  make sync-version  Sync all package versions from VERSION file"
	@echo ""
	@echo "Utilities:"
	@echo "  make docs          Generate documentation"
	@echo "  make serve         Start API server"
	@echo "  make tui           Start TUI"

# =============================================================================
# Development
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,build]"
	pip install pre-commit pytest-xdist pytest-split
	pre-commit install || true

test:
	pytest tests/unit -v --tb=short -n 2 --dist loadscope

test-parallel:
	pytest tests/unit -v --tb=short -n auto --dist loadscope

test-definition-boundaries:
	pytest tests/unit/core/verticals/test_definition_import_boundaries.py -q

test-all:
	pytest -v --tb=short

test-cov:
	pytest tests/unit --cov=victor --cov-report=html --cov-report=term-missing

test-quick:
	pytest tests/unit -v --tb=short -m "not slow"

test-split:
	pytest tests/unit --splits=4 --group=1 -v --tb=short

lint:
	ruff check victor tests
	black --check victor tests
	mypy victor
	python scripts/ci/repo_hygiene_check.py

check-repo-hygiene:
	python scripts/ci/repo_hygiene_check.py

format:
	black victor tests
	ruff check --fix victor tests

format-check:
	black --check victor tests
	ruff check victor tests

pre-commit:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Distribution
# =============================================================================

build: clean
	pip install build
	python -m build
	@echo ""
	@echo "Built packages:"
	@ls -la dist/

build-binary: clean
	pip install -e ".[build]"
	python scripts/build_binary.py --onefile
	@echo ""
	@echo "Built binary:"
	@ls -la dist/

docker:
	docker build -t victor:latest .
	@echo ""
	@echo "Built Docker image: victor:latest"

docker-push: docker
	docker tag victor:latest vijayksingh/victor:latest
	docker push vijayksingh/victor:latest

# Version management
sync-version:  ## Sync all package versions from VERSION files
	python scripts/sync_version.py

sync-version-ai:  ## Sync victor-ai version only
	python scripts/sync_version.py --ai

sync-version-sdk:  ## Sync victor-sdk version only
	python scripts/sync_version.py --sdk

check-version:  ## Verify all package versions are consistent
	python scripts/check_version_sync.py

# Release victor-ai (requires VERSION)
release:
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=0.1.0)
endif
	@echo "Creating victor-ai release v$(VERSION)..."
	echo "$(VERSION)" > VERSION
	python scripts/sync_version.py --ai
	python scripts/check_version_sync.py
	git add VERSION pyproject.toml
	git commit -m "release: victor-ai v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release victor-ai v$(VERSION)"
	@echo ""
	@echo "Release v$(VERSION) created!"
	@echo "Run 'git push && git push --tags' to trigger the release workflow"

# Release victor-sdk independently (requires VERSION)
release-sdk:
ifndef VERSION
	$(error VERSION is required. Usage: make release-sdk VERSION=0.7.0)
endif
	@echo "Creating victor-sdk release sdk-v$(VERSION)..."
	echo "$(VERSION)" > victor-sdk/VERSION
	python scripts/sync_version.py --sdk
	python scripts/check_version_sync.py
	git add victor-sdk/VERSION victor-sdk/pyproject.toml pyproject.toml
	git commit -m "release: victor-sdk v$(VERSION)"
	git tag -a "sdk-v$(VERSION)" -m "Release victor-sdk v$(VERSION)"
	@echo ""
	@echo "SDK release sdk-v$(VERSION) created!"
	@echo "Run 'git push && git push --tags' to trigger the SDK release workflow"

# =============================================================================
# Utilities
# =============================================================================

docs:
	pip install -e ".[docs]"
	mkdocs build

docs-serve:
	pip install -e ".[docs]"
	mkdocs serve

serve:
	victor serve

tui:
	victor

# Check if all distribution files are ready
check-dist:
	@echo "Checking distribution readiness..."
	@test -f pyproject.toml && echo "✓ pyproject.toml" || echo "✗ pyproject.toml missing"
	@test -f MANIFEST.in && echo "✓ MANIFEST.in" || echo "✗ MANIFEST.in missing"
	@test -f requirements.txt && echo "✓ requirements.txt" || echo "✗ requirements.txt missing"
	@test -f README.md && echo "✓ README.md" || echo "✗ README.md missing"
	@test -f LICENSE && echo "✓ LICENSE" || echo "✗ LICENSE missing"
	@test -f Dockerfile && echo "✓ Dockerfile" || echo "✗ Dockerfile missing"
	@test -f .github/workflows/release.yml && echo "✓ release.yml" || echo "✗ release.yml missing"
	@test -f scripts/install/install.sh && echo "✓ install.sh" || echo "✗ install.sh missing"
	@test -f scripts/install/install.ps1 && echo "✓ install.ps1" || echo "✗ install.ps1 missing"
	@test -f Formula/victor.rb && echo "✓ Homebrew formula" || echo "✗ Homebrew formula missing"
	@echo ""
	@echo "Run 'make build' to test the build process"
