# Victor Makefile
# Common development and distribution commands
#
# Usage:
#   make install      # Install for development
#   make test         # Run tests
#   make build        # Build distribution packages
#   make release      # Create a release (requires version)

.PHONY: help install install-dev test lint format clean build build-binary docker release

# Default target
help:
	@echo "Victor Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install for development"
	@echo "  make install-dev   Install with all dev dependencies"
	@echo "  make test          Run unit tests"
	@echo "  make test-all      Run all tests including integration"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Distribution:"
	@echo "  make build         Build Python packages (sdist + wheel)"
	@echo "  make build-binary  Build standalone binary"
	@echo "  make docker        Build Docker image"
	@echo "  make release       Create a release (VERSION=x.y.z required)"
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
	pre-commit install || true

test:
	pytest tests/unit -v --tb=short

test-all:
	pytest -v --tb=short

test-cov:
	pytest tests/unit --cov=victor --cov-report=html --cov-report=term-missing

lint:
	ruff check victor
	black --check victor
	mypy victor --ignore-missing-imports || true

format:
	black victor tests
	ruff check --fix victor tests

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

# Release (requires VERSION)
release:
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=0.1.0)
endif
	@echo "Creating release v$(VERSION)..."
	@# Update version in pyproject.toml
	sed -i.bak 's/version = ".*"/version = "$(VERSION)"/' pyproject.toml && rm pyproject.toml.bak
	@# Commit and tag
	git add pyproject.toml
	git commit -m "Release v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo ""
	@echo "Release v$(VERSION) created!"
	@echo "Run 'git push && git push --tags' to trigger the release workflow"

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
