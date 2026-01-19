# Victor Makefile
# Common development and distribution commands
#
# Usage:
#   make install      # Install for development
#   make test         # Run tests
#   make build        # Build distribution packages
#   make release      # Create a release (requires version)

.PHONY: help install install-dev test lint format clean build build-binary docker release security security-full load-test load-test-quick load-test-report benchmark dev-tools check-protocol lint-vertical validate-config profile-coordinators coverage-report generate-docs qa qa-fast qa-report release-checklist release-validate

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
	@echo "  make security     Run quick security scans"
	@echo "  make security-full Run comprehensive security scans"
	@echo "  make benchmark     Run performance benchmarks"
	@echo "  make load-test     Run load tests with Locust"
	@echo "  make load-test-quick Run quick load tests"
	@echo "  make load-test-report Generate load test report"
	@echo ""
	@echo "Developer Tools (Phase 4):"
	@echo "  make dev-tools     Run all developer tools"
	@echo "  make check-protocol Check protocol conformance"
	@echo "  make lint-vertical Lint verticals"
	@echo "  make validate-config Validate YAML configs"
	@echo "  make profile-coordinators Profile coordinator performance"
	@echo "  make coverage-report Generate coverage reports"
	@echo "  make generate-docs Generate documentation"
	@echo ""
	@echo "Quality Assurance (Phase 5):"
	@echo "  make qa           Run comprehensive QA suite"
	@echo "  make qa-fast      Run quick QA validation (skip slow tests)"
	@echo "  make qa-report    Generate QA report in JSON format"
	@echo "  make release-checklist Show release checklist"
	@echo "  make release-validate Validate release readiness"

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

# =============================================================================
# Security
# =============================================================================

security:
	@echo "Running quick security scans..."
	@echo "================================"
	@echo ""
	@echo "Bandit (High/Critical severity only)..."
	bandit -r victor/ --exclude victor/test/,victor/tests/,tests/,archive/ -lll || true
	@echo ""
	@echo "Safety check..."
	safety check || true
	@echo ""
	@echo "Semgrep (ERROR severity only)..."
	semgrep --config=auto --exclude=tests/ --exclude=archive/ --severity ERROR || true
	@echo ""
	@echo "Pip Audit..."
	pip-audit || true
	@echo ""
	@echo "Quick security scans complete!"

security-full:
	@echo "Running comprehensive security scans..."
	@echo "======================================="
	@echo ""
	@echo "Creating reports directory..."
	mkdir -p security-reports
	@echo ""
	@echo "Bandit (all findings)..."
	bandit -r victor/ --exclude victor/test/,victor/tests/,tests/,archive/ -f json -o security-reports/bandit-report.json || true
	bandit -r victor/ --exclude victor/test/,victor/tests/,tests/,archive/ -f txt -o security-reports/bandit-report.txt || true
	@echo ""
	@echo "Safety check..."
	safety check --save-json security-reports/safety-report.json || true
	safety check --output text > security-reports/safety-report.txt || true
	@echo ""
	@echo "Semgrep..."
	semgrep --config=auto --exclude=tests/ --exclude=archive/ --json --output=security-reports/semgrep-report.json || true
	semgrep --config=auto --exclude=tests/ --exclude=archive/ > security-reports/semgrep-report.txt || true
	@echo ""
	@echo "Pip Audit..."
	pip-audit --format json --output security-reports/pip-audit-report.json || true
	pip-audit > security-reports/pip-audit-report.txt || true
	@echo ""
	@echo "Comprehensive security scans complete!"
	@echo "Reports saved to security-reports/"

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

# =============================================================================
# Performance & Load Testing
# =============================================================================

benchmark:
	@echo "Running performance benchmarks..."
	pytest tests/benchmark/test_performance_baselines.py -v -m benchmark --tb=short

benchmark-all:
	@echo "Running all performance benchmarks..."
	pytest tests/benchmark/ -v -m benchmark --tb=short

load-test-quick:
	@echo "Running quick load tests (pytest-based)..."
	pytest tests/load_test/test_scalability.py::TestConcurrentRequests::test_10_concurrent_requests -v -s --tb=short

load-test:
	@echo "Running load tests with Locust..."
	@echo "Make sure the Victor API server is running on http://localhost:8000"
	@echo "Start it with: victor serve"
	@echo ""
	@if ! command -v locust >/dev/null 2>&1; then \
		echo "Locust not installed. Installing..."; \
		pip install locust>=2.0; \
	fi
	@echo "Starting Locust web interface on http://localhost:8089"
	locust -f tests/load_test/load_test_framework.py --host=http://localhost:8000

load-test-headless:
	@echo "Running headless load test..."
	@if ! command -v locust >/dev/null 2>&1; then \
		echo "Locust not installed. Installing..."; \
		pip install locust>=2.0; \
	fi
	locust -f tests/load_test/load_test_framework.py \
		--host=http://localhost:8000 \
		--headless \
		--users=100 \
		--spawn-rate=10 \
		--run-time=5m \
		--html=load-test-report.html

load-test-report:
	@echo "Generating load test report..."
	@mkdir -p load-test-reports
	@echo "Running load test and generating HTML report..."
	locust -f tests/load_test/load_test_framework.py \
		--host=http://localhost:8000 \
		--headless \
		--users=50 \
		--spawn-rate=5 \
		--run-time=2m \
		--html=load-test-reports/report_$(shell date +%Y%m%d_%H%M%S).html
	@echo "Report saved to load-test-reports/"

scalability-test:
	@echo "Running comprehensive scalability tests..."
	pytest tests/load_test/test_scalability.py -v -s -m "slow" --tb=short

scalability-report:
	@echo "Generating scalability report..."
	@mkdir -p /tmp/scalability-reports
	pytest tests/load_test/test_scalability.py -v --tb=short - scalability-reports-dir=/tmp/scalability-reports
	@echo "Scalability reports saved to /tmp/scalability-reports/"

performance-regression-check:
	@echo "Checking for performance regressions..."
	pytest tests/benchmark/test_performance_baselines.py::TestPerformanceRegression -v --tb=short

	@echo "Run 'make build' to test the build process"

# =============================================================================
# Developer Tools (Phase 4)
# =============================================================================

dev-tools: check-protocol lint-vertical validate-config
	@echo ""
	@echo "All developer tools completed!"
	@echo "Run individual tools for more details:"
	@echo "  make check-protocol"
	@echo "  make lint-vertical"
	@echo "  make validate-config"

check-protocol:
	@echo "Checking protocol conformance..."
	python scripts/check_protocol_conformance.py --all-verticals
	@echo "Protocol conformance check complete!"

lint-vertical:
	@echo "Linting verticals..."
	python scripts/lint_vertical.py --all-verticals
	@echo "Vertical linting complete!"

validate-config:
	@echo "Validating configuration files..."
	python scripts/validate_config.py --all-configs
	@echo "Configuration validation complete!"

profile-coordinators:
	@echo "Profiling coordinators..."
	python scripts/profile_coordinators.py --all-coordinators --json-output profiling_results.json
	@echo "Coordinator profiling complete!"
	@echo "Results saved to profiling_results/"

coverage-report:
	@echo "Generating coverage reports..."
	python scripts/coverage_report.py --format html --check-goals
	@echo "Coverage report complete!"
	@echo "HTML report: htmlcov/index.html"

generate-docs:
	@echo "Generating documentation..."
	python scripts/generate_docs.py --all --output docs/generated
	@echo "Documentation generation complete!"
	@echo "Generated docs: docs/generated/"

# =============================================================================
# Quality Assurance (Phase 5)
# =============================================================================

qa:
	@echo "Running comprehensive QA validation..."
	@echo "======================================="
	python scripts/run_full_qa.py --coverage --report text
	@echo ""
	@echo "QA validation complete!"
	@echo "Run 'make qa-report' for detailed JSON report"

qa-fast:
	@echo "Running quick QA validation..."
	@echo "=============================="
	python scripts/run_full_qa.py --fast --report text
	@echo ""
	@echo "Quick QA validation complete!"

qa-report:
	@echo "Generating detailed QA report..."
	@echo "================================="
	python scripts/run_full_qa.py --fast --report json --output qa_report.json
	@echo "QA report generated: qa_report.json"
	@echo ""
	@echo "Summary:"
	@python -c "import json; data = json.load(open('qa_report.json')); s = data['summary']; print(f\"  Total: {s['total_checks']}, Passed: {s['passed']}, Failed: {s['failed']}, Success: {s['success_rate']:.1f}%\")" || true

release-checklist:
	@echo "Victor AI Release Checklist"
	@echo "==========================="
	@echo ""
	@cat RELEASE_CHECKLIST.md

release-validate:
	@echo "Validating release readiness..."
	@echo "================================"
	@echo ""
	@test -f RELEASE_CHECKLIST.md && echo "✓ RELEASE_CHECKLIST.md exists" || echo "✗ RELEASE_CHECKLIST.md missing"
	@test -f CHANGELOG.md && echo "✓ CHANGELOG.md exists" || echo "✗ CHANGELOG.md missing"
	@test -f docs/RELEASE_NOTES_0.5.1.md && echo "✓ Release notes exist" || echo "✗ Release notes missing"
	@test -f README.md && echo "✓ README.md exists" || echo "✗ README.md missing"
	@test -f LICENSE && echo "✓ LICENSE exists" || echo "✗ LICENSE missing"
	@echo ""
	@echo "Running QA validation..."
	python scripts/run_full_qa.py --fast --report json --output release_qa_report.json
	@echo ""
	@echo "Release validation complete!"
	@echo "Review release_qa_report.json for details"
