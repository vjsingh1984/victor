#!/bin/bash
# Comprehensive test runner script for Victor AI
# Usage: ./scripts/run_tests.sh [profile] [options]
#
# Profiles:
#   unit       - Run unit tests only (fast)
#   integration - Run integration tests only
#   smoke      - Run smoke tests only (fastest)
#   all        - Run all tests (unit + integration + smoke)
#   coverage   - Run tests with coverage report
#   performance - Run performance benchmarks
#   parallel   - Run tests in parallel (requires pytest-xdist)
#
# Options:
#   -v, --verbose     - Enable verbose output
#   -x, --exitfirst   - Exit on first failure
#   -k, --keep-going  - Keep going on failures
#   -f, --failfast    - Exit on first failure (same as -x)
#   -s, --no-capture  - Disable output capture
#   -h, --help        - Show this help message
#
# Examples:
#   ./scripts/run_tests.sh unit
#   ./scripts/run_tests.sh coverage -v
#   ./scripts/run_tests.sh all -x -s
#   ./scripts/run_tests.sh parallel -k "test_coordinator"
#
# Environment variables:
#   VICTOR_TEST_MODE          - Enable test mode (default: true)
#   VICTOR_SKIP_ENV_FILE      - Skip .env file loading (default: 1)
#   VICTOR_COVERAGE_THRESHOLD - Minimum coverage threshold (default: 50)
#   PYTEST_XDIST_AUTO_WORKER_COUNT - Number of parallel workers (default: auto)
#
# Author: Victor AI Team
# License: Apache 2.0

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
VERBOSE=false
EXITFIRST=false
KEEP_GOING=false
NO_CAPTURE=false
COVERAGE_THRESHOLD=${VICTOR_COVERAGE_THRESHOLD:-50}
PYTEST_ARGS=""
PROFILE=""
PARALLEL_WORKERS=${PYTEST_XDIST_AUTO_WORKER_COUNT:-auto}

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$*${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_section() {
    echo ""
    echo -e "${MAGENTA}>>> $*${NC}"
    echo ""
}

show_help() {
    cat << EOF
${CYAN}Victor AI Test Runner${NC}

${GREEN}Usage:${NC}
    $0 [profile] [options]

${GREEN}Profiles:${NC}
    unit         Run unit tests only (fast, isolated)
    integration  Run integration tests only (requires services)
    smoke        Run smoke tests only (fastest sanity checks)
    framework    Run framework capability tests
    coordinators Run coordinator-specific tests
    all          Run all tests (unit + integration + smoke)
    coverage     Run tests with coverage report
    performance  Run performance benchmarks
    parallel     Run tests in parallel (requires pytest-xdist)

${GREEN}Options:${NC}
    -v, --verbose     Enable verbose output
    -x, --exitfirst   Exit on first failure
    -k, --keep-going  Keep going on failures
    -f, --failfast    Exit on first failure (same as -x)
    -s, --no-capture  Disable output capture (show print statements)
    -h, --help        Show this help message

${GREEN}Examples:${NC}
    $0 unit
    $0 coverage -v
    $0 all -x -s
    $0 parallel -k "test_coordinator"

${GREEN}Environment Variables:${NC}
    VICTOR_TEST_MODE               Enable test mode (default: true)
    VICTOR_SKIP_ENV_FILE           Skip .env file loading (default: 1)
    VICTOR_COVERAGE_THRESHOLD      Minimum coverage threshold (default: 50)
    PYTEST_XDIST_AUTO_WORKER_COUNT Number of parallel workers (default: auto)

${GREEN}Makefile Targets:${NC}
    make test          - Run unit tests
    make test-all      - Run all tests including integration
    make test-cov      - Run tests with coverage report
    make benchmark      - Run performance benchmarks

EOF
}

# Parse command line arguments
parse_args() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    PROFILE="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                PYTEST_ARGS="$PYTEST_ARGS -vv"
                shift
                ;;
            -x|--exitfirst|-f|--failfast)
                EXITFIRST=true
                PYTEST_ARGS="$PYTEST_ARGS -x"
                shift
                ;;
            -k|--keep-going)
                KEEP_GOING=true
                PYTEST_ARGS="$PYTEST_ARGS --maxfail=10"
                shift
                ;;
            -s|--no-capture)
                NO_CAPTURE=true
                PYTEST_ARGS="$PYTEST_ARGS -s"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
}

# Check if pytest is installed
check_dependencies() {
    print_section "Checking dependencies"

    if ! command -v pytest &> /dev/null; then
        log_error "pytest is not installed. Install with: pip install pytest"
        exit 1
    fi

    if ! command -v python &> /dev/null; then
        log_error "python is not installed. Install Python 3.10 or later."
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$(python --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
        log_error "Python 3.10+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python $PYTHON_VERSION detected"
    log_success "pytest $(pytest --version | head -n1 | awk '{print $2}') detected"
}

# Set environment variables for testing
setup_environment() {
    print_section "Setting up test environment"

    export VICTOR_TEST_MODE=true
    export VICTOR_SKIP_ENV_FILE=1
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

    log_info "Test mode enabled"
    log_info "Skipping .env file loading"
    log_info "PYTHONPATH updated"
}

# Run unit tests
run_unit_tests() {
    print_header "Running Unit Tests"

    cd "${PROJECT_ROOT}"

    pytest tests/unit \
        -v \
        --tb=short \
        --maxfail=5 \
        -m "not slow" \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    print_header "Running Integration Tests"

    cd "${PROJECT_ROOT}"

    pytest tests/integration \
        -v \
        --tb=short \
        --maxfail=5 \
        -m "not slow" \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Integration tests completed"
}

# Run smoke tests
run_smoke_tests() {
    print_header "Running Smoke Tests"

    cd "${PROJECT_ROOT}"

    pytest tests/smoke \
        -v \
        --tb=short \
        --maxfail=3 \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Smoke tests completed"
}

# Run framework tests
run_framework_tests() {
    print_header "Running Framework Tests"

    cd "${PROJECT_ROOT}"

    pytest tests/unit/framework \
        -v \
        --tb=short \
        --maxfail=5 \
        -m "not slow" \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Framework tests completed"
}

# Run coordinator tests
run_coordinator_tests() {
    print_header "Running Coordinator Tests"

    cd "${PROJECT_ROOT}"

    pytest tests/unit/agent/test_coordinators.py \
        tests/integration/agent/test_coordinator_integration.py \
        -v \
        --tb=short \
        --maxfail=5 \
        -m "not slow" \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Coordinator tests completed"
}

# Run all tests
run_all_tests() {
    print_header "Running All Tests"

    cd "${PROJECT_ROOT}"

    pytest tests/ \
        --ignore=tests/load_test \
        --ignore=tests/benchmark \
        -v \
        --tb=short \
        --maxfail=10 \
        -m "not slow" \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "All tests completed"
}

# Run tests with coverage
run_coverage_tests() {
    print_header "Running Tests with Coverage"

    cd "${PROJECT_ROOT}"

    pytest tests/unit \
        -v \
        --cov=victor \
        --cov-report=term-missing:skip-covered \
        --cov-report=html \
        --cov-report=xml \
        --cov-context=test \
        --cov-fail-under=${COVERAGE_THRESHOLD} \
        --tb=short \
        --maxfail=5 \
        -m "not slow" \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Coverage tests completed"

    # Show coverage summary
    echo ""
    log_info "Coverage report generated:"
    log_info "  Terminal: (see above)"
    log_info "  HTML:     file://${PROJECT_ROOT}/htmlcov/index.html"
    log_info "  XML:      ${PROJECT_ROOT}/coverage.xml"

    # Check coverage threshold
    COVERAGE_PERCENT=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
    if [[ -n "$COVERAGE_PERCENT" ]]; then
        if (( $(echo "$COVERAGE_PERCENT < $COVERAGE_THRESHOLD" | bc -l) )); then
            log_warning "Coverage ${COVERAGE_PERCENT}% is below threshold ${COVERAGE_THRESHOLD}%"
        else
            log_success "Coverage ${COVERAGE_PERCENT}% meets threshold ${COVERAGE_THRESHOLD}%"
        fi
    fi
}

# Run performance benchmarks
run_performance_tests() {
    print_header "Running Performance Benchmarks"

    cd "${PROJECT_ROOT}"

    pytest tests/benchmark/test_performance_baselines.py \
        -v \
        --benchmark-only \
        --benchmark-json=benchmark-results.json \
        --tb=short \
        -m benchmark \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Performance benchmarks completed"

    if [[ -f "${PROJECT_ROOT}/benchmark-results.json" ]]; then
        log_info "Benchmark results saved to: benchmark-results.json"
    fi
}

# Run tests in parallel
run_parallel_tests() {
    print_header "Running Tests in Parallel"

    cd "${PROJECT_ROOT}"

    # Check if pytest-xdist is installed
    if ! python -c "import xdist" 2>/dev/null; then
        log_error "pytest-xdist is not installed. Install with: pip install pytest-xdist"
        exit 1
    fi

    pytest tests/unit \
        -v \
        --tb=short \
        --maxfail=10 \
        -m "not slow" \
        -n ${PARALLEL_WORKERS} \
        --color=yes \
        ${PYTEST_ARGS}

    log_success "Parallel tests completed with ${PARALLEL_WORKERS} workers"
}

# Main execution
main() {
    parse_args "$@"

    print_header "Victor AI Test Runner"
    echo ""
    log_info "Profile: ${PROFILE}"
    log_info "Project Root: ${PROJECT_ROOT}"
    log_info "Python: $(python --version)"

    check_dependencies
    setup_environment

    case "$PROFILE" in
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        smoke)
            run_smoke_tests
            ;;
        framework)
            run_framework_tests
            ;;
        coordinators)
            run_coordinator_tests
            ;;
        all)
            run_all_tests
            ;;
        coverage)
            run_coverage_tests
            ;;
        performance)
            run_performance_tests
            ;;
        parallel)
            run_parallel_tests
            ;;
        *)
            log_error "Unknown profile: $PROFILE"
            echo ""
            show_help
            exit 1
            ;;
    esac

    echo ""
    print_header "Test Run Complete"
    log_success "All tests in '${PROFILE}' profile finished successfully!"
    echo ""
}

# Run main function
main "$@"
