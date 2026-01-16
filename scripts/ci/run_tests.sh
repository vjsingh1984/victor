#!/bin/bash
# Run test suite with configurable options
# Usage: scripts/ci/run_tests.sh [unit|integration|all] [--cov] [--verbose]

set -e

# Default values
TEST_TYPE="all"
COVERAGE="--cov=victor --cov-report=term-missing --cov-report=html"
VERBOSE=""
MARKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    unit|integration|all)
      TEST_TYPE="$1"
      shift
      ;;
    --cov)
      COVERAGE="--cov=victor --cov-report=xml --cov-report=html --cov-report=term-missing"
      shift
      ;;
    --no-cov)
      COVERAGE=""
      shift
      ;;
    --verbose|-v)
      VERBOSE="-v"
      shift
      ;;
    --markers|-m)
      MARKERS="-m $2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [unit|integration|all] [--cov] [--no-cov] [--verbose] [--markers MARKERS]"
      echo ""
      echo "Options:"
      echo "  unit|integration|all  Test suite to run (default: all)"
      echo "  --cov                Enable coverage (default)"
      echo "  --no-cov             Disable coverage"
      echo "  --verbose, -v         Verbose output"
      echo "  --markers, -m         Pytest markers (e.g., 'not slow')"
      echo "  --help, -h            Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set test paths
case $TEST_TYPE in
  unit)
    TEST_PATH="tests/unit"
    ;;
  integration)
    TEST_PATH="tests/integration"
    ;;
  all)
    TEST_PATH="tests"
    ;;
esac

# Run tests
echo "Running $TEST_TYPE tests..."
echo "pytest $TEST_PATH $VERBOSE $COVERAGE $MARKERS"

pytest $TEST_PATH $VERBOSE $COVERAGE $MARKERS \
  --tb=short \
  --maxfail=5 \
  --cov-context=test

echo ""
echo "Test suite completed successfully!"
