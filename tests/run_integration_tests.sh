#!/bin/bash
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test runner script for integration tests with coordinator support.
#
# This script runs integration tests, automatically handling coordinator-specific tests
# based on the USE_COORDINATOR_ORCHESTRATOR environment variable.
#
# Usage:
#   # Run all integration tests (coordinator tests will be skipped)
#   ./tests/run_integration_tests.sh
#
#   # Run all tests including coordinator tests (will fail if not implemented)
#   USE_COORDINATOR_ORCHESTRATOR=true ./tests/run_integration_tests.sh
#
#   # Run specific test file
#   ./tests/run_integration_tests.sh tests/integration/agent/test_orchestrator_integration.py
#
#   # Run with verbose output
#   ./tests/run_integration_tests.sh -v
#
#   # Run with coverage
#   ./tests/run_integration_tests.sh --cov

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VERBOSE=""
COVERAGE=""
TEST_PATH="tests/integration"
PYTEST_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        --cov|--coverage)
            COVERAGE="--cov=victor --cov-report=term-missing"
            shift
            ;;
        -m|--marker)
            PYTEST_ARGS="$PYTEST_ARGS -m $2"
            shift 2
            ;;
        -k|--keyword)
            PYTEST_ARGS="$PYTEST_ARGS -k $2"
            shift 2
            ;;
        *)
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# Print header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Integration Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if coordinator orchestrator is enabled
if [ "$USE_COORDINATOR_ORCHESTRATOR" = "true" ]; then
    echo -e "${GREEN}✓ Coordinator orchestrator: ENABLED${NC}"
    echo -e "${YELLOW}  Note: Coordinator tests will run (may fail if not implemented)${NC}"
else
    echo -e "${YELLOW}○ Coordinator orchestrator: DISABLED${NC}"
    echo -e "${YELLOW}  Note: Coordinator tests will be skipped${NC}"
    echo -e "${BLUE}  Enable with: USE_COORDINATOR_ORCHESTRATOR=true${NC}"
fi

echo ""

# Check Python environment
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ Error: pytest not found${NC}"
    echo "  Please install pytest: pip install pytest"
    exit 1
fi

echo -e "${GREEN}✓ pytest found: $(pytest --version)${NC}"
echo ""

# Show test path
echo -e "${BLUE}Running tests in:${NC} $TEST_PATH"
echo ""

# Run pytest
echo -e "${BLUE}========================================${NC}"
echo ""

if [ -n "$COVERAGE" ]; then
    echo -e "${BLUE}Running with coverage...${NC}"
    pytest $TEST_PATH $VERBOSE $COVERAGE $PYTEST_ARGS
else
    pytest $TEST_PATH $VERBOSE $PYTEST_ARGS
fi

# Capture exit code
EXIT_CODE=$?

echo ""
echo -e "${BLUE}========================================${NC}"

# Check results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "To see detailed failure information:"
    echo "  pytest $TEST_PATH -v"
    echo ""
    echo "To run failing tests only:"
    echo "  pytest $TEST_PATH --lf"
fi

exit $EXIT_CODE
