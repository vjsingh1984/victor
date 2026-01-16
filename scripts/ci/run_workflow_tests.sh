#!/usr/bin/env bash
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

#
# CI script to run all workflow-related tests
#
# This script runs comprehensive workflow tests including:
# 1. Workflow YAML validation tests
# 2. Team node tests
# 3. Compiler executor integration tests
# 4. Workflow execution tests
# 5. Recursion limit tests
# 6. Coverage reporting
#
# Usage:
#   ./scripts/ci/run_workflow_tests.sh
#   ./scripts/ci/run_workflow_tests.sh --verbose
#   ./scripts/ci/run_workflow_tests.sh --quick
#
# Environment variables:
#   VICTOR_TEST_VERBOSE=1  - Enable verbose output
#   VICTOR_QUICK_TESTS=1   - Run only quick tests (skip slow ones)
#

set -eou pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Test configuration
VERBOSE="${VICTOR_TEST_VERBOSE:-0}"
QUICK="${VICTOR_QUICK_TESTS:-0}"
COVERAGE_DIR="${PROJECT_ROOT}/coverage-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# =============================================================================
# Test Functions
# =============================================================================

run_yaml_validation_tests() {
    log_section "Running Workflow YAML Validation Tests"

    local pytest_args=(
        "tests/integration/workflows/test_workflow_yaml_validation.py"
        "-v"
        "--tb=short"
    )

    if [[ "$VERBOSE" == "1" ]]; then
        pytest_args+=("-vv" "-s")
    fi

    if [[ "$QUICK" == "1" ]]; then
        pytest_args+=("-m" "not slow")
    fi

    pytest_args+=(
        "--cov=victor.workflows"
        "--cov-report=term-missing"
        "--cov-report=xml:${COVERAGE_DIR}/workflow_yaml_coverage.xml"
        "--cov-report=html:${COVERAGE_DIR}/workflow_yaml_html"
    )

    log_info "Running: pytest ${pytest_args[*]}"
    pytest "${pytest_args[@]}"

    log_success "YAML validation tests passed"
}

run_team_node_tests() {
    log_section "Running Team Node Tests"

    local pytest_args=(
        "tests/integration/workflows/test_team_nodes.py"
        "-v"
        "--tb=short"
    )

    if [[ "$VERBOSE" == "1" ]]; then
        pytest_args+=("-vv" "-s")
    fi

    if [[ "$QUICK" == "1" ]]; then
        pytest_args+=("-m" "not slow")
    fi

    pytest_args+=(
        "--cov=victor.workflows.definition"
        "--cov-report=term-missing"
        "--cov-append"
        "--cov-report=xml:${COVERAGE_DIR}/team_node_coverage.xml"
    )

    log_info "Running: pytest ${pytest_args[*]}"
    pytest "${pytest_args[@]}"

    log_success "Team node tests passed"
}

run_compiler_executor_tests() {
    log_section "Running Compiler Executor Integration Tests"

    local pytest_args=(
        "tests/integration/workflows/test_compiler_executor_integration.py"
        "-v"
        "--tb=short"
    )

    if [[ "$VERBOSE" == "1" ]]; then
        pytest_args+=("-vv" "-s")
    fi

    pytest_args+=(
        "--cov=victor.workflows.unified_compiler"
        "--cov-report=term-missing"
        "--cov-append"
        "--cov-report=xml:${COVERAGE_DIR}/compiler_coverage.xml"
    )

    log_info "Running: pytest ${pytest_args[*]}"
    pytest "${pytest_args[@]}"

    log_success "Compiler executor tests passed"
}

run_workflow_execution_tests() {
    log_section "Running Workflow Execution Tests"

    local pytest_args=(
        "tests/integration/workflows/test_workflow_execution.py"
        "-v"
        "--tb=short"
    )

    if [[ "$VERBOSE" == "1" ]]; then
        pytest_args+=("-vv" "-s")
    fi

    if [[ "$QUICK" == "1" ]]; then
        pytest_args+=("-m" "not slow")
    fi

    pytest_args+=(
        "--cov=victor.framework.workflow_engine"
        "--cov-report=term-missing"
        "--cov-append"
        "--cov-report=xml:${COVERAGE_DIR}/execution_coverage.xml"
    )

    log_info "Running: pytest ${pytest_args[*]}"
    pytest "${pytest_args[@]}"

    log_success "Workflow execution tests passed"
}

run_recursion_limit_tests() {
    log_section "Running Recursion Limit Tests"

    local pytest_args=(
        "tests/integration/workflows/test_team_nodes.py::TestTeamNodeExecution"
        "-v"
        "--tb=short"
    )

    if [[ "$VERBOSE" == "1" ]]; then
        pytest_args+=("-vv" "-s")
    fi

    pytest_args+=(
        "--cov=victor.workflows.recursion"
        "--cov-report=term-missing"
        "--cov-append"
        "--cov-report=xml:${COVERAGE_DIR}/recursion_coverage.xml"
    )

    log_info "Running: pytest ${pytest_args[*]}"
    pytest "${pytest_args[@]}"

    log_success "Recursion limit tests passed"
}

validate_example_workflows() {
    log_section "Validating Example Workflows"

    log_info "Finding all workflow YAML files..."

    local workflow_files=()
    while IFS= read -r -d '' file; do
        workflow_files+=("$file")
    done < <(find victor -name "*.yaml" -o -name "*.yml" | grep -i workflow || true)

    log_info "Found ${#workflow_files[@]} workflow files"

    local failed=0
    local passed=0

    for workflow_file in "${workflow_files[@]}"; do
        log_info "Validating: $workflow_file"

        if python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from victor.workflows import load_workflow_from_file
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

try:
    loaded = load_workflow_from_file('${workflow_file}')
    compiler = UnifiedWorkflowCompiler()

    if isinstance(loaded, dict):
        workflows = loaded
    else:
        workflows = {loaded.name: loaded}

    for name, definition in workflows.items():
        compiled = compiler.compile_definition(definition)
    print('✓ Valid')
except Exception as e:
    print(f'✗ Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
        ((passed++))
        log_success "Valid: $workflow_file"
    else
        ((failed++))
        log_error "Invalid: $workflow_file"
    fi
    done

    echo ""
    log_info "Workflow validation complete: $passed passed, $failed failed"

    if [[ $failed -gt 0 ]]; then
        log_error "Some workflows failed validation"
        return 1
    fi

    log_success "All workflows validated successfully"
}

test_recursion_depths() {
    log_section "Testing Recursion Depth Limits"

    log_info "Testing various recursion depth configurations..."

    python - <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from victor.workflows import load_workflow_from_file
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.workflows.recursion import RecursionContext, RecursionDepthError

# Test various depth limits
test_depths = [1, 2, 3, 5, 10]

for depth in test_depths:
    ctx = RecursionContext(max_depth=depth)

    # Try to nest to max_depth
    for i in range(depth):
        assert ctx.can_nest(1), f"Should be able to nest at level {i}/{depth}"
        ctx.enter("workflow", f"workflow_{i}")

    # Should not be able to nest beyond max_depth
    assert not ctx.can_nest(1), f"Should not be able to nest beyond {depth}"

    # Trying to nest should raise error
    try:
        ctx.enter("workflow", "too_deep")
        print(f"✗ Depth {depth}: Should have raised RecursionDepthError")
        sys.exit(1)
    except RecursionDepthError:
        print(f"✓ Depth {depth}: Correctly prevented nesting beyond limit")

print("\n✓ All recursion depth tests passed")
EOF

    log_success "Recursion depth tests passed"
}

generate_coverage_report() {
    log_section "Generating Coverage Report"

    log_info "Combining coverage data..."

    # Combine all coverage files
    local coverage_files=(
        "${COVERAGE_DIR}/workflow_yaml_coverage.xml"
        "${COVERAGE_DIR}/team_node_coverage.xml"
        "${COVERAGE_DIR}/compiler_coverage.xml"
        "${COVERAGE_DIR}/execution_coverage.xml"
        "${COVERAGE_DIR}/recursion_coverage.xml"
    )

    # Generate combined report
    pytest --cov=victor.workflows \
           --cov=victor.framework.workflow_engine \
           --cov-branch \
           --cov-report=term-missing \
           --cov-report=html:${COVERAGE_DIR}/combined_html \
           --cov-report=json:${COVERAGE_DIR}/coverage.json \
           --cov-report=xml:${COVERAGE_DIR}/coverage.xml \
           tests/integration/workflows/ -q || true

    # Print coverage summary
    if [[ -f "${COVERAGE_DIR}/coverage.json" ]]; then
        log_info "Coverage Summary:"
        python - <<'EOF'
import json
from pathlib import Path

coverage_file = Path("${COVERAGE_DIR}/coverage.json")
if coverage_file.exists():
    with open(coverage_file) as f:
        data = json.load(f)

    totals = data.get('totals', {})
    percent_covered = totals.get('percent_covered', 0)

    print(f"  Overall Coverage: {percent_covered:.1f}%")
    print(f"  Lines Covered: {totals.get('covered_lines', 0)}")
    print(f"  Lines Total: {totals.get('num_statements', 0)}")
    print(f"  Missing Lines: {totals.get('missing_lines', 0)}")
EOF
    fi

    log_success "Coverage report generated in ${COVERAGE_DIR}/"
}

print_summary() {
    local exit_code="$1"
    local elapsed_time="$2"

    echo ""
    log_section "Test Summary"

    echo "Total Time: ${elapsed_time}s"
    echo "Exit Code: $exit_code"

    if [[ $exit_code -eq 0 ]]; then
        log_success "All workflow tests passed!"
    else
        log_error "Some workflow tests failed"
    fi

    echo ""
    echo "Coverage Reports: ${COVERAGE_DIR}/"
    echo "  - HTML: ${COVERAGE_DIR}/combined_html/index.html"
    echo "  - XML: ${COVERAGE_DIR}/coverage.xml"
    echo "  - JSON: ${COVERAGE_DIR}/coverage.json"
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    local start_time=$(date +%s)

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verbose|-v)
                VERBOSE=1
                VICTOR_TEST_VERBOSE=1
                shift
                ;;
            --quick|-q)
                QUICK=1
                VICTOR_QUICK_TESTS=1
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--verbose] [--quick]"
                echo ""
                echo "Options:"
                echo "  --verbose, -v    Enable verbose output"
                echo "  --quick, -q      Run only quick tests (skip slow ones)"
                echo "  --help, -h       Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Create coverage directory
    mkdir -p "$COVERAGE_DIR"

    # Run all tests
    local exit_code=0

    run_yaml_validation_tests || exit_code=$?
    run_team_node_tests || exit_code=$?
    run_compiler_executor_tests || exit_code=$?
    run_workflow_execution_tests || exit_code=$?
    run_recursion_limit_tests || exit_code=$?
    validate_example_workflows || exit_code=$?
    test_recursion_depths || exit_code=$?
    generate_coverage_report || exit_code=$?

    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))

    # Print summary
    print_summary $exit_code $elapsed_time

    exit $exit_code
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
