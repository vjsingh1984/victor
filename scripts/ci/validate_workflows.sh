#!/bin/bash
# Validate all YAML workflows in the project
# Usage: scripts/ci/validate_workflows.sh [--ci] [workflow_paths...]

set -e

CI_MODE=""
VERBOSE=""
WORKFLOW_PATHS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI_MODE="--ci"
      shift
      ;;
    --verbose|-v)
      VERBOSE="--verbose"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--ci] [--verbose] [workflow_paths...]"
      echo ""
      echo "Options:"
      echo "  --ci       CI mode (JSON output, no colors)"
      echo "  --verbose  Show detailed validation output"
      echo "  --help     Show this help"
      echo ""
      echo "Arguments:"
      echo "  workflow_paths  Specific workflow files to validate (default: all)"
      exit 0
      ;;
    *)
      WORKFLOW_PATHS+=("$1")
      shift
      ;;
  esac
done

# If no paths provided, find all workflow files
if [ ${#WORKFLOW_PATHS[@]} -eq 0 ]; then
  echo "Finding all workflow YAML files..."
  WORKFLOW_PATHS=($(find victor -name "workflows" -type d -exec find {} -name "*.yaml" -o -name "*.yml" \;))
fi

# Install Victor if needed
if ! python -c "import victor" 2>/dev/null; then
  echo "Installing Victor..."
  pip install -e ".[dev]" > /dev/null 2>&1
fi

# Validation counters
TOTAL=0
PASSED=0
FAILED=0
FAILED_WORKFLOWS=()

echo "Validating ${#WORKFLOW_PATHS[@]} workflow files..."
echo ""

# Validate each workflow
for workflow in "${WORKFLOW_PATHS[@]}"; do
  TOTAL=$((TOTAL + 1))

  if [ -n "$VERBOSE" ]; then
    echo "Validating: $workflow"
  fi

  # Use Python to validate the workflow
  if python -c "
import sys
from pathlib import Path

try:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    compiler = UnifiedWorkflowCompiler()
    loaded = load_workflow_from_file('$workflow')

    if isinstance(loaded, dict):
        workflow_defs = loaded
    else:
        workflow_defs = {loaded.name: loaded}

    for name, definition in workflow_defs.items():
        compiled = compiler.compile_definition(definition)

    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1 | grep -q "OK"; then
    if [ -n "$VERBOSE" ]; then
      echo "  ✓ Valid"
    fi
    PASSED=$((PASSED + 1))
  else
    echo "✗ Failed: $workflow"
    FAILED_WORKFLOWS+=("$workflow")
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=" 60
echo "Validation Results:"
echo "  Total:   $TOTAL"
echo "  Passed:  $PASSED"
echo "  Failed:  $FAILED"
echo "=" 60

if [ $FAILED -gt 0 ]; then
  echo ""
  echo "Failed workflows:"
  for workflow in "${FAILED_WORKFLOWS[@]}"; do
    echo "  - $workflow"
  done
  exit 1
else
  echo ""
  echo "✓ All workflows validated successfully!"
  exit 0
fi
