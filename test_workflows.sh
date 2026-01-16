#!/bin/bash
# Comprehensive YAML Workflow Testing Script

set -e

PROVIDER="ollama"
MODEL="qwen2.5-coder:1.5b"
ENDPOINT="http://localhost:11434"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Victor AI - YAML Workflow Testing & Deployment              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# Test result array
declare -a FAILED_WORKFLOWS
declare -a PASSED_WORKFLOWS

# Function to validate a workflow
test_workflow() {
    local workflow_file=$1
    local workflow_name=$(basename "$workflow_file" .yaml)
    local vertical=$(echo "$workflow_file" | cut -d'/' -f2)

    echo -e "${BLUE}Testing: ${vertical}/${workflow_name}${NC}"
    echo "File: $workflow_file"

    # Step 1: Validate YAML syntax
    echo "  → Validating YAML syntax..."
    if ! python -c "import yaml; yaml.safe_load(open('$workflow_file'))" 2>/dev/null; then
        echo -e "    ${RED}✗ FAILED: Invalid YAML syntax${NC}"
        ((FAILED++))
        FAILED_WORKFLOWS+=("$vertical/$workflow_name (YAML syntax error)")
        return 1
    fi

    # Step 2: Try to load/compile the workflow
    echo "  → Loading workflow..."
    if python -c "
import sys
sys.path.insert(0, '.')
try:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    workflow = load_workflow_from_file('$workflow_file')
    compiler = UnifiedWorkflowCompiler()
    compiled = compiler.compile_workflow(workflow)
    print('    Successfully loaded and compiled')
except Exception as e:
    print(f'    Error: {e}')
    sys.exit(1)
" 2>&1 | tail -5; then
        # Check if it was successful
        if ! grep -q "Error:" <<< "$(python -c "
import sys
sys.path.insert(0, '.')
try:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    workflow = load_workflow_from_file('$workflow_file')
    compiler = UnifiedWorkflowCompiler()
    compiled = compiler.compile_workflow(workflow)
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {e}')
" 2>&1)"; then
            echo -e "    ${GREEN}✓ PASSED: Workflow loaded successfully${NC}"
            ((PASSED++))
            PASSED_WORKFLOWS+=("$vertical/$workflow_name")
            return 0
        fi
    else
        echo -e "    ${RED}✗ FAILED: Could not load workflow${NC}"
        ((FAILED++))
        FAILED_WORKFLOWS+=("$vertical/$workflow_name (load error)")
        return 1
    fi

    echo ""
}

# Array of all workflows
WORKFLOWS=(
    # Coding workflows
    "victor/coding/workflows/bugfix.yaml"
    "victor/coding/workflows/code_review.yaml"
    "victor/coding/workflows/feature.yaml"
    "victor/coding/workflows/refactor.yaml"
    "victor/coding/workflows/tdd.yaml"

    # DevOps workflows
    "victor/devops/workflows/container_setup.yaml"
    "victor/devops/workflows/deploy.yaml"

    # RAG workflows
    "victor/rag/workflows/ingest.yaml"
    "victor/rag/workflows/query.yaml"

    # Data Analysis workflows
    "victor/dataanalysis/workflows/data_cleaning.yaml"
    "victor/dataanalysis/workflows/statistical_analysis.yaml"

    # Research workflows
    "victor/research/workflows/fact_check.yaml"
    "victor/research/workflows/literature_review.yaml"

    # Framework workflows
    "victor/workflows/feature_workflows.yaml"
    "victor/workflows/mode_workflows.yaml"
)

# Main testing loop
echo "Starting workflow validation..."
echo "Found ${#WORKFLOWS[@]} workflows to test"
echo ""

for workflow in "${WORKFLOWS[@]}"; do
    ((TOTAL++))
    test_workflow "$workflow"
    echo ""
done

# Summary
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        Summary                                   ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo -e "║ Total Workflows:     $TOTAL"
echo -e "║ ${GREEN}Passed:             $PASSED${NC}"
echo -e "║ ${RED}Failed:              $FAILED${NC}"
echo -e "║ ${YELLOW}Skipped:             $SKIPPED${NC}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed Workflows:${NC}"
    for failed in "${FAILED_WORKFLOWS[@]}"; do
        echo "  - $failed"
    done
    echo ""
fi

if [ $PASSED -gt 0 ]; then
    echo -e "${GREEN}Successfully Tested Workflows:${NC}"
    for passed in "${PASSED_WORKFLOWS[@]}"; do
        echo "  ✓ $passed"
    done
    echo ""
fi

exit $FAILED
