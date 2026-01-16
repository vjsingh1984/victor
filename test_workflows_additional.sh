#!/bin/bash
# Test example and common workflows

set -e

echo "=== Testing Example and Common Workflows ===" && echo ""

# Example/migrated workflows
EXAMPLE_WORKFLOWS=(
    "victor/coding/workflows/examples/migrated_example.yaml"
    "victor/devops/workflows/examples/migrated_example.yaml"
    "victor/rag/workflows/examples/migrated_example.yaml"
    "victor/research/workflows/examples/migrated_example.yaml"
)

# Common/framework workflows
COMMON_WORKFLOWS=(
    "victor/workflows/common/hitl_gates.yaml"
    "victor/workflows/feature_workflows.yaml"
    "victor/workflows/mode_workflows.yaml"
)

echo "### Example Workflows ###" && echo ""

for workflow in "${EXAMPLE_WORKFLOWS[@]}"; do
    workflow_name=$(basename "$workflow" .yaml)
    vertical=$(echo "$workflow" | cut -d'/' -f2)

    echo "Testing: ${vertical}/examples/${workflow_name}"

    python -c "
import sys
sys.path.insert(0, '.')
try:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    loaded = load_workflow_from_file('$workflow')

    if isinstance(loaded, dict):
        workflows = loaded
    else:
        workflows = {loaded.name: loaded}

    compiler = UnifiedWorkflowCompiler()
    for name, definition in workflows.items():
        compiled = compiler.compile_definition(definition)

    print(f'  ✓ PASSED ({len(workflows)} workflow(s))')
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | grep -E "(PASSED|FAILED)" || echo "  ✗ FAILED"
    echo ""
done

echo ""
echo "### Common/Framework Workflows ###" && echo ""

for workflow in "${COMMON_WORKFLOWS[@]}"; do
    workflow_name=$(basename "$workflow" .yaml)
    path_name=$(echo "$workflow" | cut -d'/' -f2-)

    echo "Testing: ${path_name}"

    python -c "
import sys
sys.path.insert(0, '.')
try:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    loaded = load_workflow_from_file('$workflow')

    if isinstance(loaded, dict):
        workflows = loaded
    else:
        workflows = {loaded.name: loaded}

    compiler = UnifiedWorkflowCompiler()
    for name, definition in workflows.items():
        compiled = compiler.compile_definition(definition)

    print(f'  ✓ PASSED ({len(workflows)} workflow(s))')
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | grep -E "(PASSED|FAILED)" || echo "  ✗ FAILED"
    echo ""
done
