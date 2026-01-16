#!/bin/bash
# Phase 2: Test remaining workflows

set -e

echo "=== Testing Remaining Workflows ===" && echo ""

# Phase 2 workflows
WORKFLOWS=(
    # Complex coding workflows
    "victor/coding/workflows/multi_agent_consensus.yaml"
    "victor/coding/workflows/team_node_example.yaml"

    # Advanced data analysis
    "victor/dataanalysis/workflows/automl_pipeline.yaml"
    "victor/dataanalysis/workflows/eda_pipeline.yaml"
    "victor/dataanalysis/workflows/ml_pipeline.yaml"

    # Research workflows
    "victor/research/workflows/competitive_analysis.yaml"
    "victor/research/workflows/deep_research.yaml"

    # Benchmark workflows
    "victor/benchmark/workflows/agentic_bench.yaml"
    "victor/benchmark/workflows/code_generation.yaml"
    "victor/benchmark/workflows/passk.yaml"
    "victor/benchmark/workflows/swe_bench.yaml"
)

for workflow in "${WORKFLOWS[@]}"; do
    workflow_name=$(basename "$workflow" .yaml)
    vertical=$(echo "$workflow" | cut -d'/' -f2)

    echo "Testing: ${vertical}/${workflow_name}"

    python -c "
import sys
sys.path.insert(0, '.')
try:
    from victor.workflows import load_workflow_from_file
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

    workflow = load_workflow_from_file('$workflow')
    compiler = UnifiedWorkflowCompiler()
    compiled = compiler.compile_graph(workflow)
    print('  ✓ PASSED')
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    sys.exit(1)
" 2>&1 | grep -E "(PASSED|FAILED)" || echo "  ✗ FAILED"
    echo ""
done
