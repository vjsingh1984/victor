#!/bin/bash
# Verification script for Phase 4: Enhanced Entry Points Implementation
#
# This script verifies that the SDK protocol discovery system works correctly
# after external verticals are reinstalled with the new entry points.
#
# Usage:
#   ./verify_phase4_implementation.sh

set -e

echo "==================================================================="
echo "Phase 4: Enhanced Entry Points - Verification Script"
echo "==================================================================="
echo ""

# Step 1: Build victor-sdk (if not already built)
echo "Step 1: Building victor-sdk..."
cd /Users/vijaysingh/code/codingagent/victor-sdk
python -m pip install -e . --quiet
echo "✓ victor-sdk installed"
echo ""

# Step 2: Build victor-ai (if not already built)
echo "Step 2: Building victor-ai..."
cd /Users/vijaysingh/code/codingagent
python -m pip install -e . --quiet
echo "✓ victor-ai installed"
echo ""

# Step 3: Build and install external verticals with new entry points
echo "Step 3: Installing external verticals with new SDK entry points..."

# victor-coding
echo "  - Installing victor-coding..."
cd /Users/vijaysingh/code/victor-coding
python -m pip install -e . --quiet
echo "    ✓ victor-coding installed"

# victor-devops
echo "  - Installing victor-devops..."
cd /Users/vijaysingh/code/victor-devops
python -m pip install -e . --quiet
echo "    ✓ victor-devops installed"

# victor-rag
echo "  - Installing victor-rag..."
cd /Users/vijaysingh/code/victor-rag
python -m pip install -e . --quiet
echo "    ✓ victor-rag installed"

# victor-research
echo "  - Installing victor-research..."
cd /Users/vijaysingh/code/victor-research
python -m pip install -e . --quiet
echo "    ✓ victor-research installed"

# victor-dataanalysis
echo "  - Installing victor-dataanalysis..."
cd /Users/vijaysingh/code/victor-dataanalysis
python -m pip install -e . --quiet
echo "    ✓ victor-dataanalysis installed"

echo ""
echo "Step 4: Running verification tests..."
echo ""

# Test SDK discovery
python -c "
from victor.core.verticals import (
    discover_sdk_protocols,
    get_sdk_discovery_stats,
    get_sdk_discovery_summary,
    get_sdk_tool_providers,
    get_sdk_safety_providers,
    get_sdk_workflow_providers,
    get_sdk_prompt_providers,
    get_sdk_capability_providers,
)

print('Testing SDK Protocol Discovery...')
print('=' * 60)

# Discover all SDK protocols
stats = discover_sdk_protocols()
print(f'Discovery Statistics:')
print(f'  Verticals: {stats.total_verticals}')
print(f'  Protocols: {stats.total_protocols}')
print(f'  Capabilities: {stats.total_capabilities}')
print(f'  Validators: {stats.total_validators}')
print(f'  Failed Loads: {stats.failed_loads}')
print()

# Get specific provider types
tool_providers = get_sdk_tool_providers()
safety_providers = get_sdk_safety_providers()
workflow_providers = get_sdk_workflow_providers()
prompt_providers = get_sdk_prompt_providers()
capability_providers = get_sdk_capability_providers()

print(f'Provider Counts:')
print(f'  Tool Providers: {len(tool_providers)}')
print(f'  Safety Providers: {len(safety_providers)}')
print(f'  Workflow Providers: {len(workflow_providers)}')
print(f'  Prompt Providers: {len(prompt_providers)}')
print(f'  Capability Providers: {len(capability_providers)}')
print()

# List tool providers
if tool_providers:
    print('Tool Providers:')
    for p in tool_providers:
        tools = p.get_tools()
        print(f'  - {p.__class__.__name__}: {len(tools)} tools')
    print()

# List capability providers
if capability_providers:
    print('Capability Providers:')
    for name in sorted(capability_providers.keys()):
        print(f'  - {name}')
    print()

# Print full discovery summary
print('=' * 60)
print('Full Discovery Summary:')
print('=' * 60)
print(get_sdk_discovery_summary())
"

echo ""
echo "==================================================================="
echo "Verification Complete!"
echo "==================================================================="
echo ""
echo "If you see protocols and capabilities listed above, Phase 4 is"
echo "successfully implemented and working!"
echo ""
echo "Next Steps:"
echo "  1. Run: pytest tests/unit/test_sdk_discovery.py"
echo "  2. Create E2E test for zero-dependency vertical"
echo "  3. Update documentation"
echo ""
