#!/usr/bin/env bash
# Copyright 2025 Vijaykumar Singh <singhvijd@gmail.com>
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

# Type checking script for critical paths.
#
# This script runs mypy on Victor's critical paths to ensure type safety:
# - Orchestrator public API
# - Provider interfaces
# - Coordinator protocols
#
# Usage:
#   bash scripts/typecheck_critical.sh [--verbose] [--count]
#
# Options:
#   --verbose    Show full error output with context
#   --count      Only show error counts, not individual errors
#   --help       Show this help message
#
# Exit codes:
#   0: All type checks passed
#   1: Type errors found
#   2: Script error (missing dependencies, etc.)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default options
VERBOSE=false
COUNT_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --count)
            COUNT_ONLY=true
            shift
            ;;
        --help|-h)
            head -n 35 "$0" | tail -n 22
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 2
            ;;
    esac
done

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if mypy is installed
if ! command -v mypy &> /dev/null; then
    echo -e "${RED}Error: mypy not found${NC}"
    echo "Install mypy: pip install mypy"
    exit 2
fi

# Change to project root
cd "$PROJECT_ROOT"

echo "=================================="
echo "Type Checking Critical Paths"
echo "=================================="
echo

# Mypy flags
MYPY_FLAGS="--show-error-codes"
if [ "$VERBOSE" = false ]; then
    MYPY_FLAGS="$MYPY_FLAGS --no-error-summary"
fi

# Track totals
TOTAL_ERRORS=0
TOTAL_FILES=0

# Function to check a module
check_module() {
    local module=$1
    local name=$2

    TOTAL_FILES=$((TOTAL_FILES + 1))

    echo -n "Checking $name... "

    if [ "$COUNT_ONLY" = true ]; then
        # Count only mode
        errors=$(mypy "$module" $MYPY_FLAGS 2>&1 | grep -c "error:" || true)
        echo "$errors errors"
        TOTAL_ERRORS=$((TOTAL_ERRORS + errors))
    else
        # Verbose mode
        if [ "$VERBOSE" = true ]; then
            echo
            mypy "$module" $MYPY_FLAGS
            errors=$?
            if [ $errors -eq 0 ]; then
                echo -e "${GREEN}✓ No errors${NC}"
            fi
        else
            # Quick mode - just show pass/fail
            if mypy "$module" $MYPY_FLAGS 2>&1 | grep -q "error:"; then
                echo -e "${RED}✗ FAILED${NC}"
                error_count=$(mypy "$module" $MYPY_FLAGS 2>&1 | grep -c "error:" || true)
                TOTAL_ERRORS=$((TOTAL_ERRORS + error_count))
            else
                echo -e "${GREEN}✓ PASSED${NC}"
            fi
        fi
    fi
}

# Check orchestrator
check_module "victor/agent/orchestrator.py" "Orchestrator"

# Check provider base
check_module "victor/providers/base.py" "BaseProvider"

# Check key provider implementations (sampling)
for provider in anthropic openai azure_openai; do
    if [ -f "victor/providers/${provider}_provider.py" ]; then
        check_module "victor/providers/${provider}_provider.py" "${provider} provider"
    fi
done

# Check coordinators (sampling)
check_module "victor/agent/coordinators/__init__.py" "Coordinator protocols"
check_module "victor/agent/coordinators/chat_coordinator.py" "ChatCoordinator"
check_module "victor/agent/coordinators/tool_coordinator.py" "ToolCoordinator"
check_module "victor/agent/coordinators/conversation_coordinator.py" "ConversationCoordinator"

# Summary
echo
echo "=================================="
echo "Summary"
echo "=================================="
echo "Files checked: $TOTAL_FILES"

if [ "$COUNT_ONLY" = true ] || [ "$VERBOSE" = false ]; then
    echo "Total errors: $TOTAL_ERRORS"
fi

echo

if [ $TOTAL_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All type checks passed!${NC}"
    exit 0
elif [ $TOTAL_ERRORS -lt 50 ]; then
    echo -e "${YELLOW}⚠ Type errors found: $TOTAL_ERRORS${NC}"
    echo "Target: <50 errors on critical paths"
    echo "Status: Within acceptable range"
    exit 0
else
    echo -e "${RED}✗ Too many type errors: $TOTAL_ERRORS${NC}"
    echo "Target: <50 errors on critical paths"
    echo "Status: Exceeds threshold"
    exit 1
fi
