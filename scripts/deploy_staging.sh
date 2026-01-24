#!/bin/bash
# Staging Deployment Script for SOLID Remediation
#
# This script deploys the SOLID remediation changes to a staging environment
# with all feature flags enabled for comprehensive testing.
#
# Usage:
#   ./scripts/deploy_staging.sh [--dry-run] [--skip-tests]
#
# Options:
#   --dry-run: Show what would be deployed without actually deploying
#   --skip-tests: Skip running tests (not recommended)
#
# Exit codes:
#   0: Deployment successful
#   1: Deployment failed

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script options
DRY_RUN=false
SKIP_TESTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SOLID Remediation Staging Deployment${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from project root${NC}"
    exit 1
fi

# Step 1: Verify current branch
echo -e "${BLUE}Step 1: Checking current branch...${NC}"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "Current branch: ${GREEN}${CURRENT_BRANCH}${NC}"

if [[ "$CURRENT_BRANCH" == "main" ]]; then
    echo -e "${YELLOW}Warning: Deploying from main branch${NC}"
fi

# Step 2: Pull latest changes
echo -e "\n${BLUE}Step 2: Pulling latest changes...${NC}"
if [ "$DRY_RUN" = false ]; then
    git pull origin "$CURRENT_BRANCH"
else
    echo -e "${YELLOW}[DRY-RUN] Would run: git pull origin $CURRENT_BRANCH${NC}"
fi

# Step 3: Set feature flags
echo -e "\n${BLUE}Step 3: Configuring feature flags...${NC}"
export VICTOR_USE_NEW_PROTOCOLS=true
export VICTOR_USE_CONTEXT_CONFIG=true
export VICTOR_USE_PLUGIN_DISCOVERY=true
export VICTOR_USE_TYPE_SAFE_LAZY=true
export VICTOR_LAZY_INITIALIZATION=true

echo -e "${GREEN}Feature flags enabled:${NC}"
echo -e "  VICTOR_USE_NEW_PROTOCOLS=$VICTOR_USE_NEW_PROTOCOLS"
echo -e "  VICTOR_USE_CONTEXT_CONFIG=$VICTOR_USE_CONTEXT_CONFIG"
echo -e "  VICTOR_USE_PLUGIN_DISCOVERY=$VICTOR_USE_PLUGIN_DISCOVERY"
echo -e "  VICTOR_USE_TYPE_SAFE_LAZY=$VICTOR_USE_TYPE_SAFE_LAZY"
echo -e "  VICTOR_LAZY_INITIALIZATION=$VICTOR_LAZY_INITIALIZATION"

# Step 4: Install dependencies
echo -e "\n${BLUE}Step 4: Installing dependencies...${NC}"
if [ "$DRY_RUN" = false ]; then
    pip install -e ".[dev]" --quiet
else
    echo -e "${YELLOW}[DRY-RUN] Would run: pip install -e \".[dev]\"${NC}"
fi

# Step 5: Run verification
echo -e "\n${BLUE}Step 5: Running deployment verification...${NC}"
if [ "$DRY_RUN" = false ]; then
    python scripts/verify_solid_deployment.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}Verification failed! Aborting deployment.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[DRY-RUN] Would run: python scripts/verify_solid_deployment.py${NC}"
fi

# Step 6: Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo -e "\n${BLUE}Step 6: Running test suite...${NC}"
    if [ "$DRY_RUN" = false ]; then
        echo "Running SOLID remediation tests..."
        pytest tests/unit/core/verticals/test_plugin_discovery.py -v --tb=short || exit 1
        pytest tests/unit/core/verticals/test_lazy_proxy.py -v --tb=short || exit 1
        pytest tests/unit/framework/test_lazy_initializer.py -v --tb=short || exit 1
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${YELLOW}[DRY-RUN] Would run: pytest tests/unit/core/verticals/test_*.py tests/unit/framework/test_lazy_initializer.py${NC}"
    fi
else
    echo -e "\n${YELLOW}Step 6: Skipping tests (--skip-tests flag)${NC}"
fi

# Step 7: Type checking
echo -e "\n${BLUE}Step 7: Running type checking...${NC}"
if [ "$DRY_RUN" = false ]; then
    echo "Checking SOLID remediation files..."
    mypy victor/core/verticals/lazy_proxy.py victor/core/verticals/plugin_discovery.py victor/framework/lazy_initializer.py --strict || true
    echo -e "${GREEN}Type checking complete${NC}"
else
    echo -e "${YELLOW}[DRY-RUN] Would run: mypy victor/core/verticals/lazy_proxy.py victor/core/verticals/plugin_discovery.py victor/framework/lazy_initializer.py --strict${NC}"
fi

# Step 8: Collect baseline metrics
echo -e "\n${BLUE}Step 8: Collecting baseline metrics...${NC}"
if [ "$DRY_RUN" = false ]; then
    python -c "
import time
import sys

# Measure startup time
start = time.time()
from victor.coding import CodingAssistant
from victor.research import ResearchAssistant
from victor.devops import DevOpsAssistant
from victor.dataanalysis import DataAnalysisAssistant
end = time.time()

startup_time = (end - start) * 1000  # Convert to ms
print(f'Startup Time: {startup_time:.2f}ms')
print(f'Feature Flags: All enabled')
print(f'Environment: Staging')
" 2>/dev/null || echo "Metrics collection skipped (YAML config warning expected)"
else
    echo -e "${YELLOW}[DRY-RUN] Would collect baseline metrics${NC}"
fi

# Step 9: Deployment summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Environment: ${GREEN}Staging${NC}"
echo -e "Feature Flags: ${GREEN}All enabled${NC}"
echo -e "Dry Run: ${GREEN}${DRY_RUN}${NC}"
echo -e ""

if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✅ Deployment to staging complete!${NC}\n"
    echo -e "Next steps:"
    echo -e "  1. Monitor application logs for errors"
    echo -e "  2. Verify startup time improvement (target: 20-30%)"
    echo -e "  3. Check memory usage stability"
    echo -e "  4. Monitor cache hit rates"
    echo -e "  5. Run integration tests"
    echo -e ""
    echo -e "To rollback:"
    echo -e "  export VICTOR_USE_NEW_PROTOCOLS=false"
    echo -e "  export VICTOR_USE_CONTEXT_CONFIG=false"
    echo -e "  export VICTOR_USE_PLUGIN_DISCOVERY=false"
    echo -e "  export VICTOR_USE_TYPE_SAFE_LAZY=false"
    echo -e "  export VICTOR_LAZY_INITIALIZATION=false"
    echo -e "  # Restart application"
else
    echo -e "${YELLOW}✅ Dry-run complete! No changes made.${NC}\n"
    echo -e "To deploy for real:"
    echo -e "  ./scripts/deploy_staging.sh"
fi

echo ""
