#!/bin/bash
# Victor Production Smoke Tests

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Victor Production Smoke Tests      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""

PASSED=0
FAILED=0

# Test 1: Version check
echo -n "Test 1: Version check... "
if docker exec victor-production victor --version &> /dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAILED++))
fi

# Test 2: Basic chat (Anthropic)
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -n "Test 2: Basic chat (Anthropic)... "
    OUTPUT=$(docker exec victor-production victor chat --provider anthropic --no-tui "Say 'test successful' and nothing else" 2>&1)
    if [[ $OUTPUT == *"successful"* ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "Output: $OUTPUT"
        ((FAILED++))
    fi
fi

# Test 3: Tool execution
echo -n "Test 3: Tool execution (list directory)... "
OUTPUT=$(docker exec victor-production victor chat --no-tui "List files in /app directory" 2>&1)
if [[ $OUTPUT == *"Dockerfile"* ]] || [[ $OUTPUT == *"pyproject.toml"* ]]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    echo "Output: $OUTPUT"
    ((FAILED++))
fi

# Test 4: API health check (if API server is running)
echo -n "Test 4: API health check... "
if curl -sf http://localhost:8765/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ SKIP (API server not running)${NC}"
fi

# Test 5: Prometheus metrics endpoint
echo -n "Test 5: Prometheus metrics... "
if curl -sf http://localhost:9091/-/healthy > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAILED++))
fi

# Test 6: Grafana health
echo -n "Test 6: Grafana health... "
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAILED++))
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Smoke Test Results             ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""
echo -e "  Passed: ${GREEN}$PASSED${NC}"
echo -e "  Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All smoke tests passed! 🎉${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs for details.${NC}"
    echo "View logs: docker-compose -f docker-compose.production.yml logs"
    exit 1
fi
