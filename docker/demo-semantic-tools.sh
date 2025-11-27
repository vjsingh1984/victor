#!/bin/bash
# Victor Air-Gapped Semantic Tool Selection Demo
# Demonstrates intelligent tool selection and execution without internet

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}${BLUE}Victor Air-Gapped Semantic Tool Selection Demo${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This demo showcases:"
echo "  ✓ Semantic tool selection using local embeddings"
echo "  ✓ Tool execution with fallback JSON parser"
echo "  ✓ 100% offline operation (no internet required)"
echo "  ✓ qwen3-coder:30b for high-quality code generation"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait for Ollama to be ready
echo -e "${CYAN}⏳ Waiting for Ollama to be ready...${NC}"
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
    echo "   Waiting for Ollama..."
    sleep 2
done
echo -e "${GREEN}✓ Ollama is ready${NC}"
echo ""

# Check if qwen2.5-coder:1.5b is available
echo -e "${CYAN}📦 Checking for qwen2.5-coder:1.5b model...${NC}"
if ! curl -s http://ollama:11434/api/tags | grep -q "qwen2.5-coder:1.5b"; then
    echo -e "${YELLOW}⚠ Model not found. Pulling qwen2.5-coder:1.5b (1 GB)...${NC}"
    echo "   This should take 1-3 minutes."
    curl -s http://ollama:11434/api/pull -d '{"name":"qwen2.5-coder:1.5b"}' || true
    echo""
else
    echo -e "${GREEN}✓ Model already available${NC}"
fi
echo ""

# Demo 1: Simple Function Creation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}Demo 1: Simple Function Creation${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}Prompt:${NC} \"Write a function to calculate factorial\""
echo ""

victor main "Write a function to calculate factorial in factorial.py"

echo ""
echo -e "${GREEN}✓ Demo 1 complete${NC}"
if [ -f factorial.py ]; then
    echo ""
    echo "Generated file:"
    cat factorial.py | head -20
    echo ""
fi

sleep 2
echo ""

# Demo 2: Email Validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}Demo 2: Email Validation with Regex${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}Prompt:${NC} \"Write email validation function using regex\""
echo ""

victor main "Write a Python function to validate email addresses using regex in email_validator.py"

echo ""
echo -e "${GREEN}✓ Demo 2 complete${NC}"
if [ -f email_validator.py ]; then
    echo ""
    echo "Generated file:"
    cat email_validator.py
    echo ""
fi

sleep 2
echo ""

# Demo 3: Prime Number Checker
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}Demo 3: Prime Number Checker (Advanced)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}Prompt:${NC} \"Write a function to check if a number is prime\""
echo ""

victor main "Write a function to check if a number is prime in prime_check.py" --no-stream

echo ""
echo -e "${GREEN}✓ Demo 3 complete${NC}"
if [ -f prime_check.py ]; then
    echo ""
    echo "Generated file (high-quality with docstrings and tests):"
    cat prime_check.py | head -30
    echo ""
fi

sleep 2
echo ""

# Demo 4: Calculator with Multiple Functions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}Demo 4: Calculator (Multi-Function)${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}Prompt:${NC} \"Write calculator with add, subtract, multiply, divide\""
echo ""

victor main "Write a calculator.py file with add, subtract, multiply, and divide functions"

echo ""
echo -e "${GREEN}✓ Demo 4 complete${NC}"
if [ -f calculator.py ]; then
    echo ""
    echo "Generated file:"
    cat calculator.py
    echo ""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}${GREEN}All Demos Complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Summary of Generated Files:"
ls -lh *.py 2>/dev/null || echo "  No files generated (check errors above)"
echo ""
echo "Key Features Demonstrated:"
echo "  ✓ Semantic tool selection (top 3-5 tools per query)"
echo "  ✓ Tool execution (write_file, execute_python_in_sandbox)"
echo "  ✓ Fallback JSON parser (qwen3-coder:30b compatibility)"
echo "  ✓ High-quality code generation (docstrings, tests, edge cases)"
echo "  ✓ Air-gapped operation (all-MiniLM-L12-v2 local embeddings)"
echo ""
echo "Next Steps:"
echo "  • Review generated files in the workspace"
echo "  • Run: victor profiles  (to see all available profiles)"
echo "  • Run: victor \"your custom prompt\"  (try your own tasks)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
