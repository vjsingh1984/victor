#!/bin/bash
# Victor Air-Gapped Semantic Tool Selection Demo
# Demonstrates intelligent tool selection and execution without internet

set -e

# Load shared utilities
source /app/docker/scripts/colors.sh 2>/dev/null || {
    # Fallback if colors.sh not found
    GREEN='' BLUE='' YELLOW='' CYAN='' RED='' BOLD='' NC=''
}

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

# Wait for Ollama to be ready (using shared utility)
source /app/docker/scripts/wait-for-ollama.sh || exit 1

# Ensure model is available (using shared utility)
bash /app/docker/scripts/ensure-model.sh qwen2.5-coder:1.5b "1 GB"

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
