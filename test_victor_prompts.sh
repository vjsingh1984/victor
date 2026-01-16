#!/bin/bash
# Victor AI Test Drive Script

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Victor AI - Test Drive with Varying Prompts               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

PROVIDER="ollama"
MODEL="qwen2.5-coder:1.5b"
ENDPOINT="http://localhost:11434"

# Test 1: Simple greeting
echo "📝 Test 1: Simple Greeting"
echo "─────────────────────────────────────────────────────"
victor chat --no-tui --plain --provider "$PROVIDER" --model "$MODEL" --endpoint "$ENDPOINT" \
    "Hello! Who are you? Answer in one short sentence." 2>&1 | \
    grep -E "^(I am|I'm|Hello|Hi|Victor)" | head -1
echo ""

# Test 2: Factual question
echo "📝 Test 2: Factual Knowledge"
echo "─────────────────────────────────────────────────────"
victor chat --no-tui --plain --provider "$PROVIDER" --model "$MODEL" --endpoint "$ENDPOINT" \
    "What is the capital of France? Answer in one word." 2>&1 | \
    grep -iE "^(Paris)" | head -1
echo ""

# Test 3: Math
echo "📝 Test 3: Mathematical Calculation"
echo "─────────────────────────────────────────────────────"
victor chat --no-tui --plain --provider "$PROVIDER" --model "$MODEL" --endpoint "$ENDPOINT" \
    "What is 17 multiplied by 3? Just give the number." 2>&1 | \
    grep -E "^[0-9]+" | head -1
echo ""

# Test 4: Code snippet
echo "📝 Test 4: Simple Code Generation"
echo "─────────────────────────────────────────────────────"
victor chat --no-tui --plain --provider "$PROVIDER" --model "$MODEL" --endpoint "$ENDPOINT" \
    "Write a Python function that adds two numbers. Keep it under 5 lines." 2>&1 | \
    grep -E "def " | head -3
echo ""

# Test 5: Creative request
echo "📝 Test 5: Creative Request"
echo "─────────────────────────────────────────────────────"
victor chat --no-tui --plain --provider "$PROVIDER" --model "$MODEL" --endpoint "$ENDPOINT" \
    "Write a haiku about coding. 3 lines only." 2>&1 | \
    tail -5
echo ""

# Test 6: Analytical question
echo "📝 Test 6: Analytical Reasoning"
echo "─────────────────────────────────────────────────────"
victor chat --no-tui --plain --provider "$PROVIDER" --model "$MODEL" --endpoint "$ENDPOINT" \
    "Which is larger: 15% of 200 or 20% of 150? Just tell me which one." 2>&1 | \
    grep -E "(15%|20%|First|Second)" | head -1
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Test Drive Complete!                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
