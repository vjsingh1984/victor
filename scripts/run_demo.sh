#!/bin/bash
# Victor AI Demo Runner
# Usage: ./run_demo.sh [demo_number]
# Run without arguments for interactive menu

set -e

DEMO_DIR=~/code/webui/investor_homelab
cd "$DEMO_DIR"

echo "============================================"
echo "  Victor AI Demo - investor_homelab project"
echo "============================================"
echo ""
echo "Current directory: $(pwd)"
echo ""

# Demo functions
demo_1_simple_local() {
    echo ">>> Demo 1: Simple Prompt - Local Ollama (Explore Mode)"
    echo ">>> Provider: default (qwen3-coder:30b)"
    echo ""
    victor chat --mode explore --no-tui "List all Python files in this project and give me a one-line description of each"
}

demo_2_medium_deepseek() {
    echo ">>> Demo 2: Medium Prompt - DeepSeek Cloud (Explore Mode)"
    echo ">>> Provider: deepseek"
    echo ""
    victor chat --profile deepseek --mode explore --no-tui "Analyze the database_schema.py file. What design improvements would you recommend for better data integrity and query performance?"
}

demo_3_complex_openai() {
    echo ">>> Demo 3: Complex Prompt - OpenAI GPT-4.1 (Plan Mode)"
    echo ">>> Provider: gpt-4.1"
    echo ""
    victor chat --profile gpt-4.1 --mode plan --no-tui "Create a comprehensive pytest test suite for the WebSearchClient class in utils/web_search_client.py. Include tests for initialization, API calls, and error handling."
}

demo_4_explore_local() {
    echo ">>> Demo 4: Explore Mode Deep Dive - Local"
    echo ">>> Provider: default"
    echo ""
    victor chat --mode explore --no-tui "How does the protocols/ directory implement SOLID principles? Show me the interface definitions."
}

demo_5_plan_deepseek() {
    echo ">>> Demo 5: Plan Mode Architecture - DeepSeek"
    echo ">>> Provider: deepseek"
    echo ""
    victor chat --profile deepseek --mode plan --no-tui "I want to add a new data provider for Yahoo Finance that follows the existing protocol patterns. Create a detailed implementation plan."
}

demo_6_build_openai() {
    echo ">>> Demo 6: Build Mode Implementation - GPT-4.1"
    echo ">>> Provider: gpt-4.1"
    echo ""
    victor chat --profile gpt-4.1 --mode build --no-tui "Add a retry decorator with exponential backoff to the existing code. It should retry up to 3 times with 1s, 2s, 4s delays."
}

demo_interactive() {
    echo ">>> Interactive TUI Demo"
    echo ">>> Starting Victor in TUI mode..."
    echo ""
    echo "Suggested prompts to try:"
    echo "  1. List all Python files and describe each"
    echo "  2. Analyze database_schema.py for improvements"
    echo "  3. /mode plan - then ask for implementation plan"
    echo "  4. /tools - show available tools"
    echo "  5. /stats - show session statistics"
    echo ""
    victor chat --mode explore --tui
}

# Main menu
show_menu() {
    echo "Select a demo to run:"
    echo ""
    echo "  1) Simple - Local Ollama (Explore)"
    echo "  2) Medium - DeepSeek (Explore)"
    echo "  3) Complex - OpenAI GPT-4.1 (Plan)"
    echo "  4) Explore Deep Dive - Local"
    echo "  5) Plan Architecture - DeepSeek"
    echo "  6) Build Implementation - OpenAI"
    echo "  7) Interactive TUI Demo"
    echo "  a) Run ALL demos (non-interactive)"
    echo "  q) Quit"
    echo ""
}

run_all() {
    echo "Running all demos..."
    echo ""
    demo_1_simple_local
    echo ""
    echo "============================================"
    echo ""
    demo_2_medium_deepseek
    echo ""
    echo "============================================"
    echo ""
    demo_3_complex_openai
    echo ""
    echo "============================================"
    echo ""
    demo_4_explore_local
    echo ""
    echo "============================================"
    echo ""
    demo_5_plan_deepseek
    echo ""
    echo "============================================"
    echo ""
    demo_6_build_openai
    echo ""
    echo "============================================"
    echo "All demos complete!"
}

# Handle command line argument
if [ -n "$1" ]; then
    case $1 in
        1) demo_1_simple_local ;;
        2) demo_2_medium_deepseek ;;
        3) demo_3_complex_openai ;;
        4) demo_4_explore_local ;;
        5) demo_5_plan_deepseek ;;
        6) demo_6_build_openai ;;
        7) demo_interactive ;;
        all|a) run_all ;;
        *) echo "Unknown demo: $1"; exit 1 ;;
    esac
    exit 0
fi

# Interactive menu
while true; do
    show_menu
    read -p "Enter choice: " choice
    echo ""

    case $choice in
        1) demo_1_simple_local ;;
        2) demo_2_medium_deepseek ;;
        3) demo_3_complex_openai ;;
        4) demo_4_explore_local ;;
        5) demo_5_plan_deepseek ;;
        6) demo_6_build_openai ;;
        7) demo_interactive ;;
        a|A) run_all ;;
        q|Q) echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid choice" ;;
    esac

    echo ""
    echo "Press Enter to continue..."
    read
    clear
done
