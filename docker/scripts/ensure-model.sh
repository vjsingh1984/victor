#!/bin/bash
# Ensure Ollama model is available (pull if needed)
# Usage: ensure-model.sh <model_name> [size_description]
#        bash docker/scripts/ensure-model.sh <model_name> [size_description]
#
# Examples:
#   ensure-model.sh qwen2.5-coder:1.5b "1 GB"
#   ensure-model.sh llama3.1:8b "4.9 GB"
#   ensure-model.sh qwen3-coder:30b "18 GB"
#
# From inside container:
#   bash /app/docker/scripts/ensure-model.sh qwen2.5-coder:1.5b "1 GB"

set -e

# Parse arguments
MODEL_NAME="${1:-qwen2.5-coder:1.5b}"
SIZE_DESC="${2:-1 GB}"

# Load colors if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/colors.sh" ]; then
    source "$SCRIPT_DIR/colors.sh"
else
    # Fallback: define empty color codes
    GREEN='' BLUE='' YELLOW='' CYAN='' RED='' BOLD='' NC=''
fi

echo -e "${CYAN}📦 Checking for $MODEL_NAME model...${NC}"
echo ""

# Detect if we're inside Docker container or running from host
if [ -f "/.dockerenv" ] || [ -n "$DOCKER_CONTAINER" ]; then
    # Inside container: use ollama CLI directly
    OLLAMA_CMD="ollama"
else
    # On host: use docker-compose exec
    OLLAMA_CMD="docker-compose exec ollama ollama"
fi

# Check if model exists
if $OLLAMA_CMD list 2>/dev/null | grep -q "$MODEL_NAME"; then
    echo -e "${GREEN}✓ $MODEL_NAME already available${NC}"
    echo ""
    exit 0
fi

# Model not found, pull it
echo -e "${YELLOW}⚠ Model not found. Pulling $MODEL_NAME ($SIZE_DESC)...${NC}"
echo -e "${DIM}   This may take 1-5 minutes depending on model size and connection.${NC}"
echo ""

# Pull the model
if $OLLAMA_CMD pull "$MODEL_NAME"; then
    echo ""
    echo -e "${GREEN}✓ $MODEL_NAME ready${NC}"
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}✗ Failed to pull $MODEL_NAME${NC}"
    echo -e "${YELLOW}  Troubleshooting:${NC}"
    echo -e "${YELLOW}    1. Check Ollama is running: docker-compose ps ollama${NC}"
    echo -e "${YELLOW}    2. Check Ollama logs: docker-compose logs ollama${NC}"
    echo -e "${YELLOW}    3. Check internet connection (if not air-gapped)${NC}"
    echo ""
    exit 1
fi
