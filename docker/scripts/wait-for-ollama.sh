#!/bin/bash
# Wait for Ollama to be ready
# Usage: source docker/scripts/wait-for-ollama.sh [max_retries] [delay]
#        bash docker/scripts/wait-for-ollama.sh [max_retries] [delay]
#
# Examples:
#   source docker/scripts/wait-for-ollama.sh           # Default: 30 retries, 2s delay
#   source docker/scripts/wait-for-ollama.sh 60 1      # Custom: 60 retries, 1s delay
#   bash /app/docker/scripts/wait-for-ollama.sh        # From inside container

set -e

# Load colors if available (works both when sourced and executed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/colors.sh" ]; then
    source "$SCRIPT_DIR/colors.sh"
else
    # Fallback: define empty color codes
    GREEN='' BLUE='' YELLOW='' CYAN='' RED='' BOLD='' NC=''
fi

# Configuration
MAX_RETRIES=${1:-30}
DELAY=${2:-2}
OLLAMA_URL=${OLLAMA_HOST:-http://ollama:11434}

echo -e "${CYAN}⏳ Waiting for Ollama to be ready...${NC}"
echo -e "${DIM}   URL: $OLLAMA_URL${NC}"
echo -e "${DIM}   Max retries: $MAX_RETRIES, Delay: ${DELAY}s${NC}"
echo ""

# Wait loop
for i in $(seq 1 $MAX_RETRIES); do
    # Try to connect to Ollama API
    if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is ready!${NC}"
        echo ""

        # If sourced, return success; if executed, exit success
        if [ "$0" = "${BASH_SOURCE[0]}" ]; then
            exit 0
        else
            return 0
        fi
    fi

    echo -e "${YELLOW}   Attempt $i/$MAX_RETRIES... waiting ${DELAY}s${NC}"
    sleep $DELAY
done

# Failed to connect
echo ""
echo -e "${RED}✗ Ollama not available after $MAX_RETRIES attempts${NC}"
echo -e "${YELLOW}  Check: docker-compose logs ollama${NC}"
echo ""

# If sourced, return failure; if executed, exit failure
if [ "$0" = "${BASH_SOURCE[0]}" ]; then
    exit 1
else
    return 1
fi
