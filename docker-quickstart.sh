#!/bin/bash
# Victor Docker Quick Start Script
# One-command setup for air-gapped semantic tool selection

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}${BLUE}Victor Docker Quick Start${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}❌ docker-compose not found. Please install docker-compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo ""

# Step 1: Build Victor image
echo -e "${BOLD}Step 1/4: Building Victor image${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "This includes:"
echo "  • Pre-downloading all-MiniLM-L12-v2 embedding model (120MB)"
echo "  • Pre-computing tool embeddings cache (31 tools)"
echo "  • Installing Victor with all dependencies"
echo ""

docker-compose build victor

echo ""
echo -e "${GREEN}✓ Victor image built successfully${NC}"
echo ""

# Step 2: Start Ollama
echo -e "${BOLD}Step 2/4: Starting Ollama server${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

docker-compose --profile demo up -d ollama

echo ""
echo -e "${CYAN}⏳ Waiting for Ollama to be ready...${NC}"
until docker-compose exec ollama curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "   Waiting..."
    sleep 2
done
echo -e "${GREEN}✓ Ollama is ready${NC}"
echo ""

# Step 3: Pull required models
echo -e "${BOLD}Step 3/4: Pulling Ollama models${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Lightweight distribution mode:"
echo "  • Default model: qwen2.5-coder:1.5b (~1 GB)"
echo "  • Optional: qwen2.5-coder:7b (~4.7 GB) for better quality"
echo ""

# Check if models exist
HAS_QWEN25_1_5B=$(docker-compose exec ollama ollama list | grep -c "qwen2.5-coder:1.5b" || true)

if [ "$HAS_QWEN25_1_5B" -eq "0" ]; then
    echo -e "${CYAN}📦 Pulling qwen2.5-coder:1.5b (1 GB, default model)...${NC}"
    echo "   This should take 1-3 minutes."
    docker-compose exec ollama ollama pull qwen2.5-coder:1.5b
    echo -e "${GREEN}✓ qwen2.5-coder:1.5b ready${NC}"
else
    echo -e "${GREEN}✓ qwen2.5-coder:1.5b already available${NC}"
fi

echo ""
echo -e "${YELLOW}Optional:${NC} Pull additional models for better quality:"
echo "  docker-compose exec ollama ollama pull qwen2.5-coder:7b    # 4.7 GB"
echo "  docker-compose exec ollama ollama pull qwen3-coder:30b     # 18 GB"
echo ""
echo -e "${GREEN}✓ Essential models ready${NC}"
echo ""

# Step 4: Verify setup
echo -e "${BOLD}Step 4/4: Verifying setup${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${CYAN}Testing Victor...${NC}"
docker-compose run --rm victor victor --version

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""

# Print summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}${GREEN}🎉 Victor is ready!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Quick Commands:"
echo ""
echo "  # Interactive mode"
echo -e "  ${CYAN}docker-compose run --rm victor${NC}"
echo ""
echo "  # One-shot command"
echo -e "  ${CYAN}docker-compose run --rm victor victor main \"Write hello world\"${NC}"
echo ""
echo "  # Run demo"
echo -e "  ${CYAN}docker-compose run --rm victor bash /app/docker/demo-semantic-tools.sh${NC}"
echo ""
echo "  # Use fast profile (7B model)"
echo -e "  ${CYAN}docker-compose run --rm victor victor --profile fast \"Write a function\"${NC}"
echo ""
echo "  # List all profiles"
echo -e "  ${CYAN}docker-compose run --rm victor victor profiles${NC}"
echo ""
echo "Configuration (Lightweight Distribution):"
echo "  • Default model: qwen2.5-coder:1.5b (~1 GB)"
echo "  • Embedding model: all-MiniLM-L12-v2 (120MB)"
echo "  • Tool embeddings: Pre-computed (31 tools)"
echo "  • Semantic selection: Enabled (threshold: 0.15, top-5)"
echo "  • Total size: ~2.5 GB (Victor + Ollama + model)"
echo ""
echo "Next Steps:"
echo "  1. Try the demo: docker-compose run --rm victor bash /app/docker/demo-semantic-tools.sh"
echo "  2. Start interactive session: docker-compose run --rm victor"
echo "  3. Read the docs: cat AIR_GAPPED_TOOL_CALLING_SOLUTION.md"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
