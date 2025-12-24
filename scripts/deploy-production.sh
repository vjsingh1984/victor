#!/bin/bash
# Victor Production Deployment Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Victor Production Deployment Script  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose installed${NC}"

# Check .env.production file
if [ ! -f .env.production ]; then
    echo -e "${RED}Error: .env.production file not found${NC}"
    echo -e "${YELLOW}Please create .env.production from .env.production.example${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Environment file found${NC}"

# Load environment variables
set -a
source .env.production
set +a

# Validate required API keys (at least one provider)
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    if [ "$VICTOR_AIRGAPPED_MODE" != "true" ]; then
        echo -e "${RED}Error: No API keys configured and airgapped mode is disabled${NC}"
        echo -e "${YELLOW}Either set API keys or enable VICTOR_AIRGAPPED_MODE=true${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Configuration validated${NC}"

echo ""
echo -e "${YELLOW}Deployment Configuration:${NC}"
echo "  - Environment: production"
echo "  - API Port: ${VICTOR_API_PORT}"
echo "  - Metrics Port: ${VICTOR_METRICS_PORT}"
echo "  - Prometheus: http://localhost:${PROMETHEUS_PORT}"
echo "  - Grafana: http://localhost:${GRAFANA_PORT}"
echo "  - Air-gapped Mode: ${VICTOR_AIRGAPPED_MODE}"
echo ""

# Ask for confirmation
read -p "Proceed with deployment? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deployment cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose -f docker-compose.production.yml build --no-cache

echo ""
echo -e "${YELLOW}Starting services...${NC}"
docker-compose -f docker-compose.production.yml up -d

echo ""
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check Victor health
echo -n "Checking Victor service... "
if docker exec victor-production victor --version &> /dev/null; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${RED}✗ Failed${NC}"
    echo -e "${RED}Service health check failed. Check logs: docker-compose -f docker-compose.production.yml logs victor${NC}"
    exit 1
fi

# Check Prometheus
echo -n "Checking Prometheus... "
if curl -sf http://localhost:${PROMETHEUS_PORT}/-/healthy > /dev/null; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${YELLOW}⚠ Not responding (may still be starting)${NC}"
fi

# Check Grafana
echo -n "Checking Grafana... "
if curl -sf http://localhost:${GRAFANA_PORT}/api/health > /dev/null; then
    echo -e "${GREEN}✓ Healthy${NC}"
else
    echo -e "${YELLOW}⚠ Not responding (may still be starting)${NC}"
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Deployment Successful! 🎉         ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Access your services:${NC}"
echo "  - Victor API: http://localhost:${VICTOR_API_PORT}"
echo "  - Prometheus: http://localhost:${PROMETHEUS_PORT}"
echo "  - Grafana: http://localhost:${GRAFANA_PORT}"
echo "    Username: ${GRAFANA_ADMIN_USER}"
echo "    Password: (from .env.production)"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  - View logs: docker-compose -f docker-compose.production.yml logs -f"
echo "  - Stop services: docker-compose -f docker-compose.production.yml down"
echo "  - Restart: docker-compose -f docker-compose.production.yml restart"
echo "  - Execute Victor: docker exec -it victor-production victor chat --no-tui 'Hello'"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Open Grafana at http://localhost:${GRAFANA_PORT}"
echo "  2. Navigate to Victor Overview dashboard"
echo "  3. Monitor metrics and set up alerts"
echo "  4. Test the deployment with sample requests"
echo ""
