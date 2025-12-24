#!/bin/bash
# Victor Production Rollback Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}╔═══════════════════════════════════════╗${NC}"
echo -e "${RED}║    Victor Production Rollback         ║${NC}"
echo -e "${RED}╚═══════════════════════════════════════╝${NC}"
echo ""

# Check if running
if ! docker ps | grep -q victor-production; then
    echo -e "${RED}Error: Production deployment not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Current deployment status:${NC}"
docker ps --filter "name=victor-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

echo -e "${RED}WARNING: This will stop and remove the current deployment!${NC}"
read -p "Are you sure you want to rollback? (yes/NO) " -r
if [[ ! $REPLY == "yes" ]]; then
    echo -e "${YELLOW}Rollback cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Step 1: Backing up current volumes...${NC}"
BACKUP_DIR="./backups/rollback-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup Victor data
echo "  - Backing up Victor data..."
docker run --rm -v victor_production_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/victor-data.tar.gz -C /data .

# Backup Prometheus data
echo "  - Backing up Prometheus data..."
docker run --rm -v victor_prometheus_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/prometheus-data.tar.gz -C /data .

# Backup Grafana data
echo "  - Backing up Grafana data..."
docker run --rm -v victor_grafana_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/grafana-data.tar.gz -C /data .

echo -e "${GREEN}✓ Backups created in $BACKUP_DIR${NC}"

echo ""
echo -e "${YELLOW}Step 2: Stopping current deployment...${NC}"
docker-compose -f docker-compose.production.yml down

echo -e "${GREEN}✓ Services stopped${NC}"

echo ""
echo -e "${YELLOW}Step 3: Restore previous version (if available)...${NC}"
echo -e "${RED}Manual step required:${NC}"
echo "  1. Checkout previous git commit/tag:"
echo "     git checkout <previous-version-tag>"
echo "  2. Rebuild and deploy:"
echo "     ./scripts/deploy-production.sh"
echo ""
echo "  Or restore from backup:"
echo "     tar xzf $BACKUP_DIR/victor-data.tar.gz -C /path/to/restore"
echo ""

echo -e "${YELLOW}Backup location: $BACKUP_DIR${NC}"
echo -e "${GREEN}Rollback preparation complete${NC}"
