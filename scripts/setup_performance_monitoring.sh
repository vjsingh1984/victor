#!/bin/bash
# Performance Monitoring Setup Script
# This script deploys the Victor AI performance monitoring system

set -e

echo "Victor AI Performance Monitoring Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE=${NAMESPACE:-"victor-monitoring"}
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "Configuration:"
echo "  Namespace: ${NAMESPACE}"
echo "  Project Root: ${PROJECT_ROOT}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl not found${NC}"
    echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot access Kubernetes cluster${NC}"
    echo "Please configure kubectl to access your cluster"
    exit 1
fi

echo -e "${GREEN}✓ Kubernetes cluster accessible${NC}"
echo ""

# Step 1: Deploy performance alerts
echo "Step 1: Deploying performance alert rules..."
kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/monitoring/performance-alerts.yaml"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Performance alerts deployed${NC}"
else
    echo -e "${RED}✗ Failed to deploy performance alerts${NC}"
    exit 1
fi
echo ""

# Step 2: Update Prometheus configuration
echo "Step 2: Updating Prometheus configuration..."
kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/monitoring/prometheus-configmap.yaml"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Prometheus configuration updated${NC}"
else
    echo -e "${RED}✗ Failed to update Prometheus configuration${NC}"
    exit 1
fi
echo ""

# Step 3: Deploy Grafana dashboard
echo "Step 3: Deploying Grafana dashboard..."
if [ -f "${PROJECT_ROOT}/deployment/kubernetes/monitoring/dashboards/victor-performance.json" ]; then
    # Create dashboard configmap if it doesn't exist
    kubectl create configmap victor-performance-dashboard \
        --from-file="${PROJECT_ROOT}/deployment/kubernetes/monitoring/dashboards/victor-performance.json" \
        --namespace="${NAMESPACE}" \
        --dry-run=client -o yaml | kubectl apply -f -

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Grafana dashboard deployed${NC}"
    else
        echo -e "${RED}✗ Failed to deploy Grafana dashboard${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Dashboard file not found, skipping...${NC}"
fi
echo ""

# Step 4: Restart Prometheus to pick up new rules
echo "Step 4: Restarting Prometheus..."
kubectl rollout restart deployment/prometheus -n "${NAMESPACE}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Prometheus restarting${NC}"
else
    echo -e "${YELLOW}⚠ Could not restart Prometheus (may not be deployed)${NC}"
fi
echo ""

# Step 5: Wait for Prometheus to be ready
echo "Step 5: Waiting for Prometheus to be ready..."
kubectl wait --for=condition=available --timeout=60s \
    deployment/prometheus -n "${NAMESPACE}" 2>/dev/null || \
    echo -e "${YELLOW}⚠ Prometheus deployment not found or not ready${NC}"
echo ""

# Verification
echo "Verifying installation..."
echo ""

# Check if alerts are loaded
ALERT_COUNT=$(kubectl get configmap prometheus-config -n "${NAMESPACE}" \
    -o jsonpath='{.data.performance_alerts\.yml}' 2>/dev/null | grep -c "alert:" || echo "0")

if [ "$ALERT_COUNT" -gt "0" ]; then
    echo -e "${GREEN}✓ Performance alerts configured (${ALERT_COUNT} alerts)${NC}"
else
    echo -e "${YELLOW}⚠ No performance alerts found in Prometheus config${NC}"
fi

# Check if dashboard is deployed
if kubectl get configmap victor-performance-dashboard -n "${NAMESPACE}" &> /dev/null; then
    echo -e "${GREEN}✓ Grafana dashboard deployed${NC}"
else
    echo -e "${YELLOW}⚠ Grafana dashboard not found${NC}"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Port-forward to Grafana:"
echo "   kubectl port-forward -n ${NAMESPACE} svc/grafana 3000:3000"
echo "   open http://localhost:3000"
echo ""
echo "2. Import the dashboard:"
echo "   - Go to Dashboards -> Import"
echo "   - Upload: deployment/kubernetes/monitoring/dashboards/victor-performance.json"
echo "   - Or use configmap: victor-performance-dashboard"
echo ""
echo "3. View Prometheus alerts:"
echo "   kubectl port-forward -n ${NAMESPACE} svc/prometheus 9090:9090"
echo "   open http://localhost:9090/alerts"
echo ""
echo "4. Check AlertManager:"
echo "   kubectl port-forward -n ${NAMESPACE} svc/alertmanager 9093:9093"
echo "   open http://localhost:9093"
echo ""
echo "5. Test performance API (if Victor AI is running):"
echo "   kubectl port-forward -n victor-production svc/victor-api 8000:8000"
echo "   curl http://localhost:8000/api/performance/summary"
echo ""
echo "For more information, see: docs/observability/performance_monitoring.md"
echo ""
