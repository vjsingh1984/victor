#!/bin/bash
# Production Deployment Script for Victor AI
# Usage: scripts/ci/deploy_production.sh <environment> [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}INFO${NC}: $1"; }
print_success() { echo -e "${GREEN}✓${NC}: $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC}: $1"; }
print_error() { echo -e "${RED}✗${NC}: $1"; }

# Default values
ENVIRONMENT=""
VERSION=""
DRY_RUN=false
SKIP_TESTS=false
SKIP_BACKUP=false
FORCE=false
HELM_CHART="./config/helm/victor"
K8S_MANIFESTS="./config/k8s"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    staging|production)
      ENVIRONMENT="$1"
      shift
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=true
      shift
      ;;
    --skip-backup)
      SKIP_BACKUP=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --helm)
      DEPLOYMENT_METHOD="helm"
      shift
      ;;
    --kubectl)
      DEPLOYMENT_METHOD="kubectl"
      shift
      ;;
    --help|-h)
      cat << EOF
Usage: $0 <environment> [options]

Arguments:
  environment     Deployment environment (staging|production)

Options:
  --version VERSION      Version to deploy (default: from pyproject.toml)
  --dry-run              Show what would be deployed without actually deploying
  --skip-tests           Skip pre-deployment tests (not recommended)
  --skip-backup          Skip backup creation (not recommended)
  --force                Force deployment even with warnings
  --helm                 Use Helm for deployment (default)
  --kubectl              Use kubectl for deployment
  --help                 Show this help

Environment Variables:
  KUBE_CONFIG            Path to kubeconfig file
  REGISTRY               Docker registry (default: ghcr.io/vjsingh1984/victor)
  NAMESPACE              Kubernetes namespace (default: victor)

Examples:
  # Deploy to staging with dry-run
  $0 staging --dry-run

  # Deploy specific version to production
  $0 production --version v0.5.0

  # Deploy with Helm
  $0 production --helm

  # Force deployment skipping tests (not recommended)
  $0 production --force --skip-tests
EOF
      exit 0
      ;;
    *)
      print_error "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate environment
if [ -z "$ENVIRONMENT" ]; then
  print_error "Environment must be specified (staging|production)"
  exit 1
fi

# Set deployment method
DEPLOYMENT_METHOD="${DEPLOYMENT_METHOD:-helm}"

# Get version if not specified
if [ -z "$VERSION" ]; then
  VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml'))['project']['version'])")
  print_info "Version from pyproject.toml: $VERSION"
fi

# Set environment-specific variables
case $ENVIRONMENT in
  staging)
    NAMESPACE="${NAMESPACE:-victor-staging}"
    REPLICAS="${REPLICAS:-1}"
    REGISTRY="${REGISTRY:-ghcr.io/vjsingh1984/victor}"
    ;;
  production)
    NAMESPACE="${NAMESPACE:-victor}"
    REPLICAS="${REPLICAS:-3}"
    REGISTRY="${REGISTRY:-ghcr.io/vjsingh1984/victor}"

    # Safety check for production
    if [ "$FORCE" = false ]; then
      print_warning "Deploying to PRODUCTION environment"
      read -p "Are you sure? (yes/no): " confirm
      if [ "$confirm" != "yes" ]; then
        print_info "Deployment cancelled"
        exit 0
      fi
    fi
    ;;
esac

IMAGE="${REGISTRY}:${VERSION}"

# Print deployment summary
print_info "Deployment Summary"
echo "  Environment: $ENVIRONMENT"
echo "  Version: $VERSION"
echo "  Image: $IMAGE"
echo "  Namespace: $NAMESPACE"
echo "  Replicas: $REPLICAS"
echo "  Method: $DEPLOYMENT_METHOD"
echo "  Dry Run: $DRY_RUN"
echo ""

if [ "$DRY_RUN" = true ]; then
  print_info "This is a dry run. No actual deployment will occur."
  echo ""
  print_info "Deployment steps:"
  echo "  1. Pre-deployment checks"
  echo "  2. Run tests (unless --skip-tests)"
  echo "  3. Create backup (unless --skip-backup)"
  echo "  4. Build/push Docker image: $IMAGE"
  echo "  5. Deploy using $DEPLOYMENT_METHOD"
  echo "  6. Wait for rollout"
  echo "  7. Run smoke tests"
  echo "  8. Verify deployment"
  exit 0
fi

# =============================================================================
# Step 1: Pre-deployment checks
# =============================================================================
print_info "Step 1: Pre-deployment checks"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
  print_error "kubectl not found. Please install it first."
  exit 1
fi

# Check if helm is available (if using helm)
if [ "$DEPLOYMENT_METHOD" = "helm" ] && ! command -v helm &> /dev/null; then
  print_error "helm not found. Please install it first or use --kubectl"
  exit 1
fi

# Check kubeconfig
if [ -z "$KUBE_CONFIG" ] && [ ! -f "$HOME/.kube/config" ]; then
  print_error "Kubeconfig not found. Set KUBE_CONFIG or configure ~/.kube/config"
  exit 1
fi

# Set KUBECONFIG
if [ -n "$KUBE_CONFIG" ]; then
  export KUBECONFIG="$KUBE_CONFIG"
fi

# Check cluster connectivity
print_info "Checking cluster connectivity..."
if ! kubectl cluster-info &> /dev/null; then
  print_error "Cannot connect to Kubernetes cluster"
  exit 1
fi
print_success "Cluster connectivity OK"

# =============================================================================
# Step 2: Run tests
# =============================================================================
if [ "$SKIP_TESTS" = false ]; then
  print_info "Step 2: Running tests"

  if [ -f "scripts/ci/run_tests.sh" ]; then
    bash scripts/ci/run_tests.sh
    print_success "Tests passed"
  else
    print_warning "Test script not found, skipping tests"
  fi
else
  print_warning "Skipping tests (--skip-tests)"
fi

# =============================================================================
# Step 3: Create backup
# =============================================================================
if [ "$SKIP_BACKUP" = false ]; then
  print_info "Step 3: Creating backup"

  BACKUP_DIR="./backups/${ENVIRONMENT}/$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BACKUP_DIR"

  # Backup current deployment
  kubectl get deployment victor-api -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/deployment.yaml"
  kubectl get configmap -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/configmaps.yaml"
  kubectl get secret -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/secrets.yaml"

  print_success "Backup created: $BACKUP_DIR"
else
  print_warning "Skipping backup (--skip-backup)"
fi

# =============================================================================
# Step 4: Deploy
# =============================================================================
print_info "Step 4: Deploying version $VERSION"

if [ "$DEPLOYMENT_METHOD" = "helm" ]; then
  # Deploy using Helm
  print_info "Using Helm for deployment"

  # Check if release exists
  if helm list -n "$NAMESPACE" | grep -q "victor"; then
    print_info "Upgrading existing Helm release"
    helm upgrade victor "$HELM_CHART" \
      --namespace "$NAMESPACE" \
      --set image.repository="$REGISTRY" \
      --set image.tag="$VERSION" \
      --set replicaCount="$REPLICAS" \
      --wait \
      --timeout 10m \
      --atomic
  else
    print_info "Installing new Helm release"
    helm install victor "$HELM_CHART" \
      --namespace "$NAMESPACE" \
      --create-namespace \
      --set image.repository="$REGISTRY" \
      --set image.tag="$VERSION" \
      --set replicaCount="$REPLICAS" \
      --wait \
      --timeout 10m \
      --atomic
  fi
else
  # Deploy using kubectl
  print_info "Using kubectl for deployment"

  # Update image in deployment
  kubectl set image deployment/victor-api \
    victor-api="$IMAGE" \
    -n "$NAMESPACE" \
    --record

  # Scale to desired replica count
  kubectl scale deployment/victor-api \
    --replicas="$REPLICAS" \
    -n "$NAMESPACE"
fi

print_success "Deployment triggered"

# =============================================================================
# Step 5: Wait for rollout
# =============================================================================
print_info "Step 5: Waiting for rollout"

kubectl rollout status deployment/victor-api \
  -n "$NAMESPACE" \
  --timeout=10m

print_success "Rollout complete"

# =============================================================================
# Step 6: Verify deployment
# =============================================================================
print_info "Step 6: Verifying deployment"

# Get pod status
PODS=$(kubectl get pods -n "$NAMESPACE" -l app=victor-api --no-headers | wc -l)
READY_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=victor-api --no-headers | grep -c "Running" || echo "0")

echo "  Pods: $READY_PODS/$PODS ready"

if [ "$READY_PODS" -lt "$REPLICAS" ]; then
  print_warning "Not all pods are ready"
  kubectl get pods -n "$NAMESPACE" -l app=victor-api
fi

# Port-forward and test health endpoint
print_info "Testing health endpoint"

kubectl port-forward -n "$NAMESPACE" svc/victor-api 8000:80 > /dev/null 2>&1 &
PF_PID=$!
sleep 10

# Health checks
if curl -f http://localhost:8000/health/live > /dev/null 2>&1; then
  print_success "Liveness probe OK"
else
  print_error "Liveness probe failed"
  kill $PF_PID
  exit 1
fi

if curl -f http://localhost:8000/health/ready > /dev/null 2>&1; then
  print_success "Readiness probe OK"
else
  print_error "Readiness probe failed"
  kill $PF_PID
  exit 1
fi

# Cleanup port-forward
kill $PF_PID

# =============================================================================
# Step 7: Run smoke tests
# =============================================================================
if [ -f "scripts/ci/smoke_test.sh" ]; then
  print_info "Step 7: Running smoke tests"

  if bash scripts/ci/smoke_test.sh "$ENVIRONMENT"; then
    print_success "Smoke tests passed"
  else
    print_error "Smoke tests failed"

    # Rollback on failure
    print_warning "Rolling back deployment..."
    kubectl rollout undo deployment/victor-api -n "$NAMESPACE"
    exit 1
  fi
fi

# =============================================================================
# Deployment Summary
# =============================================================================
print_success "Deployment to $ENVIRONMENT completed successfully!"

echo ""
print_info "Deployment Details"
echo "  Environment: $ENVIRONMENT"
echo "  Version: $VERSION"
echo "  Image: $IMAGE"
echo "  Namespace: $NAMESPACE"
echo "  Replicas: $REPLICAS"
echo "  Backup: $BACKUP_DIR"
echo ""
print_info "Next Steps"
echo "  - Monitor logs: kubectl logs -f deployment/victor-api -n $NAMESPACE"
echo "  - Check metrics: http://prometheus.victor.example.com"
echo "  - View dashboards: http://grafana.victor.example.com"
echo ""
print_success "To rollback if needed:"
echo "  kubectl rollout undo deployment/victor-api -n $NAMESPACE"
