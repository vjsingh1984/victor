#!/bin/bash
# setup_test_environment.sh
# Setup test environment for rollback testing
# Supports: kind, minikube, and Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_NAMESPACE="${TEST_NAMESPACE:-victor-ai-test}"
BLUE_NAMESPACE="${TEST_NAMESPACE}-blue"
GREEN_NAMESPACE="${TEST_NAMESPACE}-green"
CLUSTER_TYPE="${CLUSTER_TYPE:-kind}"  # kind, minikube, or docker-compose

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    log_info "Docker installed: $(docker --version)"

    # Check cluster tool
    case "$CLUSTER_TYPE" in
        kind)
            if ! command -v kind &> /dev/null; then
                log_error "kind not found. Please install kind first."
                log_info "Install: brew install kind (macOS) or curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64"
                exit 1
            fi
            log_info "kind installed: $(kind version)"
            ;;
        minikube)
            if ! command -v minikube &> /dev/null; then
                log_error "minikube not found. Please install minikube first."
                exit 1
            fi
            log_info "minikube installed: $(minikube version)"
            ;;
        docker-compose)
            if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
                log_error "docker-compose not found. Please install docker-compose first."
                exit 1
            fi
            log_info "docker-compose installed"
            ;;
        *)
            log_error "Unknown cluster type: $CLUSTER_TYPE"
            log_info "Supported types: kind, minikube, docker-compose"
            exit 1
            ;;
    esac

    # Check kubectl (not needed for docker-compose)
    if [ "$CLUSTER_TYPE" != "docker-compose" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl not found. Please install kubectl first."
            exit 1
        fi
        log_info "kubectl installed: $(kubectl version --client --short 2>/dev/null || echo 'version check failed')"
    fi

    # Check helm (optional)
    if command -v helm &> /dev/null; then
        log_info "helm installed: $(helm version --short 2>/dev/null || echo 'version check failed')"
    else
        log_warn "helm not found. Helm tests will be skipped."
    fi

    log_info "All prerequisites met!"
}

# Create kind cluster
setup_kind_cluster() {
    log_step "Setting up kind cluster..."

    # Check if cluster exists
    if kind get clusters | grep -q "^victor-test$"; then
        log_warn "Kind cluster 'victor-test' already exists"
        read -p "Delete and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deleting existing cluster..."
            kind delete cluster --name victor-test
        else
            log_info "Using existing cluster"
            return
        fi
    fi

    # Create cluster config
    cat > /tmp/kind-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: victor-test
nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 8080
        protocol: TCP
      - containerPort: 443
        hostPort: 8443
        protocol: TCP
  - role: worker
  - role: worker
EOF

    # Create cluster
    log_info "Creating kind cluster..."
    kind create cluster --config /tmp/kind-config.yaml

    # Verify cluster
    log_info "Verifying cluster..."
    kubectl cluster-info --context kind-victor-test
    kubectl get nodes

    log_info "Kind cluster created successfully!"
}

# Create minikube cluster
setup_minikube_cluster() {
    log_step "Setting up minikube cluster..."

    # Check if cluster exists
    if minikube status -p victor-test 2>/dev/null | grep -q "Running"; then
        log_warn "Minikube cluster 'victor-test' already running"
        read -p "Delete and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deleting existing cluster..."
            minikube delete -p victor-test
        else
            log_info "Using existing cluster"
            return
        fi
    fi

    # Create cluster
    log_info "Creating minikube cluster..."
    minikube start \
        -p victor-test \
        --driver=docker \
        --kubernetes-version=v1.28.0 \
        --nodes=3 \
        --cpus=2 \
        --memory=4096 \
        --ports=8080:80,8443:443

    # Verify cluster
    log_info "Verifying cluster..."
    kubectl get nodes

    log_info "Minikube cluster created successfully!"
}

# Setup Docker Compose environment
setup_docker_compose() {
    log_step "Setting up Docker Compose environment..."

    # Check if running
    if docker-compose -f "${PROJECT_ROOT}/deployment/docker/docker-compose.yml" ps | grep -q "Up"; then
        log_warn "Docker Compose services already running"
        read -p "Stop and restart? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Stopping existing services..."
            docker-compose -f "${PROJECT_ROOT}/deployment/docker/docker-compose.yml" down
        else
            log_info "Using existing services"
            return
        fi
    fi

    # Start services
    log_info "Starting Docker Compose services..."
    docker-compose -f "${PROJECT_ROOT}/deployment/docker/docker-compose.yml" up -d

    # Verify services
    log_info "Verifying services..."
    sleep 5
    docker-compose -f "${PROJECT_ROOT}/deployment/docker/docker-compose.yml" ps

    log_info "Docker Compose environment ready!"
}

# Install ingress controller
setup_ingress() {
    if [ "$CLUSTER_TYPE" = "docker-compose" ]; then
        log_info "Skipping ingress setup for Docker Compose"
        return
    fi

    log_step "Setting up ingress controller..."

    # Check if ingress already installed
    if kubectl get pods -n ingress-app 2>/dev/null | grep -q "Running"; then
        log_info "Ingress controller already installed"
        return
    fi

    # Install NGINX ingress controller
    log_info "Installing NGINX ingress controller..."
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.9.4/deploy/static/provider/kind/deploy.yaml

    # Wait for ingress to be ready
    log_info "Waiting for ingress controller to be ready..."
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=90s

    log_info "Ingress controller ready!"
}

# Create namespaces
setup_namespaces() {
    if [ "$CLUSTER_TYPE" = "docker-compose" ]; then
        log_info "Skipping namespace setup for Docker Compose"
        return
    fi

    log_step "Creating test namespaces..."

    # Create test namespaces
    kubectl create namespace "$TEST_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace "$BLUE_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace "$GREEN_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    log_info "Namespaces created: $TEST_NAMESPACE, $BLUE_NAMESPACE, $GREEN_NAMESPACE"
}

# Deploy baseline version
deploy_baseline() {
    if [ "$CLUSTER_TYPE" = "docker-compose" ]; then
        log_info "Docker Compose baseline deployment handled by docker-compose up"
        return
    fi

    log_step "Deploying baseline version (0.5.0)..."

    # Build deployment manifests
    cd "$PROJECT_ROOT"

    # Use kustomize to build manifests
    if [ -f "deployment/kubernetes/overlays/staging/kustomization.yaml" ]; then
        log_info "Building staging manifests..."
        kubectl kustomize deployment/kubernetes/overlays/staging > /tmp/victor-baseline.yaml
    else
        log_warn "Staging overlay not found, using base"
        kubectl kustomize deployment/kubernetes/base > /tmp/victor-baseline.yaml
    fi

    # Apply baseline version
    log_info "Applying baseline deployment..."
    kubectl apply -f /tmp/victor-baseline.yaml -n "$GREEN_NAMESPACE"

    # Wait for rollout
    log_info "Waiting for baseline rollout to complete..."
    kubectl rollout status deployment/victor-ai -n "$GREEN_NAMESPACE" --timeout=120s

    # Verify
    log_info "Verifying baseline deployment..."
    kubectl get pods -n "$GREEN_NAMESPACE"
    kubectl get services -n "$GREEN_NAMESPACE"

    log_info "Baseline version deployed successfully!"
}

# Create ConfigMaps and Secrets
setup_config() {
    if [ "$CLUSTER_TYPE" = "docker-compose" ]; then
        log_info "Docker Compose config handled by .env file"
        return
    fi

    log_step "Setting up ConfigMaps and Secrets..."

    # Create ConfigMap
    cat > /tmp/victor-config.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: victor-ai-config
  namespace: ${TEST_NAMESPACE}
data:
  event-bus-backend: "memory"
  checkpoint-backend: "memory"
  log-format: "json"
  log-level: "INFO"
  max-workers: "4"
  tool-selection-strategy: "hybrid"
  cache-size: "1000"
  enable-telemetry: "false"
EOF

    kubectl apply -f /tmp/victor-config.yaml

    # Create placeholder secret (for testing only)
    cat > /tmp/victor-secrets.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: victor-ai-secrets
  namespace: ${TEST_NAMESPACE}
type: Opaque
stringData:
  database-url: "postgresql://test:test@localhost:5432/victor_test"
  redis-url: "redis://localhost:6379/0"
  anthropic-api-key: "test-key-for-testing-only"
  openai-api-key: "test-key-for-testing-only"
EOF

    kubectl apply -f /tmp/victor-secrets.yaml

    # Copy to other namespaces
    kubectl apply -f /tmp/victor-config.yaml --namespace="$BLUE_NAMESPACE"
    kubectl apply -f /tmp/victor-secrets.yaml --namespace="$GREEN_NAMESPACE"

    log_info "ConfigMaps and Secrets created!"
}

# Run health checks
run_health_checks() {
    log_step "Running health checks..."

    if [ "$CLUSTER_TYPE" = "docker-compose" ]; then
        # Check Docker Compose services
        log_info "Checking Docker Compose services..."
        if curl -f http://localhost:8000/health &>/dev/null; then
            log_info "Health check passed!"
        else
            log_warn "Health check failed (services may still be starting)"
        fi
        return
    fi

    # Check Kubernetes pods
    log_info "Checking pods in $GREEN_NAMESPACE..."
    kubectl get pods -n "$GREEN_NAMESPACE"

    # Port forward to test health endpoint
    log_info "Setting up port forward for health check..."
    kubectl port-forward -n "$GREEN_NAMESPACE" svc/victor-ai 8080:80 &
    PF_PID=$!
    sleep 3

    # Check health endpoint
    if curl -f http://localhost:8080/health &>/dev/null; then
        log_info "Health check passed!"
    else
        log_warn "Health check failed (services may still be starting)"
    fi

    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
}

# Print test environment info
print_environment_info() {
    log_step "Test Environment Information"

    echo ""
    echo "Cluster Type: $CLUSTER_TYPE"
    echo "Test Namespace: $TEST_NAMESPACE"
    echo "Blue Namespace: $BLUE_NAMESPACE"
    echo "Green Namespace: $GREEN_NAMESPACE"
    echo ""

    if [ "$CLUSTER_TYPE" = "docker-compose" ]; then
        echo "Docker Compose Services:"
        docker-compose -f "${PROJECT_ROOT}/deployment/docker/docker-compose.yml" ps
        echo ""
        echo "Access Victor AI at: http://localhost:8000"
    else
        echo "Kubernetes Context: $(kubectl config current-context)"
        echo ""
        echo "Namespaces:"
        kubectl get namespaces | grep -E "$TEST_NAMESPACE|$BLUE_NAMESPACE|$GREEN_NAMESPACE"
        echo ""
        echo "Pods in $GREEN_NAMESPACE:"
        kubectl get pods -n "$GREEN_NAMESPACE"
        echo ""
        echo "To access Victor AI:"
        echo "  kubectl port-forward -n $GREEN_NAMESPACE svc/victor-ai 8080:80"
        echo "  Then access at: http://localhost:8080"
    fi
    echo ""
    echo "Test environment ready!"
}

# Main function
main() {
    log_info "Starting test environment setup..."
    echo ""

    # Check prerequisites
    check_prerequisites
    echo ""

    # Setup cluster based on type
    case "$CLUSTER_TYPE" in
        kind)
            setup_kind_cluster
            setup_ingress
            setup_namespaces
            setup_config
            deploy_baseline
            ;;
        minikube)
            setup_minikube_cluster
            setup_ingress
            setup_namespaces
            setup_config
            deploy_baseline
            ;;
        docker-compose)
            setup_docker_compose
            ;;
    esac
    echo ""

    # Run health checks
    run_health_checks
    echo ""

    # Print environment info
    print_environment_info
}

# Run main function
main "$@"
