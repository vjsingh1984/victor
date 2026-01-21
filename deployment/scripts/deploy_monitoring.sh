#!/bin/bash
# Victor AI Monitoring Stack Deployment Script
# Production-ready monitoring deployment automation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/deployment/kubernetes/monitoring"
HELM_DIR="${PROJECT_ROOT}/deployment/helm/monitoring-chart"
NAMESPACE="victor-monitoring"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed. Please install helm first."
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."

    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_warning "Namespace ${NAMESPACE} already exists. Skipping creation."
    else
        kubectl create namespace "${NAMESPACE}"
        log_success "Namespace ${NAMESPACE} created"
    fi
}

deploy_with_kubectl() {
    log_info "Deploying monitoring stack with kubectl..."

    # Apply manifests in order
    local manifests=(
        "namespace.yaml"
        "prometheus-configmap.yaml"
        "prometheus-deployment.yaml"
        "grafana-configs.yaml"
        "grafana-deployment.yaml"
        "alertmanager-deployment.yaml"
        "otel-collector-deployment.yaml"
        "ingress.yaml"
    )

    for manifest in "${manifests[@]}"; do
        log_info "Applying ${manifest}..."
        if kubectl apply -f "${K8S_DIR}/${manifest}"; then
            log_success "Applied ${manifest}"
        else
            log_error "Failed to apply ${manifest}"
            exit 1
        fi
    done

    log_success "Monitoring stack deployed with kubectl"
}

deploy_with_helm() {
    log_info "Deploying monitoring stack with Helm..."

    # Check if chart exists
    if [[ ! -d "${HELM_DIR}" ]]; then
        log_error "Helm chart not found at ${HELM_DIR}"
        exit 1
    fi

    # Update dependencies
    log_info "Updating Helm dependencies..."
    helm dependency update "${HELM_DIR}" || true

    # Install or upgrade chart
    log_info "Installing/upgrading Helm release..."
    if helm upgrade --install victor-monitoring "${HELM_DIR}" \
        --namespace "${NAMESPACE}" \
        --create-namespace \
        --wait \
        --timeout 10m \
        --values "${HELM_DIR}/values.yaml" \
        "${HELM_EXTRA_ARGS:-}"; then
        log_success "Helm release deployed successfully"
    else
        log_error "Failed to deploy Helm release"
        exit 1
    fi
}

wait_for_pods() {
    log_info "Waiting for pods to be ready..."

    local timeout=300
    local elapsed=0

    while [[ ${elapsed} -lt ${timeout} ]]; do
        local not_ready=$(kubectl get pods -n "${NAMESPACE}" -o json | jq -r '.items[] | select(.status.phase != "Running" or (.status.conditions[] | select(.type == "Ready" and .status != "True")) | .metadata.name' | wc -l)

        if [[ ${not_ready} -eq 0 ]]; then
            log_success "All pods are ready"
            return 0
        fi

        log_info "Waiting for ${not_ready} pod(s) to be ready..."
        sleep 5
        elapsed=$((elapsed + 5))
    done

    log_warning "Timeout waiting for pods to be ready"
    return 1
}

display_access_info() {
    log_info "Monitoring stack access information:"
    echo ""
    echo "Grafana:"
    echo "  URL: http://grafana.victor-ai.com"
    echo "  Username: admin"
    echo "  Password: changeme123 (CHANGE THIS IN PRODUCTION)"
    echo ""
    echo "Prometheus:"
    echo "  URL: http://prometheus.victor-ai.com"
    echo ""
    echo "AlertManager:"
    echo "  URL: http://alertmanager.victor-ai.com"
    echo ""
    echo "Port-forwarding (if no ingress):"
    echo "  Grafana: kubectl port-forward -n ${NAMESPACE} svc/grafana 3000:3000"
    echo "  Prometheus: kubectl port-forward -n ${NAMESPACE} svc/prometheus 9090:9090"
    echo "  AlertManager: kubectl port-forward -n ${NAMESPACE} svc/alertmanager 9093:9093"
    echo ""
}

main() {
    log_info "Starting Victor AI monitoring stack deployment..."
    echo ""

    # Parse arguments
    DEPLOYMENT_METHOD="kubectl"
    SKIP_WAIT=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --helm)
                DEPLOYMENT_METHOD="helm"
                shift
                ;;
            --skip-wait)
                SKIP_WAIT=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --helm       Deploy using Helm chart instead of kubectl"
                echo "  --skip-wait  Skip waiting for pods to be ready"
                echo "  --help       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Main deployment flow
    check_prerequisites
    create_namespace

    if [[ "${DEPLOYMENT_METHOD}" == "helm" ]]; then
        deploy_with_helm
    else
        deploy_with_kubectl
    fi

    if [[ "${SKIP_WAIT}" == false ]]; then
        wait_for_pods
    fi

    display_access_info

    log_success "Monitoring stack deployment completed successfully!"
}

# Run main function
main "$@"
