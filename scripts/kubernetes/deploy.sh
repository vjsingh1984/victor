#!/bin/bash
#############################################################################
# Kubernetes Deployment Script for Victor AI
#
# Features:
# - Helm chart installation
# - ConfigMap/Secret management
# - Rolling updates
# - Pod verification
#
# Usage: ./deploy.sh [staging|production] [--upgrade] [--install]
#############################################################################

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
HELM_DIR="${PROJECT_ROOT}/deploy/helm"
NAMESPACE="${KUBE_NAMESPACE:-victor}"
RELEASE_NAME="${RELEASE_NAME:-victor}"
CHART_NAME="${RELEASE_NAME}"

# Kubernetes configuration
CONTEXT="${KUBE_CONTEXT:-}"
VALUES_FILE="${VALUES_FILE:-}"
DEPLOYMENT_ENV="${1:-staging}"

# Flags
UPGRADE=false
INSTALL=false
DRY_RUN=false
WAIT_TIMEOUT=300

# Logging functions
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

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Trap errors
trap 'error_exit "Deployment failed at line $LINENO"' ERR

#############################################################################
# Parse Arguments
#############################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            staging|production)
                DEPLOYMENT_ENV="$1"
                shift
                ;;
            --upgrade)
                UPGRADE=true
                shift
                ;;
            --install)
                INSTALL=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --context)
                CONTEXT="$2"
                shift 2
                ;;
            --values)
                VALUES_FILE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [staging|production] [options]"
                echo ""
                echo "Arguments:"
                echo "  staging       Deploy to staging environment"
                echo "  production    Deploy to production environment"
                echo ""
                echo "Options:"
                echo "  --upgrade     Upgrade existing release"
                echo "  --install     Install new release (default)"
                echo "  --dry-run     Simulate deployment without applying changes"
                echo "  --namespace   Kubernetes namespace (default: victor)"
                echo "  --context     Kubernetes context"
                echo "  --values      Custom values file"
                echo "  --help        Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 staging --install"
                echo "  $0 production --upgrade --values custom-values.yaml"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

#############################################################################
# Validate Kubernetes Environment
#############################################################################
validate_kubernetes() {
    log_info "Validating Kubernetes environment..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error_exit "kubectl is not installed"
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        error_exit "helm is not installed"
    fi

    # Set context if provided
    if [ -n "$CONTEXT" ]; then
        kubectl config use-context "$CONTEXT"
        log_info "Using context: $CONTEXT"
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi

    log_success "Kubernetes environment validated"

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace ${NAMESPACE} does not exist. Creating..."
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace ${NAMESPACE} created"
    fi
}

#############################################################################
# Create or Update Secrets
#############################################################################
manage_secrets() {
    log_info "Managing Kubernetes secrets..."

    # Check if secret exists
    if kubectl get secret "victor-secrets" -n "$NAMESPACE" &> /dev/null; then
        log_info "Secret already exists. Skipping creation."
        log_info "To update secrets, use: kubectl create secret generic victor-secrets --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -"
    else
        log_info "Creating new secret..."

        if [ -f "${PROJECT_ROOT}/.env" ]; then
            kubectl create secret generic victor-secrets \
                --from-env-file="${PROJECT_ROOT}/.env" \
                -n "$NAMESPACE"

            log_success "Secret created from .env file"
        else
            log_warning ".env file not found. Creating empty secret..."
            kubectl create secret generic victor-secrets \
                --from-literal=VICTOR_PROFILE=production \
                -n "$NAMESPACE"

            log_warning "Please update the secret with actual values"
        fi
    fi
}

#############################################################################
# Deploy with Helm
#############################################################################
helm_deploy() {
    log_info "Deploying with Helm..."

    cd "${PROJECT_ROOT}"

    # Determine values files
    local values_args=()

    # Default values
    if [ -f "${HELM_DIR}/${CHART_NAME}/values.yaml" ]; then
        values_args+=("-f" "${HELM_DIR}/${CHART_NAME}/values.yaml")
    fi

    # Environment-specific values
    if [ -f "${HELM_DIR}/${CHART_NAME}/values-${DEPLOYMENT_ENV}.yaml" ]; then
        values_args+=("-f" "${HELM_DIR}/${CHART_NAME}/values-${DEPLOYMENT_ENV}.yaml")
    fi

    # Custom values file
    if [ -n "$VALUES_FILE" ] && [ -f "$VALUES_FILE" ]; then
        values_args+=("-f" "$VALUES_FILE")
    fi

    # Set namespace
    values_args+=("--namespace" "$NAMESPACE")

    # Dry run
    if [ "$DRY_RUN" = true ]; then
        values_args+=("--dry-run" "--debug")
    fi

    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "^${RELEASE_NAME}"; then
        log_info "Release ${RELEASE_NAME} exists. Performing upgrade..."

        helm upgrade "${RELEASE_NAME}" \
            "${HELM_DIR}/${CHART_NAME}" \
            "${values_args[@]}" \
            --wait \
            --timeout "${WAIT_TIMEOUT}s" \
            --atomic

        log_success "Release upgraded successfully"
    else
        log_info "Installing new release..."

        helm install "${RELEASE_NAME}" \
            "${HELM_DIR}/${CHART_NAME}" \
            "${values_args[@]}" \
            --wait \
            --timeout "${WAIT_TIMEOUT}s" \
            --create-namespace

        log_success "Release installed successfully"
    fi
}

#############################################################################
# Verify Deployment
#############################################################################
verify_deployment() {
    log_info "Verifying deployment..."

    # Wait for deployment to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait \
        --for=condition=ready \
        pod -l app="${RELEASE_NAME}" \
        -n "$NAMESPACE" \
        --timeout=300s

    # Get pod status
    log_info "Pod status:"
    kubectl get pods -n "$NAMESPACE" -l app="${RELEASE_NAME}"

    # Check deployment status
    log_info "Deployment status:"
    kubectl get deployment "${RELEASE_NAME}" -n "$NAMESPACE"

    # Get service status
    log_info "Service status:"
    kubectl get service "${RELEASE_NAME}" -n "$NAMESPACE"

    # Get ingress status (if exists)
    if kubectl get ingress "${RELEASE_NAME}" -n "$NAMESPACE" &> /dev/null; then
        log_info "Ingress status:"
        kubectl get ingress "${RELEASE_NAME}" -n "$NAMESPACE"
    fi

    log_success "Deployment verified successfully"
}

#############################################################################
# Run Health Checks
#############################################################################
run_health_checks() {
    log_info "Running health checks..."

    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service "${RELEASE_NAME}" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

    local service_host
    service_host=$(kubectl get service "${RELEASE_NAME}" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

    local endpoint=""

    if [ -n "$service_ip" ]; then
        endpoint="http://${service_ip}:8000"
    elif [ -n "$service_host" ]; then
        endpoint="http://${service_host}:8000"
    else
        # Use port-forward for local testing
        log_warning "No external endpoint found. Use port-forward for testing:"
        log_info "  kubectl port-forward service/${RELEASE_NAME} 8000:8000 -n $NAMESPACE"
        return
    fi

    log_info "Health check endpoint: ${endpoint}/health"

    # Wait for service to be accessible
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "${endpoint}/health" > /dev/null 2>&1; then
            log_success "Service is healthy"
            return
        fi

        log_info "Waiting for service to be ready... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done

    log_warning "Health check endpoint not accessible after $max_attempts attempts"
}

#############################################################################
# Display Deployment Info
#############################################################################
display_deployment_info() {
    log_info "Deployment Information:"
    echo ""
    echo "=========================================="
    echo "Release: ${RELEASE_NAME}"
    echo "Namespace: ${NAMESPACE}"
    echo "Environment: ${DEPLOYMENT_ENV}"
    echo "=========================================="
    echo ""
    echo "To check deployment status:"
    echo "  kubectl get pods -n ${NAMESPACE}"
    echo "  kubectl logs -f deployment/${RELEASE_NAME} -n ${NAMESPACE}"
    echo ""
    echo "To access the service:"
    echo "  kubectl port-forward service/${RELEASE_NAME} 8000:8000 -n ${NAMESPACE}"
    echo ""
    echo "To check logs:"
    echo "  kubectl logs -f deployment/${RELEASE_NAME} -n ${NAMESPACE}"
    echo "  kubectl logs -f deployment/${RELEASE_NAME} -n ${NAMESPACE} --all-containers=true"
    echo ""
    echo "To scale deployment:"
    echo "  kubectl scale deployment/${RELEASE_NAME} --replicas=3 -n ${NAMESPACE}"
    echo ""
}

#############################################################################
# Main Deployment Flow
#############################################################################
main() {
    log_info "Starting Kubernetes deployment for Victor AI..."
    log_info "Deployment started at $(date)"

    # Parse arguments
    parse_args "$@"

    # Validate Kubernetes
    validate_kubernetes

    # Manage secrets
    manage_secrets

    # Deploy with Helm
    helm_deploy

    # Verify deployment
    verify_deployment

    # Health checks
    run_health_checks

    # Display info
    display_deployment_info

    log_success "Kubernetes deployment completed successfully!"
    log_info "Deployment finished at $(date)"
}

# Run main deployment
main "$@"
