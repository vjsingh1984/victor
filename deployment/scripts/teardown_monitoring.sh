#!/bin/bash
# Victor AI Monitoring Stack Teardown Script
# Remove monitoring stack (for testing/redeployment)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NAMESPACE="victor-monitoring"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/deployment/kubernetes/monitoring"
HELM_DIR="${PROJECT_ROOT}/deployment/helm/monitoring-chart"

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

confirm_deletion() {
    log_warning "This will delete the entire monitoring stack!"
    log_warning "Namespace: ${NAMESPACE}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirmation

    if [[ "${confirmation}" != "yes" ]]; then
        log_info "Teardown cancelled"
        exit 0
    fi
}

teardown_with_helm() {
    log_info "Uninstalling Helm release..."

    if helm list -n "${NAMESPACE}" | grep -q "victor-monitoring"; then
        if helm uninstall victor-monitoring -n "${NAMESPACE}"; then
            log_success "Helm release uninstalled"
        else
            log_error "Failed to uninstall Helm release"
            return 1
        fi
    else
        log_warning "Helm release not found"
    fi
}

teardown_with_kubectl() {
    log_info "Deleting monitoring resources with kubectl..."

    local manifests=(
        "ingress.yaml"
        "otel-collector-deployment.yaml"
        "alertmanager-deployment.yaml"
        "grafana-deployment.yaml"
        "prometheus-deployment.yaml"
    )

    for manifest in "${manifests[@]}"; do
        local manifest_path="${K8S_DIR}/${manifest}"
        if [[ -f "${manifest_path}" ]]; then
            log_info "Deleting ${manifest}..."
            kubectl delete -f "${manifest_path}" --ignore-not-found=true || true
        fi
    done

    log_success "Monitoring resources deleted"
}

delete_persistent_volumes() {
    log_info "Deleting persistent volume claims..."

    local pvcs=("prometheus-pvc" "grafana-pvc" "alertmanager-pvc")

    for pvc in "${pvcs[@]}"; do
        if kubectl get pvc -n "${NAMESPACE}" "${pvc}" &> /dev/null; then
            log_info "Deleting ${pvc}..."
            kubectl delete pvc -n "${NAMESPACE}" "${pvc}" --ignore-not-found=true || true
        fi
    done

    log_success "Persistent volumes deleted"
}

delete_namespace() {
    log_info "Deleting namespace ${NAMESPACE}..."

    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        if kubectl delete namespace "${NAMESPACE}"; then
            log_success "Namespace deleted"
        else
            log_error "Failed to delete namespace"
            return 1
        fi
    else
        log_warning "Namespace not found"
    fi
}

cleanup_finalizers() {
    log_info "Checking for stuck resources with finalizers..."

    # Check for stuck pods
    local stuck_pods=$(kubectl get pods -n "${NAMESPACE}" -o json 2>/dev/null | jq -r '.items[] | select(.metadata.deletionTimestamp != null) | .metadata.name')

    if [[ -n "${stuck_pods}" ]]; then
        log_warning "Found pods stuck in deletion:"
        echo "${stuck_pods}"

        for pod in ${stuck_pods}; do
            log_info "Force deleting pod: ${pod}"
            kubectl delete pod -n "${NAMESPACE}" "${pod}" --force --grace-period=0 || true
        done
    fi
}

wait_for_cleanup() {
    log_info "Waiting for cleanup to complete..."

    local timeout=120
    local elapsed=0

    while [[ ${elapsed} -lt ${timeout} ]]; do
        if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
            log_success "Namespace deleted successfully"
            return 0
        fi

        local pods=$(kubectl get pods -n "${NAMESPACE}" -o json 2>/dev/null | jq '.items | length' 2>/dev/null || echo "0")
        log_info "Waiting... (${pods} pods remaining)"

        sleep 5
        elapsed=$((elapsed + 5))
    done

    log_warning "Cleanup timed out. Some resources may still exist."
    return 1
}

main() {
    log_info "Starting Victor AI monitoring stack teardown..."
    echo ""

    # Parse arguments
    DELETION_METHOD="kubectl"
    SKIP_CONFIRMATION=false
    KEEP_PVS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --helm)
                DELETION_METHOD="helm"
                shift
                ;;
            --yes)
                SKIP_CONFIRMATION=true
                shift
                ;;
            --keep-pvs)
                KEEP_PVS=true
                shift
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --helm          Uninstall using Helm (if deployed with Helm)"
                echo "  --yes           Skip confirmation prompt"
                echo "  --keep-pvs      Keep persistent volumes"
                echo "  --namespace NAMESPACE  Monitoring namespace (default: victor-monitoring)"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Confirm deletion
    if [[ "${SKIP_CONFIRMATION}" == false ]]; then
        confirm_deletion
    fi

    # Teardown flow
    if [[ "${DELETION_METHOD}" == "helm" ]]; then
        teardown_with_helm
    else
        teardown_with_kubectl
    fi

    # Delete persistent volumes unless requested to keep
    if [[ "${KEEP_PVS}" == false ]]; then
        delete_persistent_volumes
    else
        log_warning "Keeping persistent volumes as requested"
    fi

    # Delete namespace
    delete_namespace

    # Wait for cleanup
    wait_for_cleanup

    # Check for stuck resources
    cleanup_finalizers

    log_success "Monitoring stack teardown completed!"
}

# Run main function
main "$@"
