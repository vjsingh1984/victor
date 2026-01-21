#!/bin/bash
# Victor AI Monitoring Stack Verification Script
# Health check for monitoring components

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NAMESPACE="victor-monitoring"
TIMEOUT=300

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

check_pods() {
    log_info "Checking pod status..."

    local pods=$(kubectl get pods -n "${NAMESPACE}" -o json)
    local total=$(echo "${pods}" | jq '.items | length')
    local ready=$(echo "${pods}" | jq '[.items[] | select(.status.phase == "Running" and ([.status.conditions[] | select(.type == "Ready" and .status == "True")] | length > 0))] | length')
    local not_ready=$((total - ready))

    echo "  Total pods: ${total}"
    echo "  Ready pods: ${ready}"
    echo "  Not ready: ${not_ready}"

    if [[ ${not_ready} -gt 0 ]]; then
        log_warning "Some pods are not ready:"
        echo "${pods}" | jq -r '.items[] | select(.status.phase != "Running" or ([.status.conditions[] | select(.type == "Ready" and .status != "True")] | length > 0)) | "    - \(.metadata.name): \(.status.phase)"'
        return 1
    else
        log_success "All pods are running and ready"
        return 0
    fi
}

check_services() {
    log_info "Checking services..."

    local services=("prometheus" "grafana" "alertmanager" "otel-collector")
    local all_ok=true

    for svc in "${services[@]}"; do
        if kubectl get svc -n "${NAMESPACE}" "${svc}" &> /dev/null; then
            local type=$(kubectl get svc -n "${NAMESPACE}" "${svc}" -o jsonpath='{.spec.type}')
            local cluster_ip=$(kubectl get svc -n "${NAMESPACE}" "${svc}" -o jsonpath='{.spec.clusterIP}')
            echo "  ✓ ${svc}: ${type} (${cluster_ip})"
        else
            log_error "  ✗ ${svc}: Not found"
            all_ok=false
        fi
    done

    if [[ "${all_ok}" == true ]]; then
        log_success "All services are available"
        return 0
    else
        log_error "Some services are missing"
        return 1
    fi
}

check_persistent_volumes() {
    log_info "Checking persistent volume claims..."

    local pvcs=("prometheus-pvc" "grafana-pvc" "alertmanager-pvc")
    local all_ok=true

    for pvc in "${pvcs[@]}"; do
        if kubectl get pvc -n "${NAMESPACE}" "${pvc}" &> /dev/null; then
            local status=$(kubectl get pvc -n "${NAMESPACE}" "${pvc}" -o jsonpath='{.status.phase}')
            local capacity=$(kubectl get pvc -n "${NAMESPACE}" "${pvc}" -o jsonpath='{.spec.resources.requests.storage}')
            echo "  ✓ ${pvc}: ${status} (${capacity})"
        else
            log_warning "  ? ${pvc}: Not found or not bound"
        fi
    done

    log_success "PVC check completed"
    return 0
}

check_prometheus() {
    log_info "Checking Prometheus health..."

    local prometheus_pod=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/component=prometheus -o jsonpath='{.items[0].metadata.name}')

    if [[ -z "${prometheus_pod}" ]]; then
        log_error "Prometheus pod not found"
        return 1
    fi

    # Check if Prometheus is healthy
    local health=$(kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" -- wget -q -O- http://localhost:9090/-/healthy 2>/dev/null || echo "unhealthy")

    if [[ "${health}" == "Prometheus is Healthy." ]]; then
        log_success "Prometheus is healthy"

        # Get target count
        local targets=$(kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" -- wget -q -O- http://localhost:9090/api/v1/targets 2>/dev/null | jq '.data.activeTargets | length')
        echo "  Active targets: ${targets}"

        return 0
    else
        log_error "Prometheus health check failed"
        return 1
    fi
}

check_grafana() {
    log_info "Checking Grafana health..."

    local grafana_pod=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/component=grafana -o jsonpath='{.items[0].metadata.name}')

    if [[ -z "${grafana_pod}" ]]; then
        log_error "Grafana pod not found"
        return 1
    fi

    # Check if Grafana is healthy
    local health=$(kubectl exec -n "${NAMESPACE}" "${grafana_pod}" -- wget -q -O- http://localhost:3000/api/health 2>/dev/null)

    local database=$(echo "${health}" | jq -r '.database' || echo "unknown")
    echo "  Database: ${database}"

    if [[ "${database}" == "ok" ]]; then
        log_success "Grafana is healthy"
        return 0
    else
        log_error "Grafana health check failed"
        return 1
    fi
}

check_alertmanager() {
    log_info "Checking AlertManager health..."

    local alertmanager_pod=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/component=alertmanager -o jsonpath='{.items[0].metadata.name}')

    if [[ -z "${alertmanager_pod}" ]]; then
        log_error "AlertManager pod not found"
        return 1
    fi

    # Check if AlertManager is healthy
    local health=$(kubectl exec -n "${NAMESPACE}" "${alertmanager_pod}" -- wget -q -O- http://localhost:9093/-/healthy 2>/dev/null || echo "unhealthy")

    if [[ "${health}" == "OK" ]]; then
        log_success "AlertManager is healthy"
        return 0
    else
        log_warning "AlertManager health check returned: ${health}"
        return 1
    fi
}

check_otel_collector() {
    log_info "Checking OpenTelemetry Collector health..."

    local otel_pod=$(kubectl get pods -n "${NAMESPACE}" -l app.kubernetes.io/component=otel-collector -o jsonpath='{.items[0].metadata.name}')

    if [[ -z "${otel_pod}" ]]; then
        log_error "OpenTelemetry Collector pod not found"
        return 1
    fi

    # Check if OTel collector is healthy
    local health=$(kubectl exec -n "${NAMESPACE}" "${otel_pod}" -- wget -q -O- http://localhost:13133/ 2>/dev/null | jq -r '.status' || echo "unknown")

    if [[ "${health}" == "Server available" || "${health}" == "ok" ]]; then
        log_success "OpenTelemetry Collector is healthy"
        return 0
    else
        log_warning "OpenTelemetry Collector health check returned: ${health}"
        return 1
    fi
}

check_resource_usage() {
    log_info "Checking resource usage..."

    kubectl top pods -n "${NAMESPACE}" 2>/dev/null || log_warning "metrics-server not available"

    echo ""
    kubectl top nodes 2>/dev/null || log_warning "Cannot get node metrics"
}

generate_summary() {
    log_info "Generating verification summary..."

    local total=0
    local passed=0
    local failed=0

    # Run all checks
    check_pods && ((passed++)) || ((failed++))
    ((total++))

    check_services && ((passed++)) || ((failed++))
    ((total++))

    check_persistent_volumes && ((passed++)) || ((failed++))
    ((total++))

    check_prometheus && ((passed++)) || ((failed++))
    ((total++))

    check_grafana && ((passed++)) || ((failed++))
    ((total++))

    check_alertmanager && ((passed++)) || ((failed++))
    ((total++))

    check_otel_collector && ((passed++)) || ((failed++))
    ((total++))

    echo ""
    log_info "Verification Summary:"
    echo "  Total checks: ${total}"
    echo "  Passed: ${passed}"
    echo "  Failed: ${failed}"

    if [[ ${failed} -eq 0 ]]; then
        log_success "All verification checks passed!"
        return 0
    else
        log_warning "Some checks failed. Please review the output above."
        return 1
    fi
}

main() {
    log_info "Starting monitoring stack verification..."
    echo ""

    # Parse arguments
    VERBOSE=false
    SKIP_RESOURCE_USAGE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose)
                VERBOSE=true
                shift
                ;;
            --skip-resources)
                SKIP_RESOURCE_USAGE=true
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
                echo "  --verbose          Show detailed output"
                echo "  --skip-resources   Skip resource usage check"
                echo "  --namespace NAMESPACE  Monitoring namespace (default: victor-monitoring)"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Run verification
    generate_summary

    if [[ "${SKIP_RESOURCE_USAGE}" == false && "${VERBOSE}" == true ]]; then
        echo ""
        check_resource_usage
    fi

    exit $?
}

# Run main function
main "$@"
