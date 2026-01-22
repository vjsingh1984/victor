#!/bin/bash
# Victor AI - Monitoring Stack Verification Script
# Performs comprehensive health checks and validation of the monitoring stack
#
# Usage:
#   ./verify_monitoring_complete.sh [options]
#
# Options:
#   --namespace NAME       Namespace (default: victor-monitoring)
#   --detailed             Show detailed output
#   --quiet                Only show errors
#   --fix                  Attempt to fix common issues
#   --export-report FILE   Export verification report to file
#   --help                 Show this help message

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${NAMESPACE:-victor-monitoring}"
DETAILED=false
QUIET=false
AUTO_FIX=false
EXPORT_REPORT=""

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Verification results
declare -a CHECKS_PASSED
declare -a CHECKS_FAILED
declare -a CHECKS_WARNINGS
declare -a RECOMMENDATIONS

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    if [[ "${QUIET}" == false ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    if [[ "${QUIET}" == false ]]; then
        echo -e "${GREEN}[PASS]${NC} $1"
    fi
    CHECKS_PASSED+=("$1")
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    CHECKS_WARNINGS+=("$1")
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    CHECKS_FAILED+=("$1")
}

log_detail() {
    if [[ "${DETAILED}" == true && "${QUIET}" == false ]]; then
        echo -e "${CYAN}      ${NC}$1"
    fi
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

show_help() {
    cat << EOF
Victor AI - Monitoring Stack Verification

Usage: $0 [OPTIONS]

Options:
  --namespace NAME       Namespace (default: victor-monitoring)
  --detailed             Show detailed output
  --quiet                Only show errors
  --fix                  Attempt to fix common issues
  --export-report FILE   Export verification report to file
  --help                 Show this help message

Examples:
  # Basic verification
  $0

  # Detailed verification
  $0 --detailed

  # Fix common issues automatically
  $0 --fix

  # Export report
  $0 --export-report report.txt

Checks performed:
  - Kubernetes cluster connectivity
  - Namespace existence
  - Pod health and status
  - Service availability
  - Persistent volume claims
  - ConfigMaps and secrets
  - Prometheus configuration and targets
  - AlertManager configuration
  - Grafana access and dashboards
  - Alerting rules
  - Metrics collection
  - Resource usage

EOF
}

export_report() {
    local report_file=$1

    {
        echo "Victor AI - Monitoring Stack Verification Report"
        echo "================================================"
        echo ""
        echo "Date: $(date)"
        echo "Namespace: ${NAMESPACE}"
        echo ""
        echo "Summary:"
        echo "  Passed: ${#CHECKS_PASSED[@]}"
        echo "  Warnings: ${#CHECKS_WARNINGS[@]}"
        echo "  Failed: ${#CHECKS_FAILED[@]}"
        echo ""
        echo "------------------------------------------------------------"
        echo ""
        if [[ ${#CHECKS_PASSED[@]} -gt 0 ]]; then
            echo "Passed Checks:"
            for check in "${CHECKS_PASSED[@]}"; do
                echo "  ✓ ${check}"
            done
            echo ""
        fi

        if [[ ${#CHECKS_WARNINGS[@]} -gt 0 ]]; then
            echo "Warnings:"
            for warning in "${CHECKS_WARNINGS[@]}"; do
                echo "  ⚠ ${warning}"
            done
            echo ""
        fi

        if [[ ${#CHECKS_FAILED[@]} -gt 0 ]]; then
            echo "Failed Checks:"
            for fail in "${CHECKS_FAILED[@]}"; do
                echo "  ✗ ${fail}"
            done
            echo ""
        fi

        if [[ ${#RECOMMENDATIONS[@]} -gt 0 ]]; then
            echo "Recommendations:"
            for rec in "${RECOMMENDATIONS[@]}"; do
                echo "  → ${rec}"
            done
            echo ""
        fi
    } > "${report_file}"

    echo ""
    log_info "Report exported to: ${report_file}"
}

# ============================================================================
# CHECK FUNCTIONS
# ============================================================================

check_cluster_connection() {
    log_info "Checking cluster connection..."

    if kubectl cluster-info &> /dev/null; then
        log_success "Cluster connection verified"
        log_detail "Cluster: $(kubectl config view --minify -o jsonpath='{.clusters[0].name}')"
        return 0
    else
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
}

check_namespace() {
    log_info "Checking namespace..."

    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_success "Namespace ${NAMESPACE} exists"
        return 0
    else
        log_error "Namespace ${NAMESPACE} does not exist"
        RECOMMENDATIONS+=("Create namespace: kubectl create namespace ${NAMESPACE}")
        return 1
    fi
}

check_pods() {
    log_info "Checking pod status..."

    local pods=$(kubectl get pods -n "${NAMESPACE}" -o json 2>/dev/null)
    local pod_count=$(echo "${pods}" | jq -r '.items | length' 2>/dev/null || echo "0")

    if [[ ${pod_count} -eq 0 ]]; then
        log_error "No pods found in namespace ${NAMESPACE}"
        RECOMMENDATIONS+=("Deploy monitoring stack: ./deploy_monitoring_complete.sh")
        return 1
    fi

    log_detail "Found ${pod_count} pod(s)"

    local all_ready=true
    echo "${pods}" | jq -r '.items[] | "\(.metadata.name)\t\(.status.phase)\t\(.status.conditions[] | select(.type == "Ready") | .status)"' 2>/dev/null | while read -r name phase ready; do
        if [[ "${phase}" != "Running" ]] || [[ "${ready}" != "True" ]]; then
            log_error "Pod ${name} is not healthy (Phase: ${phase}, Ready: ${ready})"
            all_ready=false
        else
            log_detail "Pod ${name} is healthy"
        fi
    done

    if [[ "${all_ready}" == true ]]; then
        log_success "All pods are healthy"
        return 0
    else
        log_error "Some pods are not healthy"
        if [[ "${AUTO_FIX}" == true ]]; then
            log_info "Attempting to restart unhealthy pods..."
            kubectl delete pods -n "${NAMESPACE}" --field-selector=status.phase!=Running &> /dev/null || true
        fi
        return 1
    fi
}

check_services() {
    log_info "Checking services..."

    local services=$(kubectl get svc -n "${NAMESPACE}" -o json 2>/dev/null)
    local service_count=$(echo "${services}" | jq -r '.items | length' 2>/dev/null || echo "0")

    if [[ ${service_count} -eq 0 ]]; then
        log_warning "No services found"
        return 1
    fi

    log_detail "Found ${service_count} service(s)"

    local required_services=("prometheus" "grafana" "alertmanager")
    local all_found=true

    for svc in "${required_services[@]}"; do
        if echo "${services}" | jq -e ".items[].metadata.name" | grep -q "^${svc}$"; then
            log_detail "Service ${svc} found"
        else
            log_error "Required service ${svc} not found"
            all_found=false
        fi
    done

    if [[ "${all_found}" == true ]]; then
        log_success "All required services are present"
        return 0
    else
        log_error "Some required services are missing"
        return 1
    fi
}

check_pvcs() {
    log_info "Checking persistent volume claims..."

    local pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o json 2>/dev/null)
    local pvc_count=$(echo "${pvcs}" | jq -r '.items | length' 2>/dev/null || echo "0")

    if [[ ${pvc_count} -eq 0 ]]; then
        log_warning "No persistent volume claims found"
        RECOMMENDATIONS+=("Configure persistent storage for data persistence")
        return 1
    fi

    log_detail "Found ${pvc_count} PVC(s)"

    local all_bound=true
    echo "${pvcs}" | jq -r '.items[] | "\(.metadata.name)\t\(.status.phase)"' 2>/dev/null | while read -r name phase; do
        if [[ "${phase}" != "Bound" ]]; then
            log_error "PVC ${name} is not bound (Phase: ${phase})"
            all_bound=false
        else
            log_detail "PVC ${name} is bound"
        fi
    done

    if [[ "${all_bound}" == true ]]; then
        log_success "All PVCs are bound"
        return 0
    else
        log_error "Some PVCs are not bound"
        return 1
    fi
}

check_configmaps() {
    log_info "Checking ConfigMaps..."

    local configmaps=("prometheus-config" "alertmanager-config" "grafana-config")
    local all_found=true

    for cm in "${configmaps[@]}"; do
        if kubectl get configmap "${cm}" -n "${NAMESPACE}" &> /dev/null; then
            log_detail "ConfigMap ${cm} exists"
        else
            log_error "ConfigMap ${cm} not found"
            all_found=false
        fi
    done

    if [[ "${all_found}" == true ]]; then
        log_success "All required ConfigMaps exist"
        return 0
    else
        log_error "Some ConfigMaps are missing"
        return 1
    fi
}

check_secrets() {
    log_info "Checking secrets..."

    local secrets=("grafana-admin-credentials")
    local all_found=true

    for secret in "${secrets[@]}"; do
        if kubectl get secret "${secret}" -n "${NAMESPACE}" &> /dev/null; then
            log_detail "Secret ${secret} exists"
        else
            log_warning "Secret ${secret} not found"
            RECOMMENDATIONS+=("Setup credentials: ./setup_monitoring_credentials.sh --generate-password")
            all_found=false
        fi
    done

    if [[ "${all_found}" == true ]]; then
        log_success "All required secrets exist"
        return 0
    else
        log_warning "Some secrets are missing"
        return 1
    fi
}

check_prometheus() {
    log_info "Checking Prometheus..."

    local prometheus_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${prometheus_pod}" ]]; then
        log_error "Prometheus pod not found"
        return 1
    fi

    log_detail "Prometheus pod: ${prometheus_pod}"

    # Check Prometheus health endpoint
    local health=$(kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" \
        -- wget -q -O- http://localhost:9090/-/healthy 2>/dev/null || echo "unhealthy")

    if [[ "${health}" == "Prometheus is Healthy." ]]; then
        log_success "Prometheus is healthy"

        # Check Prometheus targets
        if [[ "${DETAILED}" == true ]]; then
            log_info "Checking Prometheus targets..."

            local targets_json=$(kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" \
                -- wget -q -O- http://localhost:9090/api/v1/targets 2>/dev/null)

            if [[ -n "${targets_json}" ]]; then
                local total_targets=$(echo "${targets_json}" | jq -r '.data.activeTargets | length')
                local up_targets=$(echo "${targets_json}" | jq -r '.data.activeTargets[] | select(.health == "up") | .job' | wc -l)

                log_detail "Targets: ${up_targets}/${total_targets} up"

                echo "${targets_json}" | jq -r '.data.activeTargets[] | "\(.job): \(.health)"' 2>/dev/null | while read -r line; do
                    log_detail "  ${line}"
                done
            fi
        fi

        return 0
    else
        log_error "Prometheus health check failed"
        return 1
    fi
}

check_alertmanager() {
    log_info "Checking AlertManager..."

    local alertmanager_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=alertmanager \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${alertmanager_pod}" ]]; then
        log_error "AlertManager pod not found"
        return 1
    fi

    log_detail "AlertManager pod: ${alertmanager_pod}"

    # Check AlertManager health endpoint
    local health=$(kubectl exec -n "${NAMESPACE}" "${alertmanager_pod}" \
        -- wget -q -O- http://localhost:9093/-/healthy 2>/dev/null || echo "unhealthy")

    if [[ "${health}" == "OK" ]]; then
        log_success "AlertManager is healthy"
        return 0
    else
        log_error "AlertManager health check failed"
        return 1
    fi
}

check_grafana() {
    log_info "Checking Grafana..."

    local grafana_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=grafana \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${grafana_pod}" ]]; then
        log_error "Grafana pod not found"
        return 1
    fi

    log_detail "Grafana pod: ${grafana_pod}"

    # Check Grafana health endpoint
    local health=$(kubectl exec -n "${NAMESPACE}" "${grafana_pod}" \
        -- wget -q -O- http://localhost:3000/api/health 2>/dev/null)

    local database_status=$(echo "${health}" | jq -r '.database' 2>/dev/null || echo "unknown")

    if [[ "${database_status}" == "ok" ]]; then
        log_success "Grafana is healthy"

        # Check dashboards
        if [[ "${DETAILED}" == true ]]; then
            log_info "Checking Grafana dashboards..."

            local dashboards=$(kubectl exec -n "${NAMESPACE}" "${grafana_pod}" \
                -- wget -q -O- http://localhost:3000/api/search?query=% 2>/dev/null \
                | jq -r '.[] | .title' 2>/dev/null || echo "")

            local dashboard_count=$(echo "${dashboards}" | wc -l)

            log_detail "Dashboards installed: ${dashboard_count}"

            if [[ ${dashboard_count} -ge 7 ]]; then
                log_detail "All 7 dashboards are installed"
            else
                log_warning "Expected 7 dashboards, found ${dashboard_count}"
                RECOMMENDATIONS+=("Import missing dashboards: ./deploy_monitoring_complete.sh --skip-alerts")
            fi
        fi

        return 0
    else
        log_error "Grafana health check failed"
        return 1
    fi
}

check_alerting_rules() {
    log_info "Checking alerting rules..."

    local prometheus_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${prometheus_pod}" ]]; then
        log_error "Cannot check rules: Prometheus pod not found"
        return 1
    fi

    # Get alerting rules
    local rules=$(kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" \
        -- wget -q -O- http://localhost:9090/api/v1/rules 2>/dev/null)

    if [[ -n "${rules}" ]]; then
        local rule_count=$(echo "${rules}" | jq -r '.data.groups[].rules[] | select(.type=="alerting") | .alert' | wc -l)

        log_detail "Alerting rules loaded: ${rule_count}"

        if [[ ${rule_count} -ge 50 ]]; then
            log_success "Alerting rules configured (${rule_count} rules)"
            return 0
        else
            log_warning "Expected 50+ rules, found ${rule_count}"
            RECOMMENDATIONS+=("Verify alerting rules configuration")
            return 1
        fi
    else
        log_warning "Could not retrieve alerting rules"
        return 1
    fi
}

check_metrics() {
    log_info "Checking metrics collection..."

    local prometheus_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${prometheus_pod}" ]]; then
        log_error "Cannot check metrics: Prometheus pod not found"
        return 1
    fi

    # Query for some sample metrics
    local sample_metrics=(
        "up"
        "prometheus_tsdb_head_samples_appended_total"
        "prometheus_config_last_reload_successful"
    )

    local metrics_found=0

    for metric in "${sample_metrics[@]}"; do
        local result=$(kubectl exec -n "${NAMESPACE}" "${prometheus_pod}" \
            -- wget -q -O- "http://localhost:9090/api/v1/query?query=${metric}" 2>/dev/null \
            | jq -r '.data.result[0].value[1]' 2>/dev/null)

        if [[ -n "${result}" ]]; then
            log_detail "Metric ${metric} is being collected"
            metrics_found=$((metrics_found + 1))
        fi
    done

    if [[ ${metrics_found} -gt 0 ]]; then
        log_success "Metrics are being collected (${metrics_found}/${#sample_metrics[@]} sample metrics found)"
        return 0
    else
        log_error "No metrics found"
        return 1
    fi
}

check_resource_usage() {
    log_info "Checking resource usage..."

    local pods=$(kubectl get pods -n "${NAMESPACE}" -o json 2>/dev/null)

    if [[ -z "${pods}" ]]; then
        log_error "Cannot check resource usage: no pods found"
        return 1
    fi

    echo "${pods}" | jq -r '.items[].metadata.name' 2>/dev/null | while read -r pod; do
        local cpu_request=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.spec.containers[0].resources.requests.cpu}')
        local cpu_limit=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.spec.containers[0].resources.limits.cpu}')
        local memory_request=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.spec.containers[0].resources.requests.memory}')
        local memory_limit=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.spec.containers[0].resources.limits.memory}')

        if [[ "${DETAILED}" == true ]]; then
            log_detail "Pod ${pod}:"
            log_detail "  CPU: ${cpu_request} / ${cpu_limit}"
            log_detail "  Memory: ${memory_request} / ${memory_limit}"
        fi
    done

    log_success "Resource usage checked"
    return 0
}

check_notifications() {
    log_info "Checking notification configuration..."

    local alertmanager_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=alertmanager \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${alertmanager_pod}" ]]; then
        log_warning "Cannot check notifications: AlertManager pod not found"
        return 1
    fi

    # Get AlertManager config
    local config=$(kubectl exec -n "${NAMESPACE}" "${alertmanager_pod}" \
        -- wget -q -O- http://localhost:9093/api/v1/status/config 2>/dev/null \
        | jq -r '.data' 2>/dev/null)

    if [[ -n "${config}" ]]; then
        # Check if Slack is configured
        if echo "${config}" | jq -e '.global.slack_api_url' &> /dev/null; then
            if [[ $(echo "${config}" | jq -r '.global.slack_api_url') != "null" ]]; then
                log_detail "Slack notifications are configured"
            fi
        fi

        # Check if email is configured
        if echo "${config}" | jq -e '.global.smtp_smarthost' &> /dev/null; then
            if [[ $(echo "${config}" | jq -r '.global.smtp_smarthost') != "null" ]]; then
                log_detail "Email notifications are configured"
            fi
        fi

        log_success "Notification configuration checked"
        return 0
    else
        log_warning "Could not retrieve notification configuration"
        RECOMMENDATIONS+=("Configure notification channels: ./setup_monitoring_credentials.sh --email creds.json --slack webhook.txt")
        return 1
    fi
}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "Victor AI - Monitoring Stack Verification"
    echo "============================================================"
    echo ""
    log_info "Namespace: ${NAMESPACE}"
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --detailed)
                DETAILED=true
                shift
                ;;
            --quiet)
                QUIET=true
                shift
                ;;
            --fix)
                AUTO_FIX=true
                shift
                ;;
            --export-report)
                EXPORT_REPORT="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Run checks
    check_cluster_connection
    check_namespace
    check_pods
    check_services
    check_pvcs
    check_configmaps
    check_secrets
    check_prometheus
    check_alertmanager
    check_grafana
    check_alerting_rules
    check_metrics
    check_resource_usage
    check_notifications

    # Summary
    echo ""
    echo "============================================================"
    echo "VERIFICATION SUMMARY"
    echo "============================================================"
    echo ""
    echo "Passed:  ${#CHECKS_PASSED[@]}"
    echo "Warnings: ${#CHECKS_WARNINGS[@]}"
    echo "Failed:  ${#CHECKS_FAILED[@]}"
    echo ""

    # Export report if requested
    if [[ -n "${EXPORT_REPORT}" ]]; then
        export_report "${EXPORT_REPORT}"
    fi

    # Show recommendations
    if [[ ${#RECOMMENDATIONS[@]} -gt 0 ]]; then
        echo "Recommendations:"
        for rec in "${RECOMMENDATIONS[@]}"; do
            echo "  → ${rec}"
        done
        echo ""
    fi

    # Exit with appropriate code
    if [[ ${#CHECKS_FAILED[@]} -gt 0 ]]; then
        echo "Status: FAILED"
        exit 1
    elif [[ ${#CHECKS_WARNINGS[@]} -gt 0 ]]; then
        echo "Status: PASSED WITH WARNINGS"
        exit 0
    else
        echo "Status: ALL CHECKS PASSED"
        exit 0
    fi
}

# Run main function
main "$@"
