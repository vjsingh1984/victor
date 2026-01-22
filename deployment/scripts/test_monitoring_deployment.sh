#!/bin/bash
# Victor AI - Monitoring Stack Deployment Test Suite
# Comprehensive test suite for monitoring deployment validation
#
# Usage:
#   ./test_monitoring_deployment.sh [options]
#
# Options:
#   --namespace NAME       Namespace (default: victor-monitoring)
#   --skip-ping            Skip connectivity tests
#   --generate-report      Generate test report
#   --help                 Show this help message

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${NAMESPACE:-victor-monitoring}"
SKIP_PING=false
GENERATE_REPORT=false

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Test results
declare -a TESTS_PASSED
declare -a TESTS_FAILED
declare -a TESTS_SKIPPED

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED+=("$1")
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED+=("$1")
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    TESTS_SKIPPED+=("$1")
}

log_test() {
    echo -e "\n${CYAN}[TEST]${NC} $1"
    echo -e "${CYAN}--------------------------------------------------------${NC}"
}

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

test_kubernetes_api() {
    log_test "Kubernetes API Connectivity"

    if kubectl cluster-info &> /dev/null; then
        log_success "Can connect to Kubernetes API"
        local version=$(kubectl version --short 2>/dev/null | grep Server | awk '{print $3}')
        log_info "Server version: ${version}"
        return 0
    else
        log_error "Cannot connect to Kubernetes API"
        return 1
    fi
}

test_namespace() {
    log_test "Namespace Existence"

    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_success "Namespace ${NAMESPACE} exists"
        return 0
    else
        log_error "Namespace ${NAMESPACE} does not exist"
        return 1
    fi
}

test_prometheus_pod() {
    log_test "Prometheus Pod"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -n "${pod}" ]]; then
        log_success "Prometheus pod found: ${pod}"

        local phase=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}')
        if [[ "${phase}" == "Running" ]]; then
            log_success "Prometheus pod is running"
        else
            log_error "Prometheus pod phase: ${phase}"
            return 1
        fi

        local ready=$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
            -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        if [[ "${ready}" == "True" ]]; then
            log_success "Prometheus pod is ready"
            return 0
        else
            log_error "Prometheus pod is not ready"
            return 1
        fi
    else
        log_error "Prometheus pod not found"
        return 1
    fi
}

test_grafana_pod() {
    log_test "Grafana Pod"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=grafana \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -n "${pod}" ]]; then
        log_success "Grafana pod found: ${pod}"

        local phase=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}')
        if [[ "${phase}" == "Running" ]]; then
            log_success "Grafana pod is running"
        else
            log_error "Grafana pod phase: ${phase}"
            return 1
        fi

        local ready=$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
            -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        if [[ "${ready}" == "True" ]]; then
            log_success "Grafana pod is ready"
            return 0
        else
            log_error "Grafana pod is not ready"
            return 1
        fi
    else
        log_error "Grafana pod not found"
        return 1
    fi
}

test_alertmanager_pod() {
    log_test "AlertManager Pod"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=alertmanager \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -n "${pod}" ]]; then
        log_success "AlertManager pod found: ${pod}"

        local phase=$(kubectl get pod "${pod}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}')
        if [[ "${phase}" == "Running" ]]; then
            log_success "AlertManager pod is running"
        else
            log_error "AlertManager pod phase: ${phase}"
            return 1
        fi

        local ready=$(kubectl get pod "${pod}" -n "${NAMESPACE}" \
            -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
        if [[ "${ready}" == "True" ]]; then
            log_success "AlertManager pod is ready"
            return 0
        else
            log_error "AlertManager pod is not ready"
            return 1
        fi
    else
        log_error "AlertManager pod not found"
        return 1
    fi
}

test_persistent_volumes() {
    log_test "Persistent Volume Claims"

    local pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o json 2>/dev/null)
    local pvc_count=$(echo "${pvcs}" | jq -r '.items | length' 2>/dev/null || echo "0")

    if [[ ${pvc_count} -ge 3 ]]; then
        log_success "Found ${pvc_count} PVCs"

        local bound_count=0
        echo "${pvcs}" | jq -r '.items[] | "\(.metadata.name)\t\(.status.phase)"' 2>/dev/null | while read -r name phase; do
            if [[ "${phase}" == "Bound" ]]; then
                log_success "PVC ${name} is bound"
                bound_count=$((bound_count + 1))
            else
                log_error "PVC ${name} is not bound (Phase: ${phase})"
            fi
        done

        return 0
    else
        log_error "Expected at least 3 PVCs, found ${pvc_count}"
        return 1
    fi
}

test_prometheus_service() {
    log_test "Prometheus Service"

    if kubectl get svc prometheus -n "${NAMESPACE}" &> /dev/null; then
        log_success "Prometheus service exists"

        local endpoint=$(kubectl get svc prometheus -n "${NAMESPACE}" \
            -o jsonpath='{.spec.type}');

        log_success "Service type: ${endpoint}"

        local port=$(kubectl get svc prometheus -n "${NAMESPACE}" \
            -o jsonpath='{.spec.ports[0].port}');

        log_success "Service port: ${port}"

        return 0
    else
        log_error "Prometheus service not found"
        return 1
    fi
}

test_grafana_service() {
    log_test "Grafana Service"

    if kubectl get svc grafana -n "${NAMESPACE}" &> /dev/null; then
        log_success "Grafana service exists"

        local endpoint=$(kubectl get svc grafana -n "${NAMESPACE}" \
            -o jsonpath='{.spec.type}');

        log_success "Service type: ${endpoint}"

        local port=$(kubectl get svc grafana -n "${NAMESPACE}" \
            -o jsonpath='{.spec.ports[0].port}');

        log_success "Service port: ${port}"

        return 0
    else
        log_error "Grafana service not found"
        return 1
    fi
}

test_alertmanager_service() {
    log_test "AlertManager Service"

    if kubectl get svc alertmanager -n "${NAMESPACE}" &> /dev/null; then
        log_success "AlertManager service exists"

        local endpoint=$(kubectl get svc alertmanager -n "${NAMESPACE}" \
            -o jsonpath='{.spec.type}');

        log_success "Service type: ${endpoint}"

        local port=$(kubectl get svc alertmanager -n "${NAMESPACE}" \
            -o jsonpath='{.spec.ports[0].port}');

        log_success "Service port: ${port}"

        return 0
    else
        log_error "AlertManager service not found"
        return 1
    fi
}

test_prometheus_health() {
    log_test "Prometheus Health Endpoint"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Prometheus pod not found, skipping health check"
        return 1
    fi

    local health=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:9090/-/healthy 2>/dev/null || echo "unhealthy")

    if [[ "${health}" == "Prometheus is Healthy." ]]; then
        log_success "Prometheus health check passed"
        return 0
    else
        log_error "Prometheus health check failed: ${health}"
        return 1
    fi
}

test_grafana_health() {
    log_test "Grafana Health Endpoint"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=grafana \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Grafana pod not found, skipping health check"
        return 1
    fi

    local health=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:3000/api/health 2>/dev/null)

    local database_status=$(echo "${health}" | jq -r '.database' 2>/dev/null || echo "unknown")

    if [[ "${database_status}" == "ok" ]]; then
        log_success "Grafana health check passed"
        return 0
    else
        log_error "Grafana health check failed: database=${database_status}"
        return 1
    fi
}

test_alertmanager_health() {
    log_test "AlertManager Health Endpoint"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=alertmanager \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "AlertManager pod not found, skipping health check"
        return 1
    fi

    local health=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:9093/-/healthy 2>/dev/null || echo "unhealthy")

    if [[ "${health}" == "OK" ]]; then
        log_success "AlertManager health check passed"
        return 0
    else
        log_error "AlertManager health check failed: ${health}"
        return 1
    fi
}

test_prometheus_metrics() {
    log_test "Prometheus Metrics Collection"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Prometheus pod not found, skipping metrics check"
        return 1
    fi

    # Query for a simple metric
    local result=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- 'http://localhost:9090/api/v1/query?query=up' 2>/dev/null \
        | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "")

    if [[ -n "${result}" ]]; then
        log_success "Prometheus is collecting metrics"
        log_info "Sample metric 'up': ${result}"
        return 0
    else
        log_error "Prometheus metrics query failed"
        return 1
    fi
}

test_prometheus_targets() {
    log_test "Prometheus Targets"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Prometheus pod not found, skipping targets check"
        return 1
    fi

    local targets=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:9090/api/v1/targets 2>/dev/null)

    if [[ -n "${targets}" ]]; then
        local total=$(echo "${targets}" | jq -r '.data.activeTargets | length')
        local up=$(echo "${targets}" | jq -r '.data.activeTargets[] | select(.health == "up") | .job' | wc -l)

        log_success "Prometheus targets: ${up}/${total} up"

        if [[ ${up} -gt 0 ]]; then
            return 0
        else
            log_error "No targets are up"
            return 1
        fi
    else
        log_error "Could not fetch Prometheus targets"
        return 1
    fi
}

test_prometheus_rules() {
    log_test "Prometheus Alerting Rules"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=prometheus \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Prometheus pod not found, skipping rules check"
        return 1
    fi

    local rules=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:9090/api/v1/rules 2>/dev/null)

    if [[ -n "${rules}" ]]; then
        local rule_count=$(echo "${rules}" | jq -r '[.data.groups[].rules[] | select(.type=="alerting")] | length')

        log_success "Alerting rules loaded: ${rule_count}"

        if [[ ${rule_count} -ge 50 ]]; then
            log_success "Expected 50+ rules, found ${rule_count}"
            return 0
        else
            log_warning "Expected 50+ rules, found ${rule_count}"
            return 0
        fi
    else
        log_error "Could not fetch alerting rules"
        return 1
    fi
}

test_grafana_dashboards() {
    log_test "Grafana Dashboards"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=grafana \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Grafana pod not found, skipping dashboards check"
        return 1
    fi

    local dashboards=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- 'http://localhost:3000/api/search?query=%' 2>/dev/null)

    if [[ -n "${dashboards}" ]]; then
        local count=$(echo "${dashboards}" | jq -r '. | length')

        log_success "Grafana dashboards installed: ${count}"

        if [[ ${count} -ge 7 ]]; then
            log_success "Expected 7 dashboards, found ${count}"
        else
            log_warning "Expected 7 dashboards, found ${count}"
        fi

        # List dashboards
        if [[ ${count} -gt 0 ]]; then
            log_info "Installed dashboards:"
            echo "${dashboards}" | jq -r '.[] | "  - \(.title)"' 2>/dev/null
        fi

        return 0
    else
        log_error "Could not fetch Grafana dashboards"
        return 1
    fi
}

test_grafana_datasources() {
    log_test "Grafana Datasources"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=grafana \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "Grafana pod not found, skipping datasources check"
        return 1
    fi

    local datasources=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:3000/api/datasources 2>/dev/null)

    if [[ -n "${datasources}" ]]; then
        local count=$(echo "${datasources}" | jq -r '. | length')

        log_success "Grafana datasources configured: ${count}"

        # Check for Prometheus datasource
        local has_prometheus=$(echo "${datasources}" | jq -r '[.[] | select(.type == "prometheus")] | length')

        if [[ ${has_prometheus} -gt 0 ]]; then
            log_success "Prometheus datasource configured"
            return 0
        else
            log_error "Prometheus datasource not found"
            return 1
        fi
    else
        log_error "Could not fetch Grafana datasources"
        return 1
    fi
}

test_grafana_credentials() {
    log_test "Grafana Credentials"

    if kubectl get secret grafana-admin-credentials -n "${NAMESPACE}" &> /dev/null; then
        log_success "Grafana credentials secret exists"

        local user=$(kubectl get secret grafana-admin-credentials -n "${NAMESPACE}" \
            -o jsonpath='{.data.admin-user}' | base64 -d 2>/dev/null || echo "N/A")

        log_info "Admin username: ${user}"
        log_info "Admin password: (hidden for security)"

        return 0
    else
        log_error "Grafana credentials secret not found"
        log_info "Setup credentials with: ./setup_monitoring_credentials.sh --generate-password"
        return 1
    fi
}

test_alertmanager_config() {
    log_test "AlertManager Configuration"

    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/component=alertmanager \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [[ -z "${pod}" ]]; then
        log_skip "AlertManager pod not found, skipping config check"
        return 1
    fi

    local config=$(kubectl exec -n "${NAMESPACE}" "${pod}" \
        -- wget -q -O- http://localhost:9093/api/v1/status/config 2>/dev/null)

    if [[ -n "${config}" ]]; then
        log_success "AlertManager configuration loaded"

        # Check for notification configuration
        local has_slack=$(echo "${config}" | jq -e '.global.slack_api_url' &> /dev/null && echo "yes" || echo "no")
        local has_smtp=$(echo "${config}" | jq -e '.global.smtp_smarthost' &> /dev/null && echo "yes" || echo "no")

        if [[ "${has_slack}" == "yes" ]]; then
            log_success "Slack notifications configured"
        else
            log_info "Slack notifications not configured"
        fi

        if [[ "${has_smtp}" == "yes" ]]; then
            log_success "Email notifications configured"
        else
            log_info "Email notifications not configured"
        fi

        return 0
    else
        log_error "Could not fetch AlertManager configuration"
        return 1
    fi
}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "Victor AI - Monitoring Stack Test Suite"
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
            --skip-ping)
                SKIP_PING=true
                shift
                ;;
            --generate-report)
                GENERATE_REPORT=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --namespace NAME       Namespace (default: victor-monitoring)"
                echo "  --skip-ping            Skip connectivity tests"
                echo "  --generate-report      Generate test report"
                echo "  --help                 Show this help message"
                echo ""
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Run tests
    test_kubernetes_api
    test_namespace
    test_prometheus_pod
    test_grafana_pod
    test_alertmanager_pod
    test_persistent_volumes
    test_prometheus_service
    test_grafana_service
    test_alertmanager_service
    test_prometheus_health
    test_grafana_health
    test_alertmanager_health
    test_prometheus_metrics
    test_prometheus_targets
    test_prometheus_rules
    test_grafana_dashboards
    test_grafana_datasources
    test_grafana_credentials
    test_alertmanager_config

    # Summary
    echo ""
    echo "============================================================"
    echo "TEST SUMMARY"
    echo "============================================================"
    echo ""
    echo "Passed:  ${#TESTS_PASSED[@]}"
    echo "Failed:  ${#TESTS_FAILED[@]}"
    echo "Skipped: ${#TESTS_SKIPPED[@]}"
    echo ""

    # Generate report
    if [[ "${GENERATE_REPORT}" == true ]]; then
        local report_file="monitoring-test-report-$(date +%Y%m%d-%H%M%S).txt"

        {
            echo "Victor AI - Monitoring Stack Test Report"
            echo "========================================"
            echo ""
            echo "Date: $(date)"
            echo "Namespace: ${NAMESPACE}"
            echo ""
            echo "Results:"
            echo "  Passed: ${#TESTS_PASSED[@]}"
            echo "  Failed: ${#TESTS_FAILED[@]}"
            echo "  Skipped: ${#TESTS_SKIPPED[@]}"
            echo ""

            if [[ ${#TESTS_PASSED[@]} -gt 0 ]]; then
                echo "Passed Tests:"
                for test in "${TESTS_PASSED[@]}"; do
                    echo "  ✓ ${test}"
                done
                echo ""
            fi

            if [[ ${#TESTS_FAILED[@]} -gt 0 ]]; then
                echo "Failed Tests:"
                for test in "${TESTS_FAILED[@]}"; do
                    echo "  ✗ ${test}"
                done
                echo ""
            fi

            if [[ ${#TESTS_SKIPPED[@]} -gt 0 ]]; then
                echo "Skipped Tests:"
                for test in "${TESTS_SKIPPED[@]}"; do
                    echo "  ⊘ ${test}"
                done
                echo ""
            fi
        } > "${report_file}"

        log_info "Test report saved to: ${report_file}"
    fi

    # Exit with appropriate code
    if [[ ${#TESTS_FAILED[@]} -gt 0 ]]; then
        echo "Status: TESTS FAILED"
        exit 1
    else
        echo "Status: ALL TESTS PASSED"
        exit 0
    fi
}

# Run main function
main "$@"
