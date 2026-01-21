#!/bin/bash
################################################################################
# Post-Deployment Verification Script
#
# This script performs comprehensive verification after deployment to ensure
# the application is healthy and functioning correctly.
#
# Usage:
#   ./verify_deployment.sh [environment]
#
# Arguments:
#   environment - Target environment (production, staging, development)
#
# Features:
# - Automated health checks
# - Smoke tests execution
# - Performance verification
# - Monitoring verification
# - Log analysis
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ROOT="${PROJECT_ROOT}/deployment"
LOG_DIR="${PROJECT_ROOT}/logs"
REPORT_DIR="${PROJECT_ROOT}/reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Verification results
declare -a VERIFICATION_RESULTS
declare -a VERIFICATION_ERRORS
VERIFICATION_SCORE=0
MAX_SCORE=100

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

add_result() {
    VERIFICATION_RESULTS+=("$1|$2")
    if [[ "$2" == "PASS" ]]; then
        VERIFICATION_SCORE=$((VERIFICATION_SCORE + 10))
    fi
}

add_error() {
    VERIFICATION_ERRORS+=("$1")
}

################################################################################
# Configuration
################################################################################

ENVIRONMENT="${1:-production}"
VERIFICATION_LOG="${LOG_DIR}/post_verification_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"
VERIFICATION_REPORT="${REPORT_DIR}/post_verification_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).json"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${REPORT_DIR}"

# Initialize log
echo "Post-Deployment Verification - ${ENVIRONMENT}" | tee "${VERIFICATION_LOG}"
echo "Started at: $(date)" | tee -a "${VERIFICATION_LOG}"
echo "=======================================================" | tee -a "${VERIFICATION_LOG}"

################################################################################
# Verification Functions
################################################################################

verify_deployment_health() {
    log_section "Verifying Deployment Health"

    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Check deployment status
    local deployment_status
    deployment_status=$(kubectl get deployment victor -n victor -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "False")

    if [[ "${deployment_status}" == "True" ]]; then
        log_success "Deployment is available"
        add_result "Deployment Health" "PASS"
    else
        log_error "Deployment is not available"
        add_result "Deployment Health" "FAIL"
        add_error "Deployment unavailable"
        return 1
    fi

    # Check replica count
    local replicas_ready
    local replicas_desired
    replicas_ready=$(kubectl get deployment victor -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    replicas_desired=$(kubectl get deployment victor -n victor -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

    log_info "Replicas: ${replicas_ready}/${replicas_desired} ready"

    if [[ "${replicas_ready}" -ge "${replicas_desired}" ]]; then
        log_success "All replicas are ready"
        add_result "Replica Health" "PASS"
    else
        log_warning "Not all replicas ready (${replicas_ready}/${replicas_desired})"
        add_result "Replica Health" "WARN"
        add_error "Replicas not ready"
    fi

    # Check pod status
    local pods_running
    local pods_total
    pods_running=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -c "Running" || echo "0")
    pods_total=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | wc -w || echo "0")

    log_info "Pods: ${pods_running}/${pods_total} running"

    if [[ "${pods_running}" -eq "${pods_total}" ]] && [[ "${pods_total}" -gt 0 ]]; then
        log_success "All pods are running"
        add_result "Pod Health" "PASS"
    else
        log_warning "Some pods are not running"
        add_result "Pod Health" "WARN"
        add_error "Pods not running"
    fi

    return 0
}

verify_service_endpoints() {
    log_section "Verifying Service Endpoints"

    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    local service_host
    service_host=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

    local endpoint="${service_ip}"
    if [[ -z "${endpoint}" ]]; then
        endpoint="${service_host}"
    fi

    if [[ -z "${endpoint}" ]]; then
        log_error "No service endpoint found"
        add_result "Service Endpoint" "FAIL"
        add_error "No endpoint available"
        return 1
    fi

    log_info "Service endpoint: ${endpoint}"

    # Test HTTP endpoint
    local max_attempts=5
    local attempt=0

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        log_info "Testing endpoint (attempt $((attempt + 1))/${max_attempts})..."

        if curl -sf "http://${endpoint}/health" &>/dev/null; then
            log_success "Health endpoint is responding"
            add_result "Health Endpoint" "PASS"
            break
        fi

        attempt=$((attempt + 1))
        sleep 5
    done

    if [[ ${attempt} -eq ${max_attempts} ]]; then
        log_error "Health endpoint not responding"
        add_result "Health Endpoint" "FAIL"
        add_error "Health check failed"
    fi

    # Test API endpoint
    if curl -sf "http://${endpoint}/api/v1/health" &>/dev/null; then
        log_success "API health endpoint is responding"
        add_result "API Health" "PASS"
    else
        log_warning "API health endpoint not responding"
        add_result "API Health" "WARN"
    fi

    return 0
}

verify_database_connectivity() {
    log_section "Verifying Database Connectivity"

    # Check PostgreSQL pods
    if ! kubectl get statefulset -n victor postgresql &>/dev/null; then
        log_warning "PostgreSQL not found"
        add_result "Database Connectivity" "WARN"
        return 0
    fi

    # Check if PostgreSQL is ready
    local pg_ready
    pg_ready=$(kubectl get statefulset postgresql -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

    if [[ "${pg_ready}" -lt 1 ]]; then
        log_error "PostgreSQL is not ready"
        add_result "Database Connectivity" "FAIL"
        add_error "PostgreSQL not ready"
        return 1
    fi

    log_success "PostgreSQL is ready"

    # Test database connection
    local pg_pod
    pg_pod=$(kubectl get pods -n victor -l app=postgresql -o name | head -n1)

    if [[ -n "${pg_pod}" ]]; then
        if kubectl exec -n victor "${pg_pod}" -- pg_isready -U victor &>/dev/null; then
            log_success "Database connection successful"
            add_result "Database Connectivity" "PASS"
        else
            log_error "Database connection failed"
            add_result "Database Connectivity" "FAIL"
            add_error "Database connection failed"
        fi
    fi

    return 0
}

verify_redis_connectivity() {
    log_section "Verifying Redis Connectivity"

    # Check Redis pods
    if ! kubectl get statefulset -n victor redis &>/dev/null; then
        log_warning "Redis not found"
        add_result "Redis Connectivity" "WARN"
        return 0
    fi

    # Check if Redis is ready
    local redis_ready
    redis_ready=$(kubectl get statefulset redis -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

    if [[ "${redis_ready}" -lt 1 ]]; then
        log_error "Redis is not ready"
        add_result "Redis Connectivity" "FAIL"
        add_error "Redis not ready"
        return 1
    fi

    log_success "Redis is ready"

    # Test Redis connection
    local redis_pod
    redis_pod=$(kubectl get pods -n victor -l app=redis -o name | head -n1)

    if [[ -n "${redis_pod}" ]]; then
        if kubectl exec -n victor "${redis_pod}" -- redis-cli ping 2>/dev/null | grep -q "PONG"; then
            log_success "Redis connection successful"
            add_result "Redis Connectivity" "PASS"
        else
            log_warning "Redis connection failed"
            add_result "Redis Connectivity" "WARN"
        fi
    fi

    return 0
}

run_smoke_tests() {
    log_section "Running Smoke Tests"

    # Get endpoint
    local service_ip
    service_ip=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    local service_host
    service_host=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

    local endpoint="${service_ip}"
    if [[ -z "${endpoint}" ]]; then
        endpoint="${service_host}"
    fi

    if [[ -z "${endpoint}" ]]; then
        log_error "No endpoint available for smoke tests"
        add_result "Smoke Tests" "FAIL"
        add_error "No endpoint for smoke tests"
        return 1
    fi

    log_info "Running smoke tests against ${endpoint}..."

    # Test 1: Basic health check
    log_info "Test 1: Health check endpoint..."
    if curl -sf "http://${endpoint}/health" | grep -q "healthy\|ok"; then
        log_success "Health check passed"
        add_result "Smoke Test: Health" "PASS"
    else
        log_error "Health check failed"
        add_result "Smoke Test: Health" "FAIL"
        add_error "Health check failed"
    fi

    # Test 2: API version endpoint
    log_info "Test 2: API version endpoint..."
    if curl -sf "http://${endpoint}/api/v1/version" &>/dev/null; then
        log_success "API version check passed"
        add_result "Smoke Test: Version" "PASS"
    else
        log_warning "API version check failed"
        add_result "Smoke Test: Version" "WARN"
    fi

    # Test 3: Readiness check
    log_info "Test 3: Readiness check..."
    if curl -sf "http://${endpoint}/ready" | grep -q "ready\|ok"; then
        log_success "Readiness check passed"
        add_result "Smoke Test: Ready" "PASS"
    else
        log_warning "Readiness check failed"
        add_result "Smoke Test: Ready" "WARN"
    fi

    return 0
}

verify_monitoring() {
    log_section "Verifying Monitoring"

    # Check Prometheus targets
    log_info "Checking Prometheus targets..."

    local prometheus_pod
    prometheus_pod=$(kubectl get pods -n monitoring -l app=prometheus -o name | head -n1 2>/dev/null || echo "")

    if [[ -n "${prometheus_pod}" ]]; then
        local targets_up
        targets_up=$(kubectl exec -n monitoring "${prometheus_pod}" -- wget -qO- 'http://localhost:9090/api/v1/targets' 2>/dev/null | jq -r '.data.activeTargets[] | select(.health=="up") | .job' | wc -l || echo "0")

        log_info "Prometheus targets up: ${targets_up}"

        if [[ ${targets_up} -gt 0 ]]; then
            log_success "Prometheus is scraping targets"
            add_result "Monitoring: Prometheus" "PASS"
        else
            log_warning "No Prometheus targets up"
            add_result "Monitoring: Prometheus" "WARN"
        fi
    else
        log_warning "Prometheus pod not found"
        add_result "Monitoring: Prometheus" "WARN"
    fi

    # Check Grafana dashboards
    log_info "Checking Grafana dashboards..."

    local grafana_pod
    grafana_pod=$(kubectl get pods -n monitoring -l app=grafana -o name | head -n1 2>/dev/null || echo "")

    if [[ -n "${grafana_pod}" ]]; then
        local dashboards
        dashboards=$(kubectl exec -n monitoring "${grafana_pod}" -- wget -qO- 'http://localhost:3000/api/search' 2>/dev/null | jq -r '.[] | .title' | wc -l || echo "0")

        log_info "Grafana dashboards: ${dashboards}"

        if [[ ${dashboards} -gt 0 ]]; then
            log_success "Grafana dashboards available"
            add_result "Monitoring: Grafana" "PASS"
        else
            log_warning "No Grafana dashboards found"
            add_result "Monitoring: Grafana" "WARN"
        fi
    else
        log_warning "Grafana pod not found"
        add_result "Monitoring: Grafana" "WARN"
    fi

    return 0
}

verify_logs() {
    log_section "Analyzing Logs"

    # Check for errors in recent logs
    log_info "Analyzing recent application logs..."

    local error_count
    error_count=$(kubectl logs -n victor -l app=victor --tail=100 --all-containers=true 2>/dev/null | grep -i "error\|exception\|failed" | wc -l || echo "0")

    log_info "Error count in recent logs: ${error_count}"

    if [[ ${error_count} -gt 10 ]]; then
        log_warning "High error count in logs: ${error_count}"
        add_result "Log Analysis" "WARN"
        add_error "High error count: ${error_count}"
    elif [[ ${error_count} -gt 0 ]]; then
        log_warning "Found ${error_count} errors in recent logs"
        add_result "Log Analysis" "WARN"
    else
        log_success "No errors found in recent logs"
        add_result "Log Analysis" "PASS"
    fi

    # Check for critical errors
    log_info "Checking for critical errors..."

    local critical_count
    critical_count=$(kubectl logs -n victor -l app=victor --tail=100 --all-containers=true 2>/dev/null | grep -i "critical\|fatal\|panic" | wc -l || echo "0")

    if [[ ${critical_count} -gt 0 ]]; then
        log_error "Found ${critical_count} critical errors in logs"
        add_result "Critical Errors" "FAIL"
        add_error "Critical errors found: ${critical_count}"
    else
        log_success "No critical errors found"
        add_result "Critical Errors" "PASS"
    fi

    return 0
}

verify_performance() {
    log_section "Verifying Performance"

    # Get endpoint
    local service_ip
    service_ip=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    local service_host
    service_host=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

    local endpoint="${service_ip}"
    if [[ -z "${endpoint}" ]]; then
        endpoint="${service_host}"
    fi

    if [[ -z "${endpoint}" ]]; then
        log_warning "No endpoint available for performance tests"
        add_result "Performance" "WARN"
        return 0
    fi

    # Test response time
    log_info "Testing response time..."

    local response_time
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://${endpoint}/health" 2>/dev/null || echo "0")

    log_info "Response time: ${response_time}s"

    if [[ $(echo "${response_time} < 1.0" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        log_success "Response time is good (< 1s)"
        add_result "Response Time" "PASS"
    elif [[ $(echo "${response_time} < 3.0" | bc -l 2>/dev/null || echo "0") -eq 1 ]]; then
        log_warning "Response time is acceptable (< 3s)"
        add_result "Response Time" "WARN"
    else
        log_error "Response time is too high (> 3s)"
        add_result "Response Time" "FAIL"
        add_error "High response time: ${response_time}s"
    fi

    # Check resource usage
    log_info "Checking resource usage..."

    local cpu_usage
    local memory_usage
    cpu_usage=$(kubectl top pods -n victor -l app=victor 2>/dev/null | awk 'NR>1 {sum+=$2} END {print sum}' || echo "0")
    memory_usage=$(kubectl top pods -n victor -l app=victor 2>/dev/null | awk 'NR>1 {sum+=$3} END {print sum}' || echo "0")

    log_info "CPU usage: ${cpu_usage}m"
    log_info "Memory usage: ${memory_usage}Mi"

    return 0
}

generate_report() {
    log_section "Generating Verification Report"

    local report_json="${VERIFICATION_REPORT}"

    cat > "${report_json}" << EOF
{
  "environment": "${ENVIRONMENT}",
  "verification_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "verification_score": ${VERIFICATION_SCORE},
  "max_score": ${MAX_SCORE},
  "health_percentage": $(( VERIFICATION_SCORE * 100 / MAX_SCORE )),
  "verifications": [
EOF

    local first=true
    for result in "${VERIFICATION_RESULTS[@]}"; do
        local check_name
        local status
        check_name=$(echo "${result}" | cut -d'|' -f1)
        status=$(echo "${result}" | cut -d'|' -f2)

        if [[ "${first}" == "true" ]]; then
            first=false
        else
            echo "," >> "${report_json}"
        fi

        cat >> "${report_json}" << EOF
    {
      "name": "${check_name}",
      "status": "${status}"
    }
EOF
    done

    cat >> "${report_json}" << EOF
  ],
  "errors": [
EOF

    first=true
    for error in "${VERIFICATION_ERRORS[@]}"; do
        if [[ "${first}" == "true" ]]; then
            first=false
        else
            echo "," >> "${report_json}"
        fi

        cat >> "${report_json}" << EOF
    "${error}"
EOF
    done

    cat >> "${report_json}" << EOF

  ]
}
EOF

    log_success "Report generated: ${report_json}"

    # Print summary
    echo ""
    echo "==================================================================="
    echo "VERIFICATION SUMMARY"
    echo "==================================================================="
    echo "Environment:        ${ENVIRONMENT}"
    echo "Verification Score: ${VERIFICATION_SCORE}/${MAX_SCORE}"
    echo "Health Percentage:  $(( VERIFICATION_SCORE * 100 / MAX_SCORE ))%"
    echo ""
    echo "Verification Results:"
    printf "%-30s %-10s\n" "Check" "Status"
    printf "%-30s %-10s\n" "-----" "------"

    for result in "${VERIFICATION_RESULTS[@]}"; do
        local check_name
        local status
        check_name=$(echo "${result}" | cut -d'|' -f1)
        status=$(echo "${result}" | cut -d'|' -f2)

        case "${status}" in
            PASS)
                printf "%-30s ${GREEN}%-10s${NC}\n" "${check_name}" "${status}"
                ;;
            WARN)
                printf "%-30s ${YELLOW}%-10s${NC}\n" "${check_name}" "${status}"
                ;;
            FAIL)
                printf "%-30s ${RED}%-10s${NC}\n" "${check_name}" "${status}"
                ;;
        esac
    done

    if [[ ${#VERIFICATION_ERRORS[@]} -gt 0 ]]; then
        echo ""
        echo "Errors:"
        for error in "${VERIFICATION_ERRORS[@]}"; do
            echo "  - ${error}"
        done
    fi

    echo ""
    echo "Report saved to: ${report_json}"

    # Check if deployment is healthy
    if [[ ${VERIFICATION_SCORE} -ge 80 ]]; then
        log_success "Deployment is healthy (Score: ${VERIFICATION_SCORE}/${MAX_SCORE})"
        return 0
    else
        log_error "Deployment has issues (Score: ${VERIFICATION_SCORE}/${MAX_SCORE})"
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    log_info "Starting post-deployment verification for ${ENVIRONMENT}"

    # Run verifications
    verify_deployment_health
    verify_service_endpoints
    verify_database_connectivity
    verify_redis_connectivity
    run_smoke_tests
    verify_monitoring
    verify_logs
    verify_performance

    # Generate report
    generate_report

    local exit_code=$?

    echo ""
    echo "Verification completed at: $(date)" | tee -a "${VERIFICATION_LOG}"

    exit ${exit_code}
}

# Run main function
main "$@"
