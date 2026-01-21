#!/bin/bash
################################################################################
# Victor AI Post-Restore Verification Script
# Purpose: Verify system health after disaster recovery
# Usage: ./verify_restore.sh [full|quick]
# Dependencies: kubectl, curl, jq
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
LOG_FILE="/var/log/backups/verify_restore.log"
REPORT_FILE="/tmp/restore_verification_report_${TIMESTAMP}.html"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERIFICATION_TYPE="${1:-full}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Verification results
declare -A CHECK_RESULTS
declare -A CHECK_MESSAGES
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    log "INFO" "${GREEN}$@${NC}"
}

log_warn() {
    log "WARN" "${YELLOW}$@${NC}"
}

log_error() {
    log "ERROR" "${RED}$@${NC}"
}

log_pass() {
    log "PASS" "${GREEN}✓ $@${NC}"
}

log_fail() {
    log "FAIL" "${RED}✗ $@${NC}"
}

send_alert() {
    local severity=$1
    local message=$2

    if [ -n "${ALERT_WEBHOOK}" ]; then
        local emoji=":warning:"
        [ "${severity}" = "CRITICAL" ] && emoji=":rotating_light:"
        [ "${severity}" = "SUCCESS" ] && emoji=":white_check_mark:"

        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"${emoji} Restore Verification ${severity}\", \"blocks\": [{\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": \"*${severity}*: ${message}\"}}]}" 2>/dev/null || true
    fi
}

# ============================================================================
# VERIFICATION CHECKS
# ============================================================================

check_kubernetes_cluster() {
    log_info "Checking Kubernetes cluster health..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    # Check if cluster is accessible
    if ! kubectl cluster-info &>/dev/null; then
        CHECK_RESULTS[kubernetes_cluster]="FAIL"
        CHECK_MESSAGES[kubernetes_cluster]="Cluster API is not accessible"
        log_fail "Kubernetes cluster is not accessible"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi

    # Check node status
    local node_count=$(kubectl get nodes --no-headers | wc -l)
    local ready_nodes=$(kubectl get nodes --no-headers | grep -c " Ready ")

    if [ ${ready_nodes} -lt ${node_count} ]; then
        CHECK_RESULTS[kubernetes_cluster]="WARNING"
        CHECK_MESSAGES[kubernetes_cluster]="Only ${ready_nodes}/${node_count} nodes are ready"
        log_warn "Only ${ready_nodes}/${node_count} nodes are ready"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        return 0
    fi

    CHECK_RESULTS[kubernetes_cluster]="PASS"
    CHECK_MESSAGES[kubernetes_cluster]="All ${node_count} nodes are ready"
    log_pass "Kubernetes cluster: All ${node_count} nodes ready"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    return 0
}

check_pods() {
    log_info "Checking pod status..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    local pod_count=$(kubectl get pods -n "${NAMESPACE}" --no-headers | wc -l)
    local running_pods=$(kubectl get pods -n "${NAMESPACE}" --no-headers | grep -c "Running ")

    if [ ${running_pods} -lt ${pod_count} ]; then
        CHECK_RESULTS[pods]="FAIL"
        CHECK_MESSAGES[pods]="Only ${running_pods}/${pod_count} pods are running"

        # Show non-running pods
        log_fail "Pod status: Only ${running_pods}/${pod_count} pods running"
        kubectl get pods -n "${NAMESPACE}" | grep -v "Running"

        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi

    CHECK_RESULTS[pods]="PASS"
    CHECK_MESSAGES[pods]="All ${pod_count} pods are running"
    log_pass "Pods: All ${pod_count} pods running"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    return 0
}

check_pvcs() {
    log_info "Checking PVC status..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    local pvc_count=$(kubectl get pvc -n "${NAMESPACE}" --no-headers | wc -l)
    local bound_pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o jsonpath='{.items[?(@.status.phase=="Bound")].metadata.name}' | wc -w)

    if [ ${bound_pvcs} -lt ${pvc_count} ]; then
        CHECK_RESULTS[pvcs]="FAIL"
        CHECK_MESSAGES[pvcs]="Only ${bound_pvcs}/${pvc_count} PVCs are bound"

        log_fail "PVCs: Only ${bound_pvcs}/${pvc_count} PVCs bound"
        kubectl get pvc -n "${NAMESPACE}"

        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi

    CHECK_RESULTS[pvcs]="PASS"
    CHECK_MESSAGES[pvcs]="All ${pvc_count} PVCs are bound"
    log_pass "PVCs: All ${pvc_count} PVCs bound"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    return 0
}

check_application_health() {
    log_info "Checking application health endpoint..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    # Try to get a pod to execute health check
    local pod=$(kubectl get pods -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${pod}" ]; then
        CHECK_RESULTS[application_health]="SKIP"
        CHECK_MESSAGES[application_health]="No application pods found"
        log_warn "No application pods found, skipping health check"
        return 0
    fi

    # Check health endpoint
    if kubectl exec -n "${NAMESPACE}" "${pod}" -- curl -f http://localhost:8000/health &>/dev/null; then
        CHECK_RESULTS[application_health]="PASS"
        CHECK_MESSAGES[application_health]="Application health endpoint is responding"
        log_pass "Application health: OK"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        CHECK_RESULTS[application_health]="FAIL"
        CHECK_MESSAGES[application_health]="Application health endpoint is not responding"
        log_fail "Application health: Not responding"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

check_database_connectivity() {
    log_info "Checking database connectivity..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    # Check if PostgreSQL exists
    if ! kubectl get statefulset -n "${NAMESPACE}" postgresql &>/dev/null; then
        CHECK_RESULTS[database_connectivity]="SKIP"
        CHECK_MESSAGES[database_connectivity]="PostgreSQL not found"
        log_warn "PostgreSQL not found, skipping database check"
        return 0
    fi

    # Get application pod
    local pod=$(kubectl get pods -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${pod}" ]; then
        CHECK_RESULTS[database_connectivity]="SKIP"
        CHECK_MESSAGES[database_connectivity]="No application pods to test connectivity"
        log_warn "No application pods, skipping database check"
        return 0
    fi

    # Test database connection from application pod
    if kubectl exec -n "${NAMESPACE}" "${pod}" -- python -c "
import os
import sys
try:
    import psycopg2
    conn = psycopg2.connect(os.environ.get('DATABASE_URL', ''))
    conn.close()
    print('OK')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | grep -q "OK"; then
        CHECK_RESULTS[database_connectivity]="PASS"
        CHECK_MESSAGES[database_connectivity]="Database connectivity verified"
        log_pass "Database connectivity: OK"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        CHECK_RESULTS[database_connectivity]="FAIL"
        CHECK_MESSAGES[database_connectivity]="Cannot connect to database"
        log_fail "Database connectivity: FAILED"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

check_redis_connectivity() {
    log_info "Checking Redis connectivity..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    # Check if Redis exists
    if ! kubectl get statefulset -n "${NAMESPACE}" redis &>/dev/null && \
       ! kubectl get deployment -n "${NAMESPACE}" redis &>/dev/null; then
        CHECK_RESULTS[redis_connectivity]="SKIP"
        CHECK_MESSAGES[redis_connectivity]="Redis not found"
        log_warn "Redis not found, skipping Redis check"
        return 0
    fi

    # Get Redis pod
    local redis_pod=$(kubectl get pods -n "${NAMESPACE}" -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${redis_pod}" ]; then
        CHECK_RESULTS[redis_connectivity]="FAIL"
        CHECK_MESSAGES[redis_connectivity]="No Redis pods found"
        log_fail "Redis: No pods found"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi

    # Test Redis
    if kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- redis-cli PING | grep -q "PONG"; then
        CHECK_RESULTS[redis_connectivity]="PASS"
        CHECK_MESSAGES[redis_connectivity]="Redis is responding"
        log_pass "Redis: OK"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        CHECK_RESULTS[redis_connectivity]="FAIL"
        CHECK_MESSAGES[redis_connectivity]="Redis is not responding"
        log_fail "Redis: FAILED"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

check_resource_utilization() {
    log_info "Checking resource utilization..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    # Check pod resource requests vs limits
    local high_cpu_pods=$(kubectl top pods -n "${NAMESPACE}" --no-headers 2>/dev/null | awk '$3 > 80 {print $1}' | wc -l || echo "0")

    if [ ${high_cpu_pods} -gt 0 ]; then
        CHECK_RESULTS[resource_utilization]="WARNING"
        CHECK_MESSAGES[resource_utilization]="${high_cpu_pods} pods with high CPU usage"
        log_warn "Resource utilization: ${high_cpu_pods} pods with high CPU"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        return 0
    fi

    CHECK_RESULTS[resource_utilization]="PASS"
    CHECK_MESSAGES[resource_utilization]="Resource utilization is normal"
    log_pass "Resource utilization: Normal"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    return 0
}

check_application_logs() {
    log_info "Checking application logs for errors..."

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    if [ "${VERIFICATION_TYPE}" = "quick" ]; then
        log_info "Skipping log check in quick mode"
        CHECK_RESULTS[application_logs]="SKIP"
        CHECK_MESSAGES[application_logs]="Skipped in quick mode"
        return 0
    fi

    # Check for errors in recent logs
    local error_count=$(kubectl logs -n "${NAMESPACE}" -l app=victor-ai --tail=100 2>/dev/null | grep -i "error\|exception\|fatal" | wc -l || echo "0")

    if [ ${error_count} -gt 10 ]; then
        CHECK_RESULTS[application_logs]="WARNING"
        CHECK_MESSAGES[application_logs]="Found ${error_count} errors in recent logs"
        log_warn "Application logs: ${error_count} errors found"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
        return 0
    fi

    CHECK_RESULTS[application_logs]="PASS"
    CHECK_MESSAGES[application_logs]="No critical errors in logs"
    log_pass "Application logs: No critical errors"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    return 0
}

# ============================================================================
# REPORT GENERATION
# ============================================================================

generate_report() {
    log_info "Generating verification report..."

    local report_file="/tmp/restore_verification_report_$(date +%Y%m%d_%H%M%S).html"

    cat > "${report_file}" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>Victor AI Restore Verification Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 32px; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary-card h3 { margin: 0 0 10px 0; font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
        .summary-card .value { font-size: 32px; font-weight: bold; color: #333; }
        .summary-card.pass .value { color: #10b981; }
        .summary-card.fail .value { color: #ef4444; }
        .summary-card.warn .value { color: #f59e0b; }
        .section { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section h2 { margin: 0 0 20px 0; font-size: 20px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        th { background: #f9fafb; font-weight: 600; color: #374151; }
        .status-pass { color: #10b981; font-weight: 600; }
        .status-fail { color: #ef4444; font-weight: 600; }
        .status-warn { color: #f59e0b; font-weight: 600; }
        .status-skip { color: #6b7280; font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Victor AI Restore Verification Report</h1>
        <p>Generated: $(date '+%Y-%m-%d %H:%M:%S') | Environment: ${NAMESPACE}</p>
    </div>

    <div class="summary">
        <div class="summary-card $( [ ${FAILED_CHECKS} -eq 0 ] && echo 'pass' || echo 'fail' )">
            <h3>Overall Status</h3>
            <div class="value">$( [ ${FAILED_CHECKS} -eq 0 ] && echo 'PASSED' || echo 'FAILED' )</div>
        </div>
        <div class="summary-card">
            <h3>Total Checks</h3>
            <div class="value">${TOTAL_CHECKS}</div>
        </div>
        <div class="summary-card pass">
            <h3>Passed</h3>
            <div class="value">${PASSED_CHECKS}</div>
        </div>
        <div class="summary-card fail">
            <h3>Failed</h3>
            <div class="value">${FAILED_CHECKS}</div>
        </div>
        <div class="summary-card warn">
            <h3>Warnings</h3>
            <div class="value">${WARNING_CHECKS}</div>
        </div>
    </div>

    <div class="section">
        <h2>Verification Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Check</th>
                    <th>Status</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
EOF

    # Add rows for each check
    for check in "${!CHECK_RESULTS[@]}"; do
        local status="${CHECK_RESULTS[$check]}"
        local message="${CHECK_MESSAGES[$check]}"
        local status_class="status-${status,,}"

        echo "                <tr>" >> "${report_file}"
        echo "                    <td>${check}</td>" >> "${report_file}"
        echo "                    <td class=\"${status_class}\">${status}</td>" >> "${report_file}"
        echo "                    <td>${message}</td>" >> "${report_file}"
        echo "                </tr>" >> "${report_file}"
    done

    cat >> "${report_file}" <<EOF
            </tbody>
        </table>
    </div>

    <div style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 30px;">
        <p>This report was automatically generated by Victor AI Restore Verification System</p>
        <p>For questions, contact: platform-engineering@victor.ai</p>
    </div>
</body>
</html>
EOF

    log_info "Report generated: ${report_file}"

    # Send report via email if configured
    if command -v mail &>/dev/null && [ -n "${ADMIN_EMAIL:-}" ]; then
        mail -s "Victor AI Restore Verification Report - $(date +%Y-%m-%d)" \
            "${ADMIN_EMAIL}" < "${report_file}"
        log_info "Report sent to ${ADMIN_EMAIL}"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Restore Verification ==="
    log_info "Verification type: ${VERIFICATION_TYPE}"
    log_info "Namespace: ${NAMESPACE}"

    local overall_start_time=$(date +%s)

    # Run verification checks
    check_kubernetes_cluster
    check_pods
    check_pvcs
    check_application_health
    check_database_connectivity
    check_redis_connectivity
    check_resource_utilization
    check_application_logs

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    # Generate report
    generate_report

    # Print summary
    echo ""
    log_info "=== Verification Summary ==="
    log_info "Total checks: ${TOTAL_CHECKS}"
    log_pass "Passed: ${PASSED_CHECKS}"
    [ ${FAILED_CHECKS} -gt 0 ] && log_fail "Failed: ${FAILED_CHECKS}"
    [ ${WARNING_CHECKS} -gt 0 ] && log_warn "Warnings: ${WARNING_CHECKS}"
    log_info "Duration: ${total_duration}s"
    echo ""

    # Determine overall result
    if [ ${FAILED_CHECKS} -eq 0 ]; then
        log_info "=== Restore Verification PASSED ==="
        send_alert "SUCCESS" "Restore verification passed: ${PASSED_CHECKS}/${TOTAL_CHECKS} checks passed"
        exit 0
    else
        log_error "=== Restore Verification FAILED ==="
        send_alert "CRITICAL" "Restore verification failed: ${FAILED_CHECKS}/${TOTAL_CHECKS} checks failed"
        exit 1
    fi
}

# Run main function
main "$@"
