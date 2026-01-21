#!/bin/bash
################################################################################
# Production Rollback Script
#
# This script performs automated rollback with health-based verification,
# traffic shifting, and rollback reporting.
#
# Usage:
#   ./rollback_production.sh [environment] [options]
#
# Arguments:
#   environment - Target environment (production, staging, development)
#
# Options:
#   --revision REVISION - Rollback to specific revision
#   --backup BACKUP_ID - Rollback to specific backup
#   --force - Force rollback without confirmation
#   --dry-run - Perform a dry run
#
# Features:
# - One-command rollback
# - Health-based verification
# - Traffic shifting
# - Rollback reporting
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ROOT="${PROJECT_ROOT}/deployment"
LOG_DIR="${PROJECT_ROOT}/logs"
REPORT_DIR="${PROJECT_ROOT}/reports"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Rollback state
ROLLBACK_ID=""
ENVIRONMENT=""
TARGET_REVISION=""
TARGET_BACKUP=""
FORCE_ROLLBACK=false
DRY_RUN=false

# Rollback tracking
declare -a ROLLBACK_STEPS

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

add_rollback_step() {
    ROLLBACK_STEPS+=("$1|$2")
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --revision)
                TARGET_REVISION="$2"
                shift 2
                ;;
            --backup)
                TARGET_BACKUP="$2"
                shift 2
                ;;
            --force)
                FORCE_ROLLBACK=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                ENVIRONMENT="$1"
                shift
                ;;
        esac
    done

    if [[ -z "${ENVIRONMENT}" ]]; then
        ENVIRONMENT="production"
    fi
}

show_help() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [OPTIONS]

Arguments:
  ENVIRONMENT       Target environment (production, staging, development)

Options:
  --revision N      Rollback to specific deployment revision
  --backup ID       Rollback to specific backup
  --force          Force rollback without confirmation
  --dry-run        Perform a dry run without making changes
  --help           Show this help message

Examples:
  $0 production
  $0 production --revision 5
  $0 staging --backup pre-deploy-20231201-120000
  $0 production --force --dry-run
EOF
}

################################################################################
# Pre-Rollback Setup
################################################################################

initialize_rollback() {
    log_section "Initializing Rollback"

    # Generate rollback ID
    ROLLBACK_ID="rollback-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"

    # Setup directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${REPORT_DIR}"

    # Setup log file
    export ROLLBACK_LOG_FILE="${LOG_DIR}/rollback_${ROLLBACK_ID}.log"

    log_info "Rollback ID: ${ROLLBACK_ID}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Dry Run: ${DRY_RUN}"

    # Set kubeconfig
    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Verify cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to cluster"
        exit 1
    fi

    log_success "Rollback initialization complete"
    add_rollback_step "Initialize Rollback" "SUCCESS"
}

confirm_rollback() {
    log_section "Confirm Rollback"

    if [[ "${FORCE_ROLLBACK}" == "true" ]]; then
        log_warning "Skipping confirmation (force mode)"
        return 0
    fi

    echo ""
    echo -e "${YELLOW}WARNING: This will rollback the ${ENVIRONMENT} environment${NC}"
    echo ""
    echo "Rollback Details:"
    echo "  Environment: ${ENVIRONMENT}"
    echo "  Revision: ${TARGET_REVISION:-latest}"
    echo "  Backup: ${TARGET_BACKUP:-none}"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirmation

    if [[ "${confirmation}" != "yes" ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi

    log_success "Rollback confirmed"
    add_rollback_step "Confirm Rollback" "SUCCESS"
}

capture_current_state() {
    log_section "Capturing Current State"

    log_info "Capturing current deployment state..."

    local state_file="${LOG_DIR}/rollback_${ROLLBACK_ID}_current_state.yaml"

    if kubectl get all,configmaps,secrets -n victor -o yaml > "${state_file}" 2>/dev/null; then
        log_success "Current state captured: ${state_file}"
        add_rollback_step "Capture Current State" "SUCCESS"
    else
        log_warning "Failed to capture current state"
        add_rollback_step "Capture Current State" "WARN"
    fi
}

################################################################################
# Rollback Functions
################################################################################

rollback_deployment() {
    log_section "Rolling Back Deployment"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would perform rollback"
        add_rollback_step "Rollback Deployment" "SUCCESS"
        return 0
    fi

    # Rollback to specific revision if provided
    if [[ -n "${TARGET_REVISION}" ]]; then
        log_info "Rolling back to revision ${TARGET_REVISION}..."

        if ! kubectl rollout undo deployment/victor -n victor --to-revision="${TARGET_REVISION}" &>/dev/null; then
            log_error "Failed to rollback to revision ${TARGET_REVISION}"
            return 1
        fi
    else
        log_info "Rolling back to previous revision..."

        if ! kubectl rollout undo deployment/victor -n victor &>/dev/null; then
            log_error "Failed to rollback deployment"
            return 1
        fi
    fi

    log_success "Rollback command executed"
    add_rollback_step "Rollback Deployment" "SUCCESS"

    return 0
}

rollback_from_backup() {
    log_section "Restoring from Backup"

    if [[ -z "${TARGET_BACKUP}" ]]; then
        log_info "No backup specified, skipping restore"
        return 0
    fi

    if ! command -v velero &>/dev/null; then
        log_error "Velero not found, cannot restore from backup"
        return 1
    fi

    log_info "Restoring from backup: ${TARGET_BACKUP}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would restore from backup"
        add_rollback_step "Restore Backup" "SUCCESS"
        return 0
    fi

    # Create restore
    local restore_name="restore-${TARGET_BACKUP}-$(date +%Y%m%d-%H%M%S)"

    if velero restore create "${restore_name}" \
        --from-backup "${TARGET_BACKUP}" \
        --namespace-mappings victor:victor \
        --wait &>/dev/null; then
        log_success "Backup restored successfully"
        add_rollback_step "Restore Backup" "SUCCESS"
    else
        log_error "Failed to restore from backup"
        return 1
    fi

    return 0
}

wait_for_rollback() {
    log_section "Waiting for Rollback"

    log_info "Waiting for deployment rollout..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would wait for rollout"
        return 0
    fi

    # Wait for rollout
    local timeout=300
    local elapsed=0
    local interval=5

    while [[ ${elapsed} -lt ${timeout} ]]; do
        local rollout_status
        rollout_status=$(kubectl rollout status deployment/victor -n victor --timeout=1s 2>&1 || echo "failed")

        if [[ "${rollout_status}" =~ "successfully rolled out" ]]; then
            log_success "Rollback completed successfully"
            add_rollback_step "Wait for Rollback" "SUCCESS"
            return 0
        fi

        sleep "${interval}"
        elapsed=$((elapsed + interval))
    done

    log_error "Rollback timed out after ${timeout}s"
    return 1
}

verify_rollback_health() {
    log_section "Verifying Rollback Health"

    log_info "Running health checks after rollback..."

    # Check deployment status
    local deployment_status
    deployment_status=$(kubectl get deployment victor -n victor -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "False")

    if [[ "${deployment_status}" != "True" ]]; then
        log_error "Deployment is not available after rollback"
        add_rollback_step "Verify Health" "FAIL"
        return 1
    fi

    log_success "Deployment is available"

    # Check replicas
    local replicas_ready
    local replicas_desired
    replicas_ready=$(kubectl get deployment victor -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    replicas_desired=$(kubectl get deployment victor -n victor -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

    log_info "Replicas: ${replicas_ready}/${replicas_desired} ready"

    if [[ "${replicas_ready}" -lt "${replicas_desired}" ]]; then
        log_warning "Not all replicas ready"
        add_rollback_step "Verify Health" "WARN"
    else
        log_success "All replicas ready"
        add_rollback_step "Verify Health" "PASS"
    fi

    # Check pods
    local pods_running
    local pods_total
    pods_running=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -c "Running" || echo "0")
    pods_total=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | wc -w || echo "0")

    log_info "Pods: ${pods_running}/${pods_total} running"

    if [[ "${pods_running}" -ne "${pods_total}" ]]; then
        log_warning "Not all pods running"
    else
        log_success "All pods running"
    fi

    # Test endpoint
    local service_ip
    service_ip=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

    if [[ -n "${service_ip}" ]]; then
        log_info "Testing endpoint: ${service_ip}"

        if curl -sf "http://${service_ip}/health" &>/dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed"
            add_rollback_step "Verify Health" "WARN"
        fi
    fi

    return 0
}

switch_traffic() {
    log_section "Switching Traffic"

    log_info "Updating service to point to rollback version..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would switch traffic"
        return 0
    fi

    # Get current color
    local current_color
    current_color=$(kubectl get service victor -n victor -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "")

    if [[ -z "${current_color}" ]]; then
        log_warning "Service not using color selector, skipping traffic switch"
        return 0
    fi

    # Switch to other color
    local new_color
    if [[ "${current_color}" == "blue" ]]; then
        new_color="green"
    else
        new_color="blue"
    fi

    log_info "Switching from ${current_color} to ${new_color}..."

    local service_patch="[{\"op\": \"replace\", \"path\": \"/spec/selector/color\", \"value\": \"${new_color}\"}]"

    if kubectl patch service victor -n victor --type=json -p="${service_patch}" &>/dev/null; then
        log_success "Traffic switched to ${new_color}"
        add_rollback_step "Switch Traffic" "SUCCESS"
    else
        log_warning "Failed to switch traffic"
        add_rollback_step "Switch Traffic" "WARN"
    fi

    return 0
}

################################################################################
# Reporting
################################################################################

generate_rollback_report() {
    log_section "Generating Rollback Report"

    local report_file="${REPORT_DIR}/rollback_${ROLLBACK_ID}.json"

    cat > "${report_file}" << EOF
{
  "rollback_id": "${ROLLBACK_ID}",
  "environment": "${ENVIRONMENT}",
  "target_revision": "${TARGET_REVISION}",
  "target_backup": "${TARGET_BACKUP}",
  "dry_run": ${DRY_RUN},
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "SUCCESS",
  "steps": [
EOF

    local first=true
    for step in "${ROLLBACK_STEPS[@]}"; do
        local step_name
        local step_status
        step_name=$(echo "${step}" | cut -d'|' -f1)
        step_status=$(echo "${step}" | cut -d'|' -f2)

        if [[ "${first}" == "true" ]]; then
            first=false
        else
            echo "," >> "${report_file}"
        fi

        cat >> "${report_file}" << EOF
    {
      "name": "${step_name}",
      "status": "${step_status}"
    }
EOF
    done

    cat >> "${report_file}" << EOF

  ]
}
EOF

    log_success "Report generated: ${report_file}"
}

print_summary() {
    echo ""
    echo "==================================================================="
    echo "ROLLBACK SUMMARY"
    echo "==================================================================="
    echo "Rollback ID:     ${ROLLBACK_ID}"
    echo "Environment:     ${ENVIRONMENT}"
    echo "Target Revision: ${TARGET_REVISION:-previous}"
    echo "Target Backup:   ${TARGET_BACKUP:-none}"
    echo "Status:          SUCCESS"
    echo ""
    echo "Rollback Steps:"
    printf "%-30s %-15s\n" "Step" "Status"
    printf "%-30s %-15s\n" "-----" "------"

    for step in "${ROLLBACK_STEPS[@]}"; do
        local step_name
        local step_status
        step_name=$(echo "${step}" | cut -d'|' -f1)
        step_status=$(echo "${step}" | cut -d'|' -f2)

        case "${step_status}" in
            SUCCESS)
                printf "%-30s ${GREEN}%-15s${NC}\n" "${step_name}" "${step_status}"
                ;;
            FAILED)
                printf "%-30s ${RED}%-15s${NC}\n" "${step_name}" "${step_status}"
                ;;
            WARN)
                printf "%-30s ${YELLOW}%-15s${NC}\n" "${step_name}" "${step_status}"
                ;;
        esac
    done

    echo ""
    echo "Logs:   ${ROLLBACK_LOG_FILE}"
    echo "Report: ${REPORT_DIR}/rollback_${ROLLBACK_ID}.json"
    echo "==================================================================="
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_arguments "$@"

    initialize_rollback
    confirm_rollback
    capture_current_state

    if rollback_deployment; then
        if wait_for_rollback; then
            verify_rollback_health
            switch_traffic
            rollback_from_backup
            generate_rollback_report
            print_summary
            log_success "Rollback completed successfully"
            exit 0
        fi
    fi

    # Rollback failed
    log_error "Rollback failed"
    generate_rollback_report
    exit 1
}

# Run main function
main "$@"
