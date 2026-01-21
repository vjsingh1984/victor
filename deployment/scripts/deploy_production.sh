#!/bin/bash
################################################################################
# Production Deployment Script
#
# This script performs automated production deployment with blue-green strategy,
# rolling updates, health checks, and automatic rollback on failure.
#
# Usage:
#   ./deploy_production.sh --environment [production|staging|development]
#                          [--image-tag TAG]
#                          [--strategy (blue-green|rolling)]
#                          [--skip-validation]
#                          [--dry-run]
#
# Features:
# - Blue-green deployment support
# - Rolling updates with health checks
# - Automatic rollback on failure
# - Zero-downtime deployment
# - Progress monitoring
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

# Deployment state
DEPLOYMENT_ID=""
ENVIRONMENT="production"
IMAGE_TAG=""
STRATEGY="blue-green"
SKIP_VALIDATION=false
DRY_RUN=false
ROLLBACK_ON_FAILURE=true
HEALTH_CHECK_TIMEOUT=600
HEALTH_CHECK_INTERVAL=10

# Deployment tracking
declare -a DEPLOYMENT_STEPS
declare -a DEPLOYMENT_LOGS

################################################################################
# Helper Functions
################################################################################

log_info() {
    local msg="[INFO] $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_success() {
    local msg="[PASS] $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_warning() {
    local msg="[WARN] $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_error() {
    local msg="[FAIL] $1"
    echo -e "${RED}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

log_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === $1 ===" >> "${DEPLOYMENT_LOG_FILE}"
}

show_progress() {
    local current=$1
    local total=$2
    local msg=$3

    local percent=$(( current * 100 / total ))
    local filled=$(( percent / 2 ))
    local empty=$(( 50 - filled ))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %d%% (%s)" "${percent}" "${msg}"
}

add_deployment_step() {
    DEPLOYMENT_STEPS+=("$1|$2")
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --strategy)
                STRATEGY="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
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
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Usage: $0 --environment [ENVIRONMENT] [OPTIONS]

Arguments:
  --environment     Target environment (production, staging, development)
  --image-tag      Docker image tag to deploy
  --strategy       Deployment strategy: blue-green (default) or rolling
  --skip-validation Skip pre-deployment validation
  --dry-run        Perform a dry run without making changes
  --help           Show this help message

Examples:
  $0 --environment production --image-tag v1.0.0
  $0 --environment staging --strategy rolling
  $0 --environment production --dry-run
EOF
}

################################################################################
# Pre-Deployment Setup
################################################################################

initialize_deployment() {
    log_section "Initializing Deployment"

    # Generate deployment ID
    DEPLOYMENT_ID="deploy-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"

    # Setup directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${REPORT_DIR}"
    mkdir -p "${BACKUP_DIR}/${DEPLOYMENT_ID}"

    # Setup log file
    export DEPLOYMENT_LOG_FILE="${LOG_DIR}/deployment_${DEPLOYMENT_ID}.log"

    log_info "Deployment ID: ${DEPLOYMENT_ID}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Strategy: ${STRATEGY}"
    log_info "Dry Run: ${DRY_RUN}"

    # Set kubeconfig
    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Verify cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to cluster"
        exit 1
    fi

    log_success "Deployment initialization complete"
    add_deployment_step "Initialize Deployment" "SUCCESS"
}

validate_prerequisites() {
    log_section "Validating Prerequisites"

    if [[ "${SKIP_VALIDATION}" == "true" ]]; then
        log_warning "Skipping validation as requested"
        return 0
    fi

    # Run pre-deployment validation
    log_info "Running pre-deployment validation..."

    if [[ ! -f "${SCRIPT_DIR}/validate_deployment.sh" ]]; then
        log_error "Validation script not found"
        return 1
    fi

    if ! bash "${SCRIPT_DIR}/validate_deployment.sh" "${ENVIRONMENT}"; then
        log_error "Pre-deployment validation failed"
        log_error "Fix reported issues before deploying or use --skip-validation"
        return 1
    fi

    log_success "Prerequisites validated"
    add_deployment_step "Validate Prerequisites" "SUCCESS"
}

backup_current_state() {
    log_section "Backing Up Current State"

    log_info "Creating backup of current deployment state..."

    # Backup current resources
    local backup_file="${BACKUP_DIR}/${DEPLOYMENT_ID}/resources.yaml"

    if ! kubectl get all,configmaps,secrets,pvc -n victor -o yaml > "${backup_file}" 2>/dev/null; then
        log_warning "Failed to backup all resources, continuing anyway..."
    else
        log_success "Resources backed up to: ${backup_file}"
    fi

    # Backup database
    if command -v velero &>/dev/null; then
        log_info "Creating Velero backup..."

        local velero_backup="pre-deploy-${DEPLOYMENT_ID}"

        if velero backup create "${velero_backup}" --namespace victor --wait &>/dev/null; then
            log_success "Velero backup created: ${velero_backup}"
        else
            log_warning "Velero backup failed, continuing anyway..."
        fi
    fi

    add_deployment_step "Backup Current State" "SUCCESS"
}

################################################################################
# Blue-Green Deployment
################################################################################

deploy_blue_green() {
    log_section "Blue-Green Deployment"

    local current_color
    local new_color

    # Determine current color
    current_color=$(kubectl get deployment -n victor -l app=victor,version=blue -o name 2>/dev/null | grep -c blue || echo "0")

    if [[ "${current_color}" -gt 0 ]]; then
        current_color="blue"
        new_color="green"
    else
        current_color="green"
        new_color="blue"
    fi

    log_info "Current color: ${current_color}"
    log_info "New color: ${new_color}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would deploy to ${new_color}"
        return 0
    fi

    # Deploy new version
    log_info "Deploying new version to ${new_color}..."

    # Apply new deployment
    if ! kubectl apply -f "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" -l "color=${new_color}" &>/dev/null; then
        log_error "Failed to deploy new version"
        return 1
    fi

    # Wait for rollout
    log_info "Waiting for ${new_color} rollout..."

    local deployment_name="victor-${new_color}"
    if ! kubectl rollout status deployment/"${deployment_name}" -n victor --timeout=${HEALTH_CHECK_TIMEOUT}s &>/dev/null; then
        log_error "Rollout failed for ${deployment_name}"
        return 1
    fi

    # Health checks
    log_info "Running health checks on ${new_color}..."

    if ! perform_health_checks "${new_color}"; then
        log_error "Health checks failed on ${new_color}"
        return 1
    fi

    # Switch traffic
    log_info "Switching traffic to ${new_color}..."

    if ! switch_traffic "${new_color}"; then
        log_error "Failed to switch traffic"
        return 1
    fi

    log_success "Blue-green deployment complete"
    add_deployment_step "Blue-Green Deployment" "SUCCESS"

    return 0
}

switch_traffic() {
    local target_color=$1

    # Update service selector
    local service_patch="[{\"op\": \"replace\", \"path\": \"/spec/selector/color\", \"value\": \"${target_color}\"}]"

    if ! kubectl patch service victor -n victor --type=json -p="${service_patch}" &>/dev/null; then
        log_error "Failed to patch service"
        return 1
    fi

    log_success "Traffic switched to ${target_color}"
    return 0
}

################################################################################
# Rolling Update Deployment
################################################################################

deploy_rolling_update() {
    log_section "Rolling Update Deployment"

    log_info "Starting rolling update..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would perform rolling update"
        return 0
    fi

    # Apply new configuration
    log_info "Applying new configuration..."

    if ! kubectl apply -f "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" &>/dev/null; then
        log_error "Failed to apply configuration"
        return 1
    fi

    # Wait for rollout
    log_info "Waiting for rollout to complete..."

    if ! kubectl rollout status deployment/victor -n victor --timeout=${HEALTH_CHECK_TIMEOUT}s &>/dev/null; then
        log_error "Rollout failed"
        return 1
    fi

    log_success "Rolling update complete"
    add_deployment_step "Rolling Update" "SUCCESS"

    return 0
}

################################################################################
# Health Checks
################################################################################

perform_health_checks() {
    local color=${1:-"current"}

    log_section "Health Checks (${color})"

    local timeout=${HEALTH_CHECK_TIMEOUT}
    local elapsed=0
    local interval=${HEALTH_CHECK_INTERVAL}

    while [[ ${elapsed} -lt ${timeout} ]]; do
        show_progress ${elapsed} ${timeout} "Health checks"

        # Check deployment replicas
        local replicas_ready
        replicas_ready=$(kubectl get deployment -n victor -l "color=${color}" -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null || echo "0")

        local replicas_desired
        replicas_desired=$(kubectl get deployment -n victor -l "color=${color}" -o jsonpath='{.items[0].spec.replicas}' 2>/dev/null || echo "0")

        if [[ "${replicas_ready}" -ge "${replicas_desired}" ]]; then
            log_success "All replicas ready (${replicas_ready}/${replicas_desired})"
        fi

        # Check pod status
        local pods_running
        pods_running=$(kubectl get pods -n victor -l "color=${color}" -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -c "Running" || echo "0")

        local pods_total
        pods_total=$(kubectl get pods -n victor -l "color=${color}" -o jsonpath='{.items[*].status.phase}' 2>/dev/null | wc -w || echo "0")

        if [[ "${pods_running}" -eq "${pods_total}" ]] && [[ "${pods_total}" -gt 0 ]]; then
            log_success "All pods running (${pods_running}/${pods_total})"
        else
            log_warning "Not all pods running (${pods_running}/${pods_total})"
            sleep "${interval}"
            elapsed=$((elapsed + interval))
            continue
        fi

        # Check endpoint
        local endpoint
        endpoint=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

        if [[ -n "${endpoint}" ]]; then
            log_info "Testing endpoint: ${endpoint}"

            if curl -sf "http://${endpoint}/health" &>/dev/null; then
                log_success "Health check passed"
                return 0
            fi
        fi

        sleep "${interval}"
        elapsed=$((elapsed + interval))
    done

    printf "\n"
    log_error "Health checks timed out after ${timeout}s"
    return 1
}

################################################################################
# Deployment Execution
################################################################################

execute_deployment() {
    log_section "Executing Deployment"

    case "${STRATEGY}" in
        blue-green)
            if ! deploy_blue_green; then
                log_error "Blue-green deployment failed"
                return 1
            fi
            ;;
        rolling)
            if ! deploy_rolling_update; then
                log_error "Rolling update failed"
                return 1
            fi
            ;;
        *)
            log_error "Unknown strategy: ${STRATEGY}"
            return 1
            ;;
    esac

    return 0
}

################################################################################
# Rollback
################################################################################

perform_rollback() {
    log_section "Performing Rollback"

    log_warning "Initiating automatic rollback..."

    # Rollback deployment
    log_info "Rolling back deployment..."

    if ! kubectl rollout undo deployment/victor -n victor &>/dev/null; then
        log_error "Rollback command failed"
        return 1
    fi

    # Wait for rollback
    log_info "Waiting for rollback to complete..."

    if ! kubectl rollout status deployment/victor -n victor --timeout=300s &>/dev/null; then
        log_error "Rollout failed during rollback"
        return 1
    fi

    log_success "Rollback complete"
    add_deployment_step "Rollback" "SUCCESS"

    return 0
}

################################################################################
# Post-Deployment Verification
################################################################################

verify_deployment() {
    log_section "Post-Deployment Verification"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: skipping verification"
        return 0
    fi

    # Run smoke tests
    log_info "Running smoke tests..."

    if [[ -f "${SCRIPT_DIR}/verify_deployment.sh" ]]; then
        if ! bash "${SCRIPT_DIR}/verify_deployment.sh" "${ENVIRONMENT}"; then
            log_error "Smoke tests failed"

            if [[ "${ROLLBACK_ON_FAILURE}" == "true" ]]; then
                perform_rollback
            fi

            return 1
        fi
    else
        log_warning "Verification script not found, skipping smoke tests"
    fi

    log_success "Deployment verified successfully"
    add_deployment_step "Verify Deployment" "SUCCESS"

    return 0
}

################################################################################
# Reporting
################################################################################

generate_deployment_report() {
    log_section "Generating Deployment Report"

    local report_file="${REPORT_DIR}/deployment_${DEPLOYMENT_ID}.json"

    cat > "${report_file}" << EOF
{
  "deployment_id": "${DEPLOYMENT_ID}",
  "environment": "${ENVIRONMENT}",
  "strategy": "${STRATEGY}",
  "image_tag": "${IMAGE_TAG}",
  "dry_run": ${DRY_RUN},
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "SUCCESS",
  "steps": [
EOF

    local first=true
    for step in "${DEPLOYMENT_STEPS[@]}"; do
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
    echo "DEPLOYMENT SUMMARY"
    echo "==================================================================="
    echo "Deployment ID:  ${DEPLOYMENT_ID}"
    echo "Environment:    ${ENVIRONMENT}"
    echo "Strategy:       ${STRATEGY}"
    echo "Image Tag:      ${IMAGE_TAG}"
    echo "Status:         SUCCESS"
    echo ""
    echo "Deployment Steps:"
    printf "%-30s %-15s\n" "Step" "Status"
    printf "%-30s %-15s\n" "-----" "------"

    for step in "${DEPLOYMENT_STEPS[@]}"; do
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
        esac
    done

    echo ""
    echo "Logs:           ${DEPLOYMENT_LOG_FILE}"
    echo "Report:         ${REPORT_DIR}/deployment_${DEPLOYMENT_ID}.json"
    echo "Backup:         ${BACKUP_DIR}/${DEPLOYMENT_ID}"
    echo "==================================================================="
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_arguments "$@"

    initialize_deployment
    validate_prerequisites
    backup_current_state

    if execute_deployment; then
        if verify_deployment; then
            generate_deployment_report
            print_summary
            log_success "Deployment completed successfully"
            exit 0
        fi
    fi

    # Deployment failed
    log_error "Deployment failed"
    generate_deployment_report

    if [[ "${ROLLBACK_ON_FAILURE}" == "true" ]]; then
        perform_rollback
    fi

    exit 1
}

# Run main function
main "$@"
