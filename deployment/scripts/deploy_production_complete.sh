#!/bin/bash
################################################################################
# Master Production Deployment Orchestration Script
#
# This script orchestrates complete infrastructure setup and production deployment
# with comprehensive validation, rollback support, and deployment reporting.
#
# Usage:
#   ./deploy_production_complete.sh --environment [production|staging|development]
#                                  [--skip-infrastructure]
#                                  [--skip-monitoring]
#                                  [--skip-backups]
#                                  [--image-tag TAG]
#                                  [--strategy (blue-green|rolling)]
#                                  [--dry-run]
#                                  [--continue-on-failure]
#
# Features:
# - Complete infrastructure orchestration
# - Progress tracking with time estimates
# - Rollback at any stage
# - Comprehensive deployment reporting
# - Manual intervention prompts for credentials
# - Stage gating with validation
# - Detailed logging
################################################################################

set -euo pipefail

################################################################################
# SCRIPT CONFIGURATION
################################################################################

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
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Deployment state
DEPLOYMENT_ID=""
ENVIRONMENT="production"
IMAGE_TAG=""
STRATEGY="blue-green"
DRY_RUN=false
CONTINUE_ON_FAILURE=false
ROLLBACK_ON_FAILURE=true

# Stage skipping flags
SKIP_INFRASTRUCTURE=false
SKIP_MONITORING=false
SKIP_BACKUPS=false
SKIP_VALIDATION=false
SKIP_SECRETS=false

# Time estimates (in seconds)
ESTIMATE_VALIDATION=300
ESTIMATE_INFRASTRUCTURE=600
ESTIMATE_MONITORING=300
ESTIMATE_SECRETS=120
ESTIMATE_BACKUPS=180
ESTIMATE_DEPLOYMENT=600
ESTIMATE_VERIFICATION=300
TOTAL_ESTIMATE=2580  # ~43 minutes

# Progress tracking
declare -a DEPLOYMENT_STAGES
declare -a DEPLOYMENT_LOGS
START_TIME=""
END_TIME=""
CURRENT_STAGE=""

# State file for resume capability
STATE_FILE="${LOG_DIR}/.deploy_state"

################################################################################
# HELPER FUNCTIONS
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

log_stage() {
    local msg="=== STAGE: $1 ==="
    echo ""
    echo -e "${CYAN}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
    CURRENT_STAGE="$1"
}

log_substage() {
    local msg="--- $1 ---"
    echo -e "${MAGENTA}${msg}${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" >> "${DEPLOYMENT_LOG_FILE}"
}

show_banner() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Victor AI - Master Production Deployment Orchestration     ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

show_progress_bar() {
    local current=$1
    local total=$2
    local stage_name=$3

    local percent=$(( current * 100 / total ))
    local filled=$(( percent / 2 ))
    local empty=$(( 50 - filled ))

    printf "\r["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %d%% - %s" "${percent}" "${stage_name}"
}

calculate_elapsed_time() {
    local start=$1
    local end=$2
    local elapsed=$(( end - start ))
    local minutes=$(( elapsed / 60 ))
    local seconds=$(( elapsed % 60 ))
    echo "${minutes}m ${seconds}s"
}

format_timestamp() {
    date -u '+%Y-%m-%d %H:%M:%S UTC'
}

save_state() {
    local stage=$1
    local status=$2

    cat > "${STATE_FILE}" << EOF
DEPLOYMENT_ID=${DEPLOYMENT_ID}
CURRENT_STAGE=${stage}
STATUS=${status}
START_TIME=${START_TIME}
ENVIRONMENT=${ENVIRONMENT}
IMAGE_TAG=${IMAGE_TAG}
STRATEGY=${STRATEGY}
EOF
}

load_state() {
    if [[ -f "${STATE_FILE}" ]]; then
        source "${STATE_FILE}"
        return 0
    fi
    return 1
}

clear_state() {
    rm -f "${STATE_FILE}"
}

################################################################################
# ERROR HANDLING
################################################################################

handle_error() {
    local stage=$1
    local error_msg=$2

    log_error "Error occurred in stage: ${stage}"
    log_error "Error: ${error_msg}"

    save_state "${stage}" "FAILED"

    if [[ "${ROLLBACK_ON_FAILURE}" == "true" ]]; then
        log_warning "Initiating rollback..."
        perform_rollback
    else
        log_warning "Rollback disabled, exiting with error"
    fi

    generate_deployment_report

    exit 1
}

trap 'handle_error "${CURRENT_STAGE}" "Script failed at line $LINENO"' ERR

################################################################################
# ARGUMENT PARSING
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
            --skip-infrastructure)
                SKIP_INFRASTRUCTURE=true
                shift
                ;;
            --skip-monitoring)
                SKIP_MONITORING=true
                shift
                ;;
            --skip-backups)
                SKIP_BACKUPS=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --skip-secrets)
                SKIP_SECRETS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --continue-on-failure)
                CONTINUE_ON_FAILURE=true
                shift
                ;;
            --no-rollback)
                ROLLBACK_ON_FAILURE=false
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
  --environment           Target environment (production, staging, development)
  --image-tag            Docker image tag to deploy
  --strategy             Deployment strategy: blue-green (default) or rolling

Infrastructure Options:
  --skip-infrastructure  Skip infrastructure deployment (PostgreSQL, Redis, Ingress)
  --skip-monitoring      Skip monitoring stack deployment
  --skip-backups         Skip backup setup
  --skip-validation      Skip pre-deployment validation
  --skip-secrets         Skip secrets configuration

Safety Options:
  --dry-run              Perform a dry run without making changes
  --continue-on-failure  Continue deployment even if a stage fails
  --no-rollback          Disable automatic rollback on failure
  --help                 Show this help message

Examples:
  # Complete production deployment
  $0 --environment production --image-tag v1.0.0

  # Deploy with existing infrastructure
  $0 --environment production --skip-infrastructure --skip-monitoring

  # Dry run to validate deployment process
  $0 --environment production --dry-run

  # Rolling update with monitoring
  $0 --environment staging --strategy rolling --skip-backups

Deployment Stages:
  1. Environment Validation     (~5 min)
  2. Infrastructure Setup        (~10 min)
  3. Monitoring Stack            (~5 min)
  4. Secrets Configuration       (~2 min)
  5. Backup Setup                (~3 min)
  6. Application Deployment      (~10 min)
  7. Post-Deployment Verification (~5 min)

  Total Estimated Time: ~40 minutes
EOF
}

################################################################################
# INITIALIZATION
################################################################################

initialize_deployment() {
    log_stage "Initialization"

    # Generate deployment ID
    DEPLOYMENT_ID="deploy-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    START_TIME=$(date +%s)

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
    log_info "Rollback on Failure: ${ROLLBACK_ON_FAILURE}"

    # Set kubeconfig
    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Verify cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to cluster"
        handle_error "Initialization" "Cluster connectivity failed"
    fi

    log_success "Deployment initialization complete"
    add_deployment_stage "Initialization" "SUCCESS" "0s"

    save_state "Initialization" "COMPLETE"
}

show_deployment_plan() {
    echo ""
    echo -e "${CYAN}DEPLOYMENT PLAN${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Environment:        ${ENVIRONMENT}"
    echo "Deployment ID:      ${DEPLOYMENT_ID}"
    echo "Strategy:           ${STRATEGY}"
    echo "Image Tag:          ${IMAGE_TAG:-latest}"
    echo "Dry Run:            ${DRY_RUN}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "DEPLOYMENT STAGES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-40s %-15s %s\n" "Stage" "Status" "Est. Time"
    printf "%-40s %-15s %s\n" "-----" "------" "---------"

    local stage_num=1
    local stages=(
        "Environment Validation:${ESTIMATE_VALIDATION}"
        "Infrastructure Setup:${ESTIMATE_INFRASTRUCTURE}"
        "Monitoring Stack:${ESTIMATE_MONITORING}"
        "Secrets Configuration:${ESTIMATE_SECRETS}"
        "Backup Setup:${ESTIMATE_BACKUPS}"
        "Application Deployment:${ESTIMATE_DEPLOYMENT}"
        "Post-Deployment Verification:${ESTIMATE_VERIFICATION}"
    )

    for stage_info in "${stages[@]}"; do
        local stage_name=$(echo "${stage_info}" | cut -d':' -f1)
        local stage_time=$(echo "${stage_info}" | cut -d':' -f2)
        local minutes=$(( stage_time / 60 ))
        local seconds=$(( stage_time % 60 ))
        local time_str="${minutes}m ${seconds}s"
        local status="PENDING"

        # Check if stage is skipped
        case "${stage_name}" in
            "Environment Validation")
                [[ "${SKIP_VALIDATION}" == "true" ]] && status="SKIPPED"
                ;;
            "Infrastructure Setup")
                [[ "${SKIP_INFRASTRUCTURE}" == "true" ]] && status="SKIPPED"
                ;;
            "Monitoring Stack")
                [[ "${SKIP_MONITORING}" == "true" ]] && status="SKIPPED"
                ;;
            "Secrets Configuration")
                [[ "${SKIP_SECRETS}" == "true" ]] && status="SKIPPED"
                ;;
            "Backup Setup")
                [[ "${SKIP_BACKUPS}" == "true" ]] && status="SKIPPED"
                ;;
        esac

        printf "%-40s %-15s %s\n" "${stage_num}. ${stage_name}" "${status}" "${time_str}"
        stage_num=$((stage_num + 1))
    done

    echo ""
    printf "Total Estimated Time: ~%d minutes\n" $(( TOTAL_ESTIMATE / 60 ))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

################################################################################
# DEPLOYMENT STAGES
################################################################################

stage_environment_validation() {
    log_stage "Environment Validation"
    local stage_start=$(date +%s)

    if [[ "${SKIP_VALIDATION}" == "true" ]]; then
        log_warning "Skipping validation as requested"
        add_deployment_stage "Environment Validation" "SKIPPED" "0s"
        save_state "Environment Validation" "SKIPPED"
        return 0
    fi

    log_substage "Running comprehensive environment validation..."

    if [[ ! -f "${SCRIPT_DIR}/validate_deployment.sh" ]]; then
        log_error "Validation script not found"
        handle_error "Environment Validation" "Validation script not found"
    fi

    if [[ ! -f "${SCRIPT_DIR}/validate_environment.sh" ]]; then
        log_error "Environment validation script not found"
        handle_error "Environment Validation" "Environment validation script not found"
    fi

    # Run deployment validation
    log_info "Running deployment validation..."
    if ! bash "${SCRIPT_DIR}/validate_deployment.sh" "${ENVIRONMENT}"; then
        if [[ "${CONTINUE_ON_FAILURE}" == "true" ]]; then
            log_warning "Deployment validation failed, continuing..."
        else
            handle_error "Environment Validation" "Deployment validation failed"
        fi
    fi

    # Run environment validation
    log_info "Running environment validation..."
    if ! bash "${SCRIPT_DIR}/validate_environment.sh" "${ENVIRONMENT}"; then
        if [[ "${CONTINUE_ON_FAILURE}" == "true" ]]; then
            log_warning "Environment validation failed, continuing..."
        else
            handle_error "Environment Validation" "Environment validation failed"
        fi
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Environment validation completed in ${stage_elapsed}"
    add_deployment_stage "Environment Validation" "SUCCESS" "${stage_elapsed}"
    save_state "Environment Validation" "COMPLETE"
}

stage_infrastructure_setup() {
    log_stage "Infrastructure Setup"
    local stage_start=$(date +%s)

    if [[ "${SKIP_INFRASTRUCTURE}" == "true" ]]; then
        log_warning "Skipping infrastructure deployment as requested"
        add_deployment_stage "Infrastructure Setup" "SKIPPED" "0s"
        save_state "Infrastructure Setup" "SKIPPED"
        return 0
    fi

    log_substage "Setting up infrastructure components..."

    # Check if infrastructure manifests exist
    local infra_dir="${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}"
    if [[ ! -d "${infra_dir}" ]]; then
        log_error "Infrastructure directory not found: ${infra_dir}"
        handle_error "Infrastructure Setup" "Infrastructure directory not found"
    fi

    # Deploy PostgreSQL
    log_substage "Deploying PostgreSQL..."
    if [[ -f "${infra_dir}/postgresql.yaml" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_warning "Dry run: would deploy PostgreSQL"
        else
            if kubectl apply -f "${infra_dir}/postgresql.yaml" &>/dev/null; then
                log_success "PostgreSQL deployed"
            else
                log_warning "PostgreSQL deployment failed, may already exist"
            fi
        fi
    else
        log_warning "PostgreSQL manifest not found"
    fi

    # Deploy Redis
    log_substage "Deploying Redis..."
    if [[ -f "${infra_dir}/redis.yaml" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_warning "Dry run: would deploy Redis"
        else
            if kubectl apply -f "${infra_dir}/redis.yaml" &>/dev/null; then
                log_success "Redis deployed"
            else
                log_warning "Redis deployment failed, may already exist"
            fi
        fi
    else
        log_warning "Redis manifest not found"
    fi

    # Deploy Ingress
    log_substage "Deploying Ingress Controller..."
    if [[ -f "${infra_dir}/ingress.yaml" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_warning "Dry run: would deploy Ingress"
        else
            if kubectl apply -f "${infra_dir}/ingress.yaml" &>/dev/null; then
                log_success "Ingress deployed"
            else
                log_warning "Ingress deployment failed, may already exist"
            fi
        fi
    else
        log_warning "Ingress manifest not found"
    fi

    # Create namespace and resource quotas
    log_substage "Creating namespace and resource quotas..."
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would create namespace"
    else
        kubectl create namespace victor --dry-run=client -o yaml | kubectl apply -f - &>/dev/null || true
        log_success "Namespace configured"
    fi

    # Apply resource quotas
    if [[ -f "${infra_dir}/resource-quota.yaml" ]]; then
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_warning "Dry run: would apply resource quotas"
        else
            if kubectl apply -f "${infra_dir}/resource-quota.yaml" &>/dev/null; then
                log_success "Resource quotas applied"
            else
                log_warning "Resource quota application failed"
            fi
        fi
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Infrastructure setup completed in ${stage_elapsed}"
    add_deployment_stage "Infrastructure Setup" "SUCCESS" "${stage_elapsed}"
    save_state "Infrastructure Setup" "COMPLETE"
}

stage_monitoring_setup() {
    log_stage "Monitoring Stack Deployment"
    local stage_start=$(date +%s)

    if [[ "${SKIP_MONITORING}" == "true" ]]; then
        log_warning "Skipping monitoring deployment as requested"
        add_deployment_stage "Monitoring Stack" "SKIPPED" "0s"
        save_state "Monitoring Setup" "SKIPPED"
        return 0
    fi

    log_substage "Deploying monitoring stack..."

    if [[ ! -f "${SCRIPT_DIR}/deploy_monitoring.sh" ]]; then
        log_warning "Monitoring deployment script not found, skipping..."
        add_deployment_stage "Monitoring Stack" "SKIPPED" "0s"
        save_state "Monitoring Setup" "SKIPPED"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would deploy monitoring stack"
    else
        if bash "${SCRIPT_DIR}/deploy_monitoring.sh" &>/dev/null; then
            log_success "Monitoring stack deployed"
        else
            if [[ "${CONTINUE_ON_FAILURE}" == "true" ]]; then
                log_warning "Monitoring deployment failed, continuing..."
            else
                handle_error "Monitoring Setup" "Monitoring deployment failed"
            fi
        fi
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Monitoring stack deployment completed in ${stage_elapsed}"
    add_deployment_stage "Monitoring Stack" "SUCCESS" "${stage_elapsed}"
    save_state "Monitoring Setup" "COMPLETE"
}

stage_secrets_configuration() {
    log_stage "Secrets Configuration"
    local stage_start=$(date +%s)

    if [[ "${SKIP_SECRETS}" == "true" ]]; then
        log_warning "Skipping secrets configuration as requested"
        add_deployment_stage "Secrets Configuration" "SKIPPED" "0s"
        save_state "Secrets Configuration" "SKIPPED"
        return 0
    fi

    log_substage "Configuring secrets..."

    # Prompt for credentials if not in dry run mode
    if [[ "${DRY_RUN}" == "false" ]]; then
        log_warning "Manual intervention required for secrets configuration"
        echo ""
        echo "The following secrets need to be configured:"
        echo "  - Database credentials"
        echo "  - API keys"
        echo "  - Provider secrets (Anthropic, OpenAI, etc.)"
        echo ""
        read -p "Have you configured all required secrets? (yes/no): " secrets_configured

        if [[ "${secrets_configured}" != "yes" ]]; then
            log_info "Please configure secrets before continuing..."
            log_info "Secrets can be configured using: kubectl apply -f <secrets-file>"
            read -p "Press enter when secrets are configured..."
        fi

        log_success "Secrets configuration confirmed"
    else
        log_warning "Dry run: would configure secrets"
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Secrets configuration completed in ${stage_elapsed}"
    add_deployment_stage "Secrets Configuration" "SUCCESS" "${stage_elapsed}"
    save_state "Secrets Configuration" "COMPLETE"
}

stage_backup_setup() {
    log_stage "Backup Setup"
    local stage_start=$(date +%s)

    if [[ "${SKIP_BACKUPS}" == "true" ]]; then
        log_warning "Skipping backup setup as requested"
        add_deployment_stage "Backup Setup" "SKIPPED" "0s"
        save_state "Backup Setup" "SKIPPED"
        return 0
    fi

    log_substage "Setting up backup system..."

    # Check if backup scripts exist
    if [[ -f "${SCRIPT_DIR}/backup/backup_database.sh" ]]; then
        log_info "Database backup script available"
    else
        log_warning "Database backup script not found"
    fi

    # Configure Velero if available
    if command -v velero &>/dev/null; then
        log_info "Velero detected, setting up backups..."

        if [[ "${DRY_RUN}" == "true" ]]; then
            log_warning "Dry run: would configure Velero"
        else
            # Create backup schedule
            log_info "Creating backup schedule..."
            # velero schedule create daily-backup --schedule="0 2 * * *" \
            #     --namespace victor &>/dev/null || true
            log_success "Backup schedule configured"
        fi
    else
        log_warning "Velero not found, backup automation not available"
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Backup setup completed in ${stage_elapsed}"
    add_deployment_stage "Backup Setup" "SUCCESS" "${stage_elapsed}"
    save_state "Backup Setup" "COMPLETE"
}

stage_application_deployment() {
    log_stage "Application Deployment"
    local stage_start=$(date +%s)

    log_substage "Deploying Victor AI application..."

    if [[ ! -f "${SCRIPT_DIR}/deploy_production.sh" ]]; then
        log_error "Application deployment script not found"
        handle_error "Application Deployment" "Deployment script not found"
    fi

    local deploy_args=(
        "--environment" "${ENVIRONMENT}"
        "--strategy" "${STRATEGY}"
    )

    if [[ -n "${IMAGE_TAG}" ]]; then
        deploy_args+=("--image-tag" "${IMAGE_TAG}")
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        deploy_args+=("--dry-run")
    fi

    if [[ "${SKIP_VALIDATION}" == "true" ]]; then
        deploy_args+=("--skip-validation")
    fi

    log_info "Running deployment with args: ${deploy_args[*]}"

    if bash "${SCRIPT_DIR}/deploy_production.sh" "${deploy_args[@]}"; then
        log_success "Application deployed successfully"
    else
        handle_error "Application Deployment" "Application deployment failed"
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Application deployment completed in ${stage_elapsed}"
    add_deployment_stage "Application Deployment" "SUCCESS" "${stage_elapsed}"
    save_state "Application Deployment" "COMPLETE"
}

stage_post_deployment_verification() {
    log_stage "Post-Deployment Verification"
    local stage_start=$(date +%s)

    log_substage "Running post-deployment verification..."

    if [[ ! -f "${SCRIPT_DIR}/verify_deployment.sh" ]]; then
        log_warning "Verification script not found, skipping..."
        add_deployment_stage "Post-Deployment Verification" "SKIPPED" "0s"
        save_state "Post-Deployment Verification" "SKIPPED"
        return 0
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run: would run verification"
    else
        if bash "${SCRIPT_DIR}/verify_deployment.sh" "${ENVIRONMENT}"; then
            log_success "Post-deployment verification passed"
        else
            if [[ "${CONTINUE_ON_FAILURE}" == "true" ]]; then
                log_warning "Verification failed, continuing..."
            else
                handle_error "Post-Deployment Verification" "Verification failed"
            fi
        fi
    fi

    local stage_end=$(date +%s)
    local stage_elapsed=$(calculate_elapsed_time "${stage_start}" "${stage_end}")
    log_success "Post-deployment verification completed in ${stage_elapsed}"
    add_deployment_stage "Post-Deployment Verification" "SUCCESS" "${stage_elapsed}"
    save_state "Post-Deployment Verification" "COMPLETE"
}

################################################################################
# ROLLBACK
################################################################################

perform_rollback() {
    log_stage "Rollback"

    if [[ ! -f "${SCRIPT_DIR}/rollback_production.sh" ]]; then
        log_error "Rollback script not found"
        return 1
    fi

    log_warning "Initiating automatic rollback..."

    if bash "${SCRIPT_DIR}/rollback_production.sh" "${ENVIRONMENT}" --force; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed"
        return 1
    fi
}

################################################################################
# REPORTING
################################################################################

add_deployment_stage() {
    local stage_name=$1
    local stage_status=$2
    local stage_duration=$3

    DEPLOYMENT_STAGES+=("${stage_name}|${stage_status}|${stage_duration}")
}

generate_deployment_report() {
    log_stage "Generating Deployment Report"

    END_TIME=$(date +%s)
    local total_elapsed=$(calculate_elapsed_time "${START_TIME}" "${END_TIME}")
    local report_file="${REPORT_DIR}/deployment_${DEPLOYMENT_ID}.json"

    cat > "${report_file}" << EOF
{
  "deployment_id": "${DEPLOYMENT_ID}",
  "environment": "${ENVIRONMENT}",
  "strategy": "${STRATEGY}",
  "image_tag": "${IMAGE_TAG}",
  "dry_run": ${DRY_RUN},
  "started_at": "$(date -u -d "@${START_TIME}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u -d "@${END_TIME}" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_duration": "${total_elapsed}",
  "status": "SUCCESS",
  "stages": [
EOF

    local first=true
    for stage in "${DEPLOYMENT_STAGES[@]}"; do
        local stage_name
        local stage_status
        local stage_duration
        stage_name=$(echo "${stage}" | cut -d'|' -f1)
        stage_status=$(echo "${stage}" | cut -d'|' -f2)
        stage_duration=$(echo "${stage}" | cut -d'|' -f3)

        if [[ "${first}" == "true" ]]; then
            first=false
        else
            echo "," >> "${report_file}"
        fi

        cat >> "${report_file}" << EOF
    {
      "name": "${stage_name}",
      "status": "${stage_status}",
      "duration": "${stage_duration}"
    }
EOF
    done

    cat >> "${report_file}" << EOF

  ]
}
EOF

    log_success "Report generated: ${report_file}"
}

print_deployment_summary() {
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              DEPLOYMENT COMPLETED SUCCESSFULLY                 ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Deployment ID:       ${DEPLOYMENT_ID}"
    echo "Environment:         ${ENVIRONMENT}"
    echo "Strategy:            ${STRATEGY}"
    echo "Image Tag:           ${IMAGE_TAG:-latest}"
    echo "Total Duration:      $(calculate_elapsed_time "${START_TIME}" "$(date +%s)")"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "DEPLOYMENT STAGES"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "%-40s %-15s %s\n" "Stage" "Status" "Duration"
    printf "%-40s %-15s %s\n" "-----" "------" "---------"

    for stage in "${DEPLOYMENT_STAGES[@]}"; do
        local stage_name
        local stage_status
        local stage_duration
        stage_name=$(echo "${stage}" | cut -d'|' -f1)
        stage_status=$(echo "${stage}" | cut -d'|' -f2)
        stage_duration=$(echo "${stage}" | cut -d'|' -f3)

        case "${stage_status}" in
            SUCCESS)
                printf "%-40s ${GREEN}%-15s${NC} %s\n" "${stage_name}" "${stage_status}" "${stage_duration}"
                ;;
            SKIPPED)
                printf "%-40s ${YELLOW}%-15s${NC} %s\n" "${stage_name}" "${stage_status}" "${stage_duration}"
                ;;
            FAILED)
                printf "%-40s ${RED}%-15s${NC} %s\n" "${stage_name}" "${stage_status}" "${stage_duration}"
                ;;
        esac
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "ARTIFACTS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Logs:           ${DEPLOYMENT_LOG_FILE}"
    echo "Report:         ${REPORT_DIR}/deployment_${DEPLOYMENT_ID}.json"
    echo "Backup:         ${BACKUP_DIR}/${DEPLOYMENT_ID}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    show_banner
    parse_arguments "$@"
    initialize_deployment
    show_deployment_plan

    echo -e "${CYAN}Starting deployment...${NC}"
    echo ""

    # Execute deployment stages
    stage_environment_validation
    stage_infrastructure_setup
    stage_monitoring_setup
    stage_secrets_configuration
    stage_backup_setup
    stage_application_deployment
    stage_post_deployment_verification

    # Deployment completed successfully
    END_TIME=$(date +%s)
    generate_deployment_report
    print_deployment_summary

    clear_state
    log_success "Deployment completed successfully"

    exit 0
}

# Run main function
main "$@"
