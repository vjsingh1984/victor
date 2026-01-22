#!/bin/bash
################################################################################
# Complete Deployment Workflow Script
#
# This script demonstrates the complete end-to-end deployment workflow for
# Victor AI, including infrastructure, database initialization, and application.
#
# This is an orchestration script that calls the individual deployment scripts
# in the correct order with proper error handling.
#
# Usage:
#   ./deploy_all.sh [--environment ENV] [--skip-infra] [--skip-db] [--skip-app]
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
ENVIRONMENT="production"
SKIP_INFRA=false
SKIP_DB=false
SKIP_APP=false
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  $1${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Complete deployment workflow for Victor AI production environment.

Options:
  --environment ENV          Target environment (default: production)
  --skip-infra               Skip infrastructure deployment
  --skip-db                  Skip database initialization
  --skip-app                 Skip application deployment
  --dry-run                  Perform dry run without making changes
  --help                     Show this help message

Examples:
  # Deploy everything
  $0

  # Deploy to staging
  $0 --environment staging

  # Skip infrastructure (already deployed)
  $0 --skip-infra

  # Dry run
  $0 --dry-run

Deployment Steps:
  1. Infrastructure (PostgreSQL, Redis, Ingress, cert-manager)
  2. Database Initialization (Schemas, Extensions, Tables)
  3. Application Deployment (Victor AI)
  4. Verification (Health checks, Smoke tests)

EOF
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
            --skip-infra)
                SKIP_INFRA=true
                shift
                ;;
            --skip-db)
                SKIP_DB=true
                shift
                ;;
            --skip-app)
                SKIP_APP=true
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

################################################################################
# Deployment Steps
################################################################################

deploy_infrastructure() {
    log_section "Step 1: Infrastructure Deployment"

    if [[ "${SKIP_INFRA}" == "true" ]]; then
        log_warning "Skipping infrastructure deployment"
        return 0
    fi

    local infra_args=""
    if [[ "${DRY_RUN}" == "true" ]]; then
        infra_args="--dry-run"
    fi

    log_info "Deploying infrastructure components..."

    if bash "${SCRIPT_DIR}/deploy_infrastructure.sh" ${infra_args}; then
        log_success "Infrastructure deployed successfully"
        return 0
    else
        log_error "Infrastructure deployment failed"
        return 1
    fi
}

wait_for_infrastructure() {
    log_section "Waiting for Infrastructure"

    if [[ "${SKIP_INFRA}" == "true" ]] || [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Skipping infrastructure wait"
        return 0
    fi

    log_info "Waiting for PostgreSQL to be ready..."
    local max_attempts=60
    local attempt=0

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if kubectl exec -n victor postgres-0 -- pg_isready -U victor -d victor &>/dev/null; then
            log_success "PostgreSQL is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    if [[ ${attempt} -eq ${max_attempts} ]]; then
        log_error "PostgreSQL not ready after ${max_attempts} attempts"
        return 1
    fi

    log_info "Waiting for Redis to be ready..."
    attempt=0

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if kubectl exec -n victor redis-0 -- redis-cli ping &>/dev/null; then
            log_success "Redis is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    if [[ ${attempt} -eq ${max_attempts} ]]; then
        log_error "Redis not ready after ${max_attempts} attempts"
        return 1
    fi

    return 0
}

initialize_database() {
    log_section "Step 2: Database Initialization"

    if [[ "${SKIP_DB}" == "true" ]]; then
        log_warning "Skipping database initialization"
        return 0
    fi

    if [[ "${SKIP_INFRA}" == "true" ]] && [[ "${DRY_RUN}" == "false" ]]; then
        log_info "Checking if database exists..."

        if ! kubectl get statefulset postgres -n victor &>/dev/null; then
            log_error "PostgreSQL StatefulSet not found"
            log_error "Deploy infrastructure first or use --skip-infra=false"
            return 1
        fi
    fi

    local db_args=""
    if [[ "${DRY_RUN}" == "true" ]]; then
        db_args="--dry-run"
    fi

    log_info "Initializing database..."

    if bash "${SCRIPT_DIR}/init_database.sh" ${db_args}; then
        log_success "Database initialized successfully"
        return 0
    else
        log_error "Database initialization failed"
        return 1
    fi
}

deploy_application() {
    log_section "Step 3: Application Deployment"

    if [[ "${SKIP_APP}" == "true" ]]; then
        log_warning "Skipping application deployment"
        return 0
    fi

    local app_args="--environment ${ENVIRONMENT}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        app_args="${app_args} --dry-run"
    fi

    log_info "Deploying Victor AI application..."

    if [[ -f "${SCRIPT_DIR}/deploy_production.sh" ]]; then
        if bash "${SCRIPT_DIR}/deploy_production.sh" ${app_args}; then
            log_success "Application deployed successfully"
            return 0
        else
            log_error "Application deployment failed"
            return 1
        fi
    else
        log_warning "deploy_production.sh not found, using kubectl directly..."

        local namespace="victor"
        if [[ "${ENVIRONMENT}" != "production" ]]; then
            namespace="${ENVIRONMENT}"
        fi

        if kubectl apply -f "${PROJECT_ROOT}/deployment/kubernetes/overlays/${ENVIRONMENT}"; then
            log_success "Application resources applied"
            return 0
        else
            log_error "Failed to apply application resources"
            return 1
        fi
    fi
}

verify_deployment() {
    log_section "Step 4: Deployment Verification"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Skipping verification (dry run)"
        return 0
    fi

    log_info "Running deployment verification..."

    if [[ -f "${SCRIPT_DIR}/verify_deployment.sh" ]]; then
        if bash "${SCRIPT_DIR}/verify_deployment.sh" "${ENVIRONMENT}"; then
            log_success "Deployment verified successfully"
            return 0
        else
            log_error "Deployment verification failed"
            return 1
        fi
    else
        log_warning "verify_deployment.sh not found, performing basic checks..."

        # Basic pod health check
        local pods_ready
        pods_ready=$(kubectl get pods -n victor -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -c "True" || echo "0")

        local pods_total
        pods_total=$(kubectl get pods -n victor -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | wc -w || echo "0")

        if [[ "${pods_ready}" -eq "${pods_total}" ]] && [[ "${pods_total}" -gt 0 ]]; then
            log_success "All pods are ready (${pods_ready}/${pods_total})"
            return 0
        else
            log_warning "Some pods are not ready (${pods_ready}/${pods_total})"
            return 1
        fi
    fi
}

print_deployment_info() {
    log_section "Deployment Information"

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warning "Dry run - no actual deployment performed"
        return 0
    fi

    echo ""
    echo "PostgreSQL Connection:"
    echo "  Host:     postgres.victor.svc.cluster.local"
    echo "  Port:     5432"
    echo "  Database: victor"
    echo "  User:     victor"
    echo ""

    echo "Redis Connection:"
    echo "  Host:     redis.victor.svc.cluster.local"
    echo "  Port:     6379"
    echo ""

    local lb_ip
    lb_ip=$(kubectl get service ingress-nginx-controller -n ingress-nginx \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

    echo "Ingress LoadBalancer:"
    echo "  IP:       ${lb_ip}"
    echo "  HTTP:     http://${lb_ip}"
    echo "  HTTPS:    https://${lb_ip}"
    echo ""

    echo "Useful Commands:"
    echo "  kubectl get pods -n victor"
    echo "  kubectl get pvc -n victor"
    echo "  kubectl logs -n victor -l app=victor --tail=100 -f"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_arguments "$@"

    echo "==================================================================="
    echo "Victor AI Complete Deployment Workflow"
    echo "==================================================================="
    echo "Environment: ${ENVIRONMENT}"
    echo "Dry Run:     ${DRY_RUN}"
    echo "==================================================================="
    echo ""

    local start_time=$(date +%s)

    # Execute deployment steps
    if ! deploy_infrastructure; then
        log_error "Deployment failed at infrastructure step"
        exit 1
    fi

    if ! wait_for_infrastructure; then
        log_error "Deployment failed at infrastructure wait step"
        exit 1
    fi

    if ! initialize_database; then
        log_error "Deployment failed at database initialization step"
        exit 1
    fi

    if ! deploy_application; then
        log_error "Deployment failed at application deployment step"
        exit 1
    fi

    if ! verify_deployment; then
        log_error "Deployment failed at verification step"
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    echo ""
    log_section "Deployment Complete"
    log_success "All components deployed successfully in ${minutes}m ${seconds}s"

    print_deployment_info

    exit 0
}

# Run main function
main "$@"
