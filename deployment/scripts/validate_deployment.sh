#!/bin/bash
################################################################################
# Production Deployment Pre-Validation Script
#
# This script performs comprehensive pre-deployment validation to ensure
# production readiness before deploying to production environments.
#
# Usage:
#   ./validate_deployment.sh [environment]
#
# Arguments:
#   environment - Target environment (production, staging, development)
#
# Features:
# - Cluster connectivity validation
# - Configuration validation
# - Security scans
# - Monitoring readiness checks
# - Backup verification
# - Resource availability checks
# - Dependency validation
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

# Validation results
declare -a VALIDATION_RESULTS
declare -a VALIDATION_ERRORS
VALIDATION_SCORE=0
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
    VALIDATION_RESULTS+=("$1|$2")
    if [[ "$2" == "PASS" ]]; then
        VALIDATION_SCORE=$((VALIDATION_SCORE + 10))
    fi
}

add_error() {
    VALIDATION_ERRORS+=("$1")
}

################################################################################
# Configuration
################################################################################

ENVIRONMENT="${1:-production}"
VALIDATION_LOG="${LOG_DIR}/pre_validation_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).log"
VALIDATION_REPORT="${REPORT_DIR}/pre_validation_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).json"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${REPORT_DIR}"

# Initialize log
echo "Production Pre-Deployment Validation - ${ENVIRONMENT}" | tee "${VALIDATION_LOG}"
echo "Started at: $(date)" | tee -a "${VALIDATION_LOG}"
echo "=======================================================" | tee -a "${VALIDATION_LOG}"

################################################################################
# Validation Functions
################################################################################

validate_cluster_connectivity() {
    log_section "Validating Cluster Connectivity"

    local kubeconfig="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"
    local context="victor-${ENVIRONMENT}"

    # Check if kubeconfig exists
    if [[ ! -f "${kubeconfig}" ]]; then
        log_error "Kubeconfig not found: ${kubeconfig}"
        add_result "Cluster Connectivity" "FAIL"
        add_error "Kubeconfig not found"
        return 1
    fi

    export KUBECONFIG="${kubeconfig}"

    # Check if context exists
    if ! kubectl config get-contexts "${context}" &>/dev/null; then
        log_error "Kubernetes context not found: ${context}"
        add_result "Cluster Connectivity" "FAIL"
        add_error "Kubernetes context not found"
        return 1
    fi

    # Try to connect to cluster
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to cluster"
        add_result "Cluster Connectivity" "FAIL"
        add_error "Cluster connection failed"
        return 1
    fi

    # Check cluster health
    local cluster_health
    cluster_health=$(kubectl get cs -o json 2>/dev/null | jq -r '.items[].conditions[] | select(.type == "Healthy") | .status' | grep -c "True" || echo "0")

    if [[ "${cluster_health}" -lt 1 ]]; then
        log_warning "Cluster health check returned unhealthy"
        add_result "Cluster Connectivity" "WARN"
    else
        log_success "Cluster is healthy and reachable"
        add_result "Cluster Connectivity" "PASS"
    fi

    # Get cluster info for report
    local cluster_version
    cluster_version=$(kubectl version -o json | jq -r '.serverVersion.gitVersion')
    log_info "Cluster version: ${cluster_version}"

    return 0
}

validate_configuration() {
    log_section "Validating Configuration"

    local config_dir="${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}"

    # Check if environment overlay exists
    if [[ ! -d "${config_dir}" ]]; then
        log_error "Environment overlay not found: ${config_dir}"
        add_result "Configuration Validation" "FAIL"
        add_error "Environment overlay not found"
        return 1
    fi

    # Check kustomization file
    if [[ ! -f "${config_dir}/kustomization.yaml" ]]; then
        log_error "Kustomization file not found"
        add_result "Configuration Validation" "FAIL"
        add_error "Kustomization file not found"
        return 1
    fi

    # Validate kustomization
    if ! kubectl kustomize "${config_dir}" &>/dev/null; then
        log_error "Kustomization validation failed"
        add_result "Configuration Validation" "FAIL"
        add_error "Kustomization validation failed"
        return 1
    fi

    # Check required secrets
    local required_secrets=(
        "victor-database-secret"
        "victor-api-secret"
        "victor-provider-secrets"
    )

    for secret in "${required_secrets[@]}"; do
        if ! kubectl get secret "${secret}" -n victor &>/dev/null; then
            log_warning "Secret not found: ${secret}"
            add_result "Configuration Validation" "WARN"
            add_error "Missing secret: ${secret}"
        fi
    done

    log_success "Configuration validation passed"
    add_result "Configuration Validation" "PASS"

    return 0
}

validate_security() {
    log_section "Performing Security Validation"

    # Check for vulnerabilities in images
    local images
    images=$(kubectl kustomize "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" | grep "image:" | awk '{print $2}' | sort -u)

    local vuln_found=0
    for image in ${images}; do
        log_info "Scanning image: ${image}"

        # Skip local images
        if [[ "${image}" =~ localhost ]] || [[ "${image}" =~ 127.0.0.1 ]]; then
            continue
        fi

        # Check if trivy is available
        if command -v trivy &>/dev/null; then
            local vuln_output
            vuln_output=$(trivy image --severity HIGH,CRITICAL --format json "${image}" 2>/dev/null || echo "{}")

            local vuln_count
            vuln_count=$(echo "${vuln_output}" | jq -r '.Results[]?.Vulnerabilities | length' 2>/dev/null || echo "0")

            if [[ "${vuln_count}" -gt 0 ]]; then
                log_warning "Found ${vuln_count} vulnerabilities in ${image}"
                add_result "Security Scan" "WARN"
                vuln_found=1
            fi
        fi
    done

    if [[ ${vuln_found} -eq 0 ]]; then
        log_success "No critical vulnerabilities found"
        add_result "Security Scan" "PASS"
    fi

    # Check RBAC configuration
    log_info "Validating RBAC configuration"

    local rbac_issues
    rbac_issues=$(kubectl kustomize "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" | \
        grep -i "rules:" -A 5 | grep -c '\*:\*\*' || echo "0")

    if [[ "${rbac_issues}" -gt 0 ]]; then
        log_warning "Found overly permissive RBAC rules"
        add_result "Security Scan" "WARN"
        add_error "Overly permissive RBAC rules detected"
    fi

    # Check for secrets in config
    log_info "Checking for secrets in configuration"

    if grep -r -i "password\|secret\|key" "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" | grep -v "secretName" | grep -v "BEGIN\|END" &>/dev/null; then
        log_warning "Potential hardcoded secrets found in configuration"
        add_result "Security Scan" "WARN"
        add_error "Hardcoded secrets detected"
    fi

    return 0
}

validate_monitoring() {
    log_section "Validating Monitoring Setup"

    # Check Prometheus
    if ! kubectl get statefulset prometheus -n monitoring &>/dev/null; then
        log_warning "Prometheus not found in monitoring namespace"
        add_result "Monitoring Setup" "WARN"
        add_error "Prometheus not deployed"
    else
        log_success "Prometheus is deployed"
        add_result "Monitoring Setup" "PASS"
    fi

    # Check Grafana
    if ! kubectl get deployment grafana -n monitoring &>/dev/null; then
        log_warning "Grafana not found in monitoring namespace"
        add_result "Monitoring Setup" "WARN"
        add_error "Grafana not deployed"
    else
        log_success "Grafana is deployed"
        add_result "Monitoring Setup" "PASS"
    fi

    # Check if monitoring dashboards exist
    local dashboard_count
    dashboard_count=$(kubectl get configmap -n monitoring -l app=victor-dashboard 2>/dev/null | wc -l)

    if [[ "${dashboard_count}" -lt 1 ]]; then
        log_warning "No monitoring dashboards found"
        add_result "Monitoring Setup" "WARN"
        add_error "Monitoring dashboards not configured"
    else
        log_success "Found ${dashboard_count} monitoring dashboards"
    fi

    # Check alerting rules
    if ! kubectl get prometheusrule -n victor &>/dev/null; then
        log_warning "No Prometheus alerting rules found"
        add_result "Monitoring Setup" "WARN"
        add_error "Alerting rules not configured"
    else
        log_success "Alerting rules are configured"
    fi

    return 0
}

validate_resources() {
    log_section "Validating Resource Availability"

    # Check node capacity
    local available_cpu
    local available_memory
    available_cpu=$(kubectl top nodes 2>/dev/null | awk 'NR>1 {sum+=$2} END {print sum}' || echo "0")
    available_memory=$(kubectl top nodes 2>/dev/null | awk 'NR>1 {sum+=$4} END {print sum}' || echo "0")

    log_info "Available cluster CPU: ${available_cpu}m"
    log_info "Available cluster memory: ${available_memory}Mi"

    # Check resource requests
    local requested_cpu
    local requested_memory
    requested_cpu=$(kubectl kustomize "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" | \
        grep -A 3 "resources:" | grep "cpu:" | awk '{sum+=$2} END {print sum}' || echo "0")
    requested_memory=$(kubectl kustomize "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" | \
        grep -A 3 "resources:" | grep "memory:" | awk '{sum+=$2} END {print sum}' || echo "0")

    log_info "Requested CPU: ${requested_cpu}"
    log_info "Requested Memory: ${requested_memory}"

    # Check storage class
    if ! kubectl get storageclass standard &>/dev/null; then
        log_warning "No standard storage class found"
        add_result "Resource Availability" "WARN"
        add_error "Storage class not available"
    else
        log_success "Storage class is available"
        add_result "Resource Availability" "PASS"
    fi

    # Check PVCs
    local pvc_count
    pvc_count=$(kubectl kustomize "${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}" | \
        grep -c "kind: PersistentVolumeClaim" || echo "0")

    if [[ "${pvc_count}" -gt 0 ]]; then
        log_info "Deployment requires ${pvc_count} PVCs"

        # Check if PVs are available
        local available_pvs
        available_pvs=$(kubectl get pv | grep -c "Available" || echo "0")

        if [[ "${available_pvs}" -lt "${pvc_count}" ]]; then
            log_warning "Not enough available PVs (${available_pvs}/${pvc_count})"
            add_result "Resource Availability" "WARN"
            add_error "Insufficient PVs"
        fi
    fi

    return 0
}

validate_backups() {
    log_section "Validating Backup Status"

    # Check if Velero is installed
    if ! command -v velero &>/dev/null; then
        log_warning "Velero CLI not found, skipping backup validation"
        add_result "Backup Status" "WARN"
        return 0
    fi

    # Check latest backup
    local latest_backup
    latest_backup=$(velero backup get -o json 2>/dev/null | jq -r '.items[] | select(.status.phase == "Completed") | .metadata.name' | sort -r | head -n1)

    if [[ -z "${latest_backup}" ]]; then
        log_warning "No completed backups found"
        add_result "Backup Status" "WARN"
        add_error "No backups available"
    else
        log_success "Latest backup: ${latest_backup}"

        # Check backup age
        local backup_age
        backup_age=$(velero backup get "${latest_backup}" -o json 2>/dev/null | jq -r '.status.startTimestamp')
        local backup_date
        backup_date=$(date -d "${backup_age}" +%s 2>/dev/null || echo "0")
        local current_date
        current_date=$(date +%s)
        local age_hours
        age_hours=$(( (current_date - backup_date) / 3600 ))

        if [[ ${age_hours} -gt 24 ]]; then
            log_warning "Latest backup is ${age_hours} hours old"
            add_result "Backup Status" "WARN"
        else
            log_success "Backup is recent (${age_hours} hours old)"
            add_result "Backup Status" "PASS"
        fi
    fi

    # Check schedule
    local backup_schedule
    backup_schedule=$(velero schedule get -o json 2>/dev/null | jq -r '.items[] | .metadata.name' | wc -l)

    if [[ "${backup_schedule}" -eq 0 ]]; then
        log_warning "No backup schedules configured"
        add_result "Backup Status" "WARN"
    else
        log_success "Found ${backup_schedule} backup schedules"
    fi

    return 0
}

validate_dependencies() {
    log_section "Validating Dependencies"

    # Check PostgreSQL
    if ! kubectl get statefulset -n victor postgresql &>/dev/null; then
        log_warning "PostgreSQL not found"
        add_result "Dependencies" "WARN"
        add_error "PostgreSQL not deployed"
    else
        log_success "PostgreSQL is deployed"
        add_result "Dependencies" "PASS"
    fi

    # Check Redis
    if ! kubectl get statefulset -n victor redis &>/dev/null; then
        log_warning "Redis not found"
        add_result "Dependencies" "WARN"
        add_error "Redis not deployed"
    else
        log_success "Redis is deployed"
        add_result "Dependencies" "PASS"
    fi

    # Check ingress controller
    if ! kubectl get deployment -n ingress-nginx &>/dev/null; then
        log_warning "Ingress controller not found"
        add_result "Dependencies" "WARN"
        add_error "Ingress controller not deployed"
    else
        log_success "Ingress controller is deployed"
        add_result "Dependencies" "PASS"
    fi

    # Check cert-manager
    if ! kubectl get deployment -n cert-manager cert-manager &>/dev/null; then
        log_warning "cert-manager not found"
        add_result "Dependencies" "WARN"
        add_error "cert-manager not deployed"
    else
        log_success "cert-manager is deployed"
        add_result "Dependencies" "PASS"
    fi

    return 0
}

generate_report() {
    log_section "Generating Validation Report"

    local report_json="${VALIDATION_REPORT}"

    # Build JSON report
    cat > "${report_json}" << EOF
{
  "environment": "${ENVIRONMENT}",
  "validation_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "validation_score": ${VALIDATION_SCORE},
  "max_score": ${MAX_SCORE},
  "readiness_percentage": $(( VALIDATION_SCORE * 100 / MAX_SCORE )),
  "validations": [
EOF

    local first=true
    for result in "${VALIDATION_RESULTS[@]}"; do
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
    for error in "${VALIDATION_ERRORS[@]}"; do
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

  ],
  "recommendations": []
}
EOF

    log_success "Report generated: ${report_json}"

    # Print summary
    echo ""
    echo "==================================================================="
    echo "VALIDATION SUMMARY"
    echo "==================================================================="
    echo "Environment:     ${ENVIRONMENT}"
    echo "Validation Score: ${VALIDATION_SCORE}/${MAX_SCORE}"
    echo "Readiness:       $(( VALIDATION_SCORE * 100 / MAX_SCORE ))%"
    echo ""
    echo "Validation Results:"
    printf "%-30s %-10s\n" "Check" "Status"
    printf "%-30s %-10s\n" "-----" "------"

    for result in "${VALIDATION_RESULTS[@]}"; do
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

    if [[ ${#VALIDATION_ERRORS[@]} -gt 0 ]]; then
        echo ""
        echo "Errors:"
        for error in "${VALIDATION_ERRORS[@]}"; do
            echo "  - ${error}"
        done
    fi

    echo ""
    echo "Report saved to: ${report_json}"

    # Check if ready for deployment
    if [[ ${VALIDATION_SCORE} -ge 80 ]]; then
        log_success "Environment is ready for deployment (Score: ${VALIDATION_SCORE}/${MAX_SCORE})"
        return 0
    else
        log_error "Environment is not ready for deployment (Score: ${VALIDATION_SCORE}/${MAX_SCORE})"
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    log_info "Starting pre-deployment validation for ${ENVIRONMENT}"

    # Run validations
    validate_cluster_connectivity
    validate_configuration
    validate_security
    validate_monitoring
    validate_resources
    validate_backups
    validate_dependencies

    # Generate report
    generate_report

    local exit_code=$?

    echo ""
    echo "Validation completed at: $(date)" | tee -a "${VALIDATION_LOG}"

    exit ${exit_code}
}

# Run main function
main "$@"
