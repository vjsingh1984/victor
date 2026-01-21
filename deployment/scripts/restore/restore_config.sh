#!/bin/bash
################################################################################
# Victor AI Configuration Restore Script
# Purpose: Restore Kubernetes resources, secrets, and Helm releases
# Usage: ./restore_config.sh [kubernetes|secrets|helm|all] <backup-timestamp>
# Example: ./restore_config.sh kubernetes 20260120_143000
# Dependencies: kubectl, helm, aws cli, jq
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
BACKUP_BUCKET="${BACKUP_BUCKET:-s3://victor-ai-backups}"
GIT_REPO="${CONFIG_GIT_REPO:-}"
LOG_FILE="/var/log/backups/restore_config.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESTORE_TYPE="${1:-all}"
BACKUP_TIMESTAMP="${2:-latest}"

# Safety checks
AUTO_CONFIRM="${AUTO_CONFIRM:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S'

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

log_important() {
    log "IMPORTANT" "${BLUE}$@${NC}"
}

send_alert() {
    local severity=$1
    local message=$2

    if [ -n "${ALERT_WEBHOOK}" ]; then
        local emoji=":warning:"
        [ "${severity}" = "CRITICAL" ] && emoji=":rotating_light:"
        [ "${severity}" = "INFO" ] && emoji=":white_check_mark:"

        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"${emoji} Config Restore ${severity}\", \"blocks\": [{\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": \"*${severity}*: ${message}\"}}]}" 2>/dev/null || true
    fi
}

error_exit() {
    log_error "$1"
    send_alert "CRITICAL" "$1"
    exit 1
}

confirm_action() {
    local message="$1"

    if [ "${AUTO_CONFIRM}" = "true" ]; then
        log_info "Auto-confirm enabled, proceeding"
        return 0
    fi

    log_important "${message}"
    echo -n "Type 'yes' to confirm: "
    read response

    if [ "${response}" != "yes" ]; then
        log_error "Restore cancelled by user"
        exit 1
    fi
}

# ============================================================================
# KUBERNETES RESOURCES RESTORE
# ============================================================================

restore_kubernetes_resources() {
    log_info "=== Starting Kubernetes Resources Restore ==="

    # Find backup to restore
    local backup_file=""
    if [ "${BACKUP_TIMESTAMP}" = "latest" ]; then
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/kubernetes/" | tail -1 | awk '{print $4}')
        log_info "Using latest backup: ${backup_file}"
    else
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/kubernetes/" | grep "${BACKUP_TIMESTAMP}" | tail -1 | awk '{print $4}')

        if [ -z "${backup_file}" ]; then
            error_exit "No backup found matching timestamp ${BACKUP_TIMESTAMP}"
        fi

        log_info "Using backup: ${backup_file}"
    fi

    confirm_action "Restore Kubernetes resources from ${backup_file}?"

    # Download backup
    local temp_dir="/tmp/kubernetes_restore_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    log_info "Downloading backup from S3"
    aws s3 cp "${BACKUP_BUCKET}/kubernetes/${backup_file}" "${temp_dir}/backup.tar.gz"

    # Extract
    log_info "Extracting backup"
    tar -xzf "${temp_dir}/backup.tar.gz" -C "${temp_dir}"

    # Find YAML files
    local yaml_files=$(find "${temp_dir}" -name "*.yaml" -type f)

    if [ -z "${yaml_files}" ]; then
        error_exit "No YAML files found in backup"
    fi

    # Validate YAML files
    log_info "Validating YAML files"

    for yaml_file in ${yaml_files}; do
        log_info "Validating ${yaml_file}"

        if ! kubectl apply --dry-run=server -f "${yaml_file}" &>/dev/null; then
            log_warn "Validation failed for ${yaml_file}, continuing anyway"
        fi
    done

    # Apply resources
    log_info "Applying Kubernetes resources"

    if [ "${DRY_RUN}" = "true" ]; then
        log_info "DRY RUN: Would apply resources from ${yaml_files}"
        for yaml_file in ${yaml_files}; do
            log_info "  Would apply: ${yaml_file}"
        done
    else
        # Apply ConfigMaps first
        for yaml_file in ${yaml_files}; do
            if grep -q "kind: ConfigMap" "${yaml_file}"; then
                log_info "Applying ConfigMap: ${yaml_file}"
                kubectl apply -f "${yaml_file}" --namespace="${NAMESPACE}" || log_warn "Failed to apply ${yaml_file}"
            fi
        done

        # Apply Secrets (excluding those restored separately)
        for yaml_file in ${yaml_files}; do
            if grep -q "kind: Secret" "${yaml_file}" && ! grep -q "type: kubernetes.io/service-account-token" "${yaml_file}"; then
                log_info "Applying Secret: ${yaml_file}"
                kubectl apply -f "${yaml_file}" --namespace="${NAMESPACE}" || log_warn "Failed to apply ${yaml_file}"
            fi
        done

        # Apply Deployments, StatefulSets, Services, etc.
        for yaml_file in ${yaml_files}; do
            if ! grep -q "kind: ConfigMap\|kind: Secret" "${yaml_file}"; then
                log_info "Applying resource: ${yaml_file}"
                kubectl apply -f "${yaml_file}" --namespace="${NAMESPACE}" || log_warn "Failed to apply ${yaml_file}"
            fi
        done
    fi

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "Kubernetes resources restore completed"
}

# ============================================================================
# SECRETS RESTORE
# ============================================================================

restore_secrets() {
    log_info "=== Starting Secrets Restore ==="

    # Find backup to restore
    local backup_file=""
    if [ "${BACKUP_TIMESTAMP}" = "latest" ]; then
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/secrets/" | tail -1 | awk '{print $4}')
        log_info "Using latest backup: ${backup_file}"
    else
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/secrets/" | grep "${BACKUP_TIMESTAMP}" | tail -1 | awk '{print $4}')

        if [ -z "${backup_file}" ]; then
            error_exit "No secrets backup found matching timestamp ${BACKUP_TIMESTAMP}"
        fi

        log_info "Using backup: ${backup_file}"
    fi

    confirm_action "Restore encrypted secrets from ${backup_file}?"

    # Get encryption key
    local encryption_key=""
    if command -v aws &>/dev/null; then
        encryption_key=$(aws secretsmanager get-secret-value \
            --secret-id "victor-ai/backup-encryption-key" \
            --query "SecretString" \
            --output text 2>/dev/null || echo "")
    fi

    if [ -z "${encryption_key}" ]; then
        log_error "Encryption key not found in AWS Secrets Manager"
        log_error "Please provide encryption key:"
        echo -n "Encryption key: "
        read -s encryption_key
        echo ""
    fi

    # Download encrypted backup
    local temp_dir="/tmp/secrets_restore_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    log_info "Downloading encrypted backup from S3"
    aws s3 cp "${BACKUP_BUCKET}/secrets/${backup_file}" "${temp_dir}/secrets.enc"

    # Verify checksum
    local checksum_file="${backup_file%.enc}.key_checksum"
    if aws s3 ls "${BACKUP_BUCKET}/secrets/${checksum_file}" &>/dev/null; then
        aws s3 cp "${BACKUP_BUCKET}/secrets/${checksum_file}" "${temp_dir}/key_checksum"

        local stored_checksum=$(cat "${temp_dir}/key_checksum")
        local calculated_checksum=$(echo -n "${encryption_key}" | sha256sum | awk '{print $1}')

        if [ "${calculated_checksum}" != "${stored_checksum}" ]; then
            log_error "Encryption key checksum mismatch!"
            log_error "Stored: ${stored_checksum}"
            log_error "Calculated: ${calculated_checksum}"
            error_exit "Incorrect encryption key"
        fi

        log_info "Encryption key verified"
    fi

    # Decrypt secrets
    log_info "Decrypting secrets backup"
    openssl enc -aes-256-cbc -d \
        -in "${temp_dir}/secrets.enc" \
        -out "${temp_dir}/secrets.json" \
        -k "${encryption_key}" || error_exit "Decryption failed"

    # Restore secrets
    log_info "Restoring secrets to Kubernetes"

    local secret_count=$(jq '.items | length' "${temp_dir}/secrets.json")
    log_info "Found ${secret_count} secrets to restore"

    for i in $(seq 0 $((secret_count - 1))); do
        local secret_name=$(jq -r ".items[${i}].metadata.name" "${temp_dir}/secrets.json")
        local secret_type=$(jq -r ".items[${i}].type" "${temp_dir}/secrets.json")

        # Skip service account tokens
        if [ "${secret_type}" = "kubernetes.io/service-account-token" ]; then
            log_info "Skipping service account token: ${secret_name}"
            continue
        fi

        log_info "Restoring secret: ${secret_name}"

        if [ "${DRY_RUN}" = "true" ]; then
            log_info "DRY RUN: Would restore secret ${secret_name}"
        else
            # Extract secret YAML and apply
            jq -r ".items[${i}]" "${temp_dir}/secrets.json" | \
                kubectl apply -f - --namespace="${NAMESPACE}" || log_warn "Failed to restore secret ${secret_name}"
        fi
    done

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "Secrets restore completed"
}

# ============================================================================
# HELM RELEASES RESTORE
# ============================================================================

restore_helm_releases() {
    log_info "=== Starting Helm Releases Restore ==="

    if ! command -v helm &>/dev/null; then
        log_warn "Helm not installed, skipping Helm restore"
        return 0
    fi

    # Find backup to restore
    local backup_file=""
    if [ "${BACKUP_TIMESTAMP}" = "latest" ]; then
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/helm/" | tail -1 | awk '{print $4}')
        log_info "Using latest backup: ${backup_file}"
    else
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/helm/" | grep "${BACKUP_TIMESTAMP}" | tail -1 | awk '{print $4}')

        if [ -z "${backup_file}" ]; then
            log_warn "No Helm backup found matching timestamp ${BACKUP_TIMESTAMP}"
            return 0
        fi

        log_info "Using backup: ${backup_file}"
    fi

    confirm_action "Restore Helm releases from ${backup_file}?"

    # Download backup
    local temp_dir="/tmp/helm_restore_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    log_info "Downloading backup from S3"
    aws s3 cp "${BACKUP_BUCKET}/helm/${backup_file}" "${temp_dir}/backup.tar.gz"

    # Extract
    log_info "Extracting backup"
    tar -xzf "${temp_dir}/backup.tar.gz" -C "${temp_dir}"

    # Find Helm release directories
    local release_dirs=$(find "${temp_dir}" -maxdepth 1 -type d -name "*_values_*" | sed 's/_values_[0-9]*_[0-9]*.yaml//' | sort -u)

    for release_dir in ${release_dirs}; do
        local release_name=$(basename "${release_dir}" | sed 's/_values.*//')

        log_info "Restoring Helm release: ${release_name}"

        # Find values file
        local values_file=$(find "${temp_dir}" -name "${release_name}_values_*.yaml" | head -1)

        if [ -z "${values_file}" ]; then
            log_warn "No values file found for ${release_name}"
            continue
        fi

        # Check if release exists
        if helm list -n "${NAMESPACE}" -q | grep -q "^${release_name}$"; then
            log_info "Release ${release_name} exists, upgrading"

            if [ "${DRY_RUN}" = "true" ]; then
                log_info "DRY RUN: Would upgrade release ${release_name}"
            else
                helm upgrade "${release_name}" -n "${NAMESPACE}" -f "${values_file}" || \
                    log_warn "Failed to upgrade release ${release_name}"
            fi
        else
            log_info "Release ${release_name} does not exist, installing"

            if [ "${DRY_RUN}" = "true" ]; then
                log_info "DRY RUN: Would install release ${release_name}"
            else
                helm install "${release_name}" -n "${NAMESPACE}" -f "${values_file}" || \
                    log_warn "Failed to install release ${release_name}"
            fi
        fi
    done

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "Helm releases restore completed"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Configuration Restore ==="
    log_info "Restore type: ${RESTORE_TYPE}"
    log_info "Backup timestamp: ${BACKUP_TIMESTAMP}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Dry run: ${DRY_RUN}"

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute restores based on type
    case "${RESTORE_TYPE}" in
        kubernetes|k8s)
            restore_kubernetes_resources || exit_code=$?
            ;;
        secrets)
            restore_secrets || exit_code=$?
            ;;
        helm)
            restore_helm_releases || exit_code=$?
            ;;
        all)
            restore_kubernetes_resources || exit_code=$?
            restore_secrets || exit_code=$?
            restore_helm_releases || exit_code=$?
            ;;
        *)
            log_error "Unknown restore type: ${RESTORE_TYPE}"
            log_error "Valid types: kubernetes, secrets, helm, all"
            exit 1
            ;;
    esac

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Configuration Restore Completed Successfully (${total_duration}s) ==="
        send_alert "INFO" "Configuration restore completed successfully in ${total_duration}s"
    else
        log_error "=== Configuration Restore Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Configuration restore failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
