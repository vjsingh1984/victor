#!/bin/bash
################################################################################
# Victor AI Configuration Backup Script
# Purpose: Backup Kubernetes resources, secrets, Helm releases, and configuration
# Usage: ./backup_config.sh [kubernetes|secrets|helm|all]
# Dependencies: kubectl, helm, aws cli, git, jq
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
BACKUP_BUCKET="${BACKUP_BUCKET:-s3://victor-ai-backups}"
GIT_REPO="${CONFIG_GIT_REPO:-}"
RETENTION_DAYS=365
LOG_FILE="/var/log/backups/config.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_TYPE="${1:-all}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

send_alert() {
    local severity=$1
    local message=$2

    if [ -n "${ALERT_WEBHOOK}" ]; then
        local emoji=":warning:"
        [ "${severity}" = "CRITICAL" ] && emoji=":rotating_light:"
        [ "${severity}" = "INFO" ] && emoji=":white_check_mark:"

        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"${emoji} Config Backup ${severity}\", \"blocks\": [{\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": \"*${severity}*: ${message}\"}}]}" 2>/dev/null || true
    fi
}

error_exit() {
    log_error "$1"
    send_alert "CRITICAL" "$1"
    exit 1
}

trap 'error_exit "Config backup script failed at line $LINENO"' ERR

# ============================================================================
# KUBERNETES RESOURCES BACKUP
# ============================================================================

backup_kubernetes_resources() {
    log_info "Starting Kubernetes resources backup"

    local backup_dir="/tmp/kubernetes_backup_${TIMESTAMP}"
    mkdir -p "${backup_dir}"

    # Backup all namespaced resources
    log_info "Backing up namespaced resources"

    kubectl get all,configmap,secret,pvc,ingress,role,rolebinding,serviceaccount,networkpolicy \
        -n "${NAMESPACE}" \
        -o yaml > "${backup_dir}/kubernetes_resources_${TIMESTAMP}.yaml" 2>/dev/null || {
        log_warn "Failed to backup some namespaced resources, continuing..."
    }

    # Filter out sensitive data from secrets (only keep metadata)
    log_info "Sanitizing secrets in backup"

    if kubectl get secrets -n "${NAMESPACE}" -o yaml > "${backup_dir}/secrets_metadata_${TIMESTAMP}.yaml" 2>/dev/null; then
        # Remove secret data (keep only metadata)
        awk '
            /^  data:/,/^$/ { if (!/^$/) next }
            { print }
        ' "${backup_dir}/secrets_metadata_${TIMESTAMP}.yaml" > "${backup_dir}/secrets_metadata_${TIMESTAMP}.sanitized.yaml"
        mv "${backup_dir}/secrets_metadata_${TIMESTAMP}.sanitized.yaml" "${backup_dir}/secrets_metadata_${TIMESTAMP}.yaml"
    fi

    # Backup cluster-scoped resources (filtered)
    log_info "Backing up cluster-scoped resources"

    # Only backup custom resources and cluster roles related to Victor AI
    kubectl get customresourcedefinition -o json | \
        jq '.items[] | select(.metadata.name | contains("victor") or contains("kafka") or contains("redis"))' | \
        jq -s '{items: .}' > "${backup_dir}/crds_${TIMESTAMP}.json" 2>/dev/null || true

    kubectl get clusterrole,clusterrolebinding -o json | \
        jq '.items[] | select(.metadata.namespace == "'"${NAMESPACE}"'" or .metadata.name | contains("victor"))' | \
        jq -s '{items: .}' > "${backup_dir}/cluster_resources_${TIMESTAMP}.json" 2>/dev/null || true

    # Get resource counts for verification
    local deployment_count=$(kubectl get deployments -n "${NAMESPACE}" --no-headers 2>/dev/null | wc -l || echo "0")
    local pod_count=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | wc -l || echo "0")
    local configmap_count=$(kubectl get configmaps -n "${NAMESPACE}" --no-headers 2>/dev/null | wc -l || echo "0")

    log_info "Backed up: ${deployment_count} deployments, ${pod_count} pods, ${configmap_count} configmaps"

    # Create archive
    log_info "Creating archive"
    tar -czf "${backup_dir}/kubernetes_backup_${TIMESTAMP}.tar.gz" -C "${backup_dir}" \
        "kubernetes_resources_${TIMESTAMP}.yaml" \
        "secrets_metadata_${TIMESTAMP}.yaml" \
        "crds_${TIMESTAMP}.json" \
        "cluster_resources_${TIMESTAMP}.json" 2>/dev/null || true

    # Upload to S3
    log_info "Uploading to S3"
    aws s3 cp "${backup_dir}/kubernetes_backup_${TIMESTAMP}.tar.gz" \
        "${BACKUP_BUCKET}/kubernetes/" || error_exit "S3 upload failed"

    # Git commit if repository is configured
    if [ -n "${GIT_REPO}" ] && [ -d "${GIT_REPO}" ]; then
        log_info "Committing to Git repository"

        cd "${GIT_REPO}"

        # Copy files to Git repo
        cp "${backup_dir}/kubernetes_resources_${TIMESTAMP}.yaml" "${GIT_REPO}/kubernetes/"
        cp "${backup_dir}/secrets_metadata_${TIMESTAMP}.yaml" "${GIT_REPO}/kubernetes/"

        git add kubernetes/
        git commit -m "Automated Kubernetes backup: ${TIMESTAMP}" || log_warn "No changes to commit"
        git push origin main 2>/dev/null || log_warn "Git push failed"
    fi

    # Cleanup
    rm -rf "${backup_dir}"

    log_info "Kubernetes resources backup completed successfully"
}

# ============================================================================
# SECRETS BACKUP
# ============================================================================

backup_secrets() {
    log_info "Starting secrets backup (encrypted)"

    local backup_dir="/tmp/secrets_backup_${TIMESTAMP}"
    mkdir -p "${backup_dir}"

    # Get encryption key from AWS Secrets Manager or generate
    local encryption_key=""
    if command -v aws &>/dev/null; then
        encryption_key=$(aws secretsmanager get-secret-value \
            --secret-id "victor-ai/backup-encryption-key" \
            --query "SecretString" \
            --output text 2>/dev/null || echo "")
    fi

    if [ -z "${encryption_key}" ]; then
        # Generate temporary encryption key
        encryption_key=$(openssl rand -hex 32)
        log_warn "Generated temporary encryption key (not stored in Secrets Manager)"
    fi

    # Get all secrets (excluding service account tokens)
    log_info "Extracting secrets from namespace ${NAMESPACE}"

    kubectl get secrets -n "${NAMESPACE}" -o json | \
        jq '.items[] | select(.type != "kubernetes.io/service-account-token")' | \
        jq -s '{items: .}' > "${backup_dir}/secrets_${TIMESTAMP}.json"

    # Count secrets
    local secret_count=$(jq '.items | length' "${backup_dir}/secrets_${TIMESTAMP}.json")
    log_info "Found ${secret_count} secrets to backup"

    # Encrypt secrets
    log_info "Encrypting secrets backup"

    openssl enc -aes-256-cbc -salt \
        -in "${backup_dir}/secrets_${TIMESTAMP}.json" \
        -out "${backup_dir}/secrets_${TIMESTAMP}.enc" \
        -k "${encryption_key}" || error_exit "Encryption failed"

    # Store encryption key checksum for verification
    local key_checksum=$(echo -n "${encryption_key}" | sha256sum | awk '{print $1}')
    echo "${key_checksum}" > "${backup_dir}/secrets_${TIMESTAMP}.key_checksum"

    # Upload encrypted backup
    log_info "Uploading encrypted backup to S3"

    aws s3 cp "${backup_dir}/secrets_${TIMESTAMP}.enc" \
        "${BACKUP_BUCKET}/secrets/" || error_exit "S3 upload failed"

    aws s3 cp "${backup_dir}/secrets_${TIMESTAMP}.key_checksum" \
        "${BACKUP_BUCKET}/secrets/" || error_exit "Checksum upload failed"

    # Tag with metadata
    aws s3api put-object-tagging \
        --bucket victor-ai-backups \
        --key "secrets/secrets_${TIMESTAMP}.enc" \
        --tagging 'TagSet=[{Key=Encrypted,Value=true},{Key=Type,Value=Secrets},{Key=Retention,Value=365days}]' 2>/dev/null || true

    # Cleanup
    rm -rf "${backup_dir}"

    log_info "Secrets backup completed successfully (${secret_count} secrets encrypted)"
}

# ============================================================================
# HELM RELEASES BACKUP
# ============================================================================

backup_helm_releases() {
    log_info "Starting Helm releases backup"

    if ! command -v helm &>/dev/null; then
        log_warn "Helm not installed, skipping Helm backup"
        return 0
    fi

    local backup_dir="/tmp/helm_backup_${TIMESTAMP}"
    mkdir -p "${backup_dir}"

    # Get all Helm releases in namespace
    log_info "Discovering Helm releases in namespace ${NAMESPACE}"

    local helm_releases=$(helm list -n "${NAMESPACE}" -q 2>/dev/null || echo "")

    if [ -z "${helm_releases}" ]; then
        log_warn "No Helm releases found in namespace ${NAMESPACE}"
        rm -rf "${backup_dir}"
        return 0
    fi

    local release_count=0

    for release in ${helm_releases}; do
        log_info "Backing up Helm release: ${release}"

        # Get release values
        helm get values -n "${NAMESPACE}" "${release}" -a > \
            "${backup_dir}/${release}_values_${TIMESTAMP}.yaml" 2>/dev/null || {
            log_warn "Failed to get values for ${release}"
            continue
        }

        # Get release manifest
        helm get manifest -n "${NAMESPACE}" "${release}" > \
            "${backup_dir}/${release}_manifest_${TIMESTAMP}.yaml" 2>/dev/null || true

        # Get release hooks (if any)
        helm get hooks -n "${NAMESPACE}" "${release}" > \
            "${backup_dir}/${release}_hooks_${TIMESTAMP}.yaml" 2>/dev/null || true

        # Get release metadata
        helm status -n "${NAMESPACE}" "${release}" -o json > \
            "${backup_dir}/${release}_status_${TIMESTAMP}.json" 2>/dev/null || true

        release_count=$((release_count + 1))
    done

    log_info "Backed up ${release_count} Helm releases"

    # Create archive
    log_info "Creating Helm archive"
    tar -czf "${backup_dir}/helm_backup_${TIMESTAMP}.tar.gz" -C "${backup_dir}" .

    # Upload to S3
    log_info "Uploading to S3"
    aws s3 cp "${backup_dir}/helm_backup_${TIMESTAMP}.tar.gz" \
        "${BACKUP_BUCKET}/helm/" || error_exit "S3 upload failed"

    # Cleanup
    rm -rf "${backup_dir}"

    log_info "Helm releases backup completed successfully"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Configuration Backup ==="
    log_info "Backup type: ${BACKUP_TYPE}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Timestamp: ${TIMESTAMP}"

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute backups based on type
    case "${BACKUP_TYPE}" in
        kubernetes|k8s)
            backup_kubernetes_resources || exit_code=$?
            ;;
        secrets)
            backup_secrets || exit_code=$?
            ;;
        helm)
            backup_helm_releases || exit_code=$?
            ;;
        all)
            backup_kubernetes_resources || exit_code=$?
            backup_secrets || exit_code=$?
            backup_helm_releases || exit_code=$?
            ;;
        *)
            log_error "Unknown backup type: ${BACKUP_TYPE}"
            log_error "Valid types: kubernetes, secrets, helm, all"
            exit 1
            ;;
    esac

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Configuration Backup Completed Successfully (${total_duration}s) ==="
        send_alert "INFO" "Configuration backup completed successfully in ${total_duration}s"
    else
        log_error "=== Configuration Backup Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Configuration backup failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
