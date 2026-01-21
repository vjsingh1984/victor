#!/bin/bash
################################################################################
# Victor AI Database Restore Script
# Purpose: Restore PostgreSQL, conversation store, and Redis from backups
# Usage: ./restore_database.sh [postgresql|conversation|redis|all] <backup-timestamp>
# Example: ./restore_database.sh postgresql 20260120_143000
# Dependencies: kubectl, aws cli, postgresql-client, redis-tools
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
BACKUP_BUCKET="${BACKUP_BUCKET:-s3://victor-ai-backups}"
LOG_FILE="/var/log/backups/restore.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESTORE_TYPE="${1:-all}"
BACKUP_TIMESTAMP="${2:-latest}"

# Safety checks
AUTO_CONFIRM="${AUTO_CONFIRM:-false}"
SCALE_DOWN_APP="${SCALE_DOWN_APP:-true}"

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
        [ "${severity}" = "SUCCESS" ] && emoji=":white_check_mark:"

        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"${emoji} Database Restore ${severity}\",
                \"blocks\": [{
                    \"type\": \"section\",
                    \"text\": {
                        \"type\": \"mrkdwn\",
                        \"text\": \"*${severity}*: ${message}\"
                    }
                }]
            }" 2>/dev/null || true
    fi
}

error_exit() {
    log_error "$1"
    send_alert "CRITICAL" "$1"
    exit 1
}

# ============================================================================
# CONFIRMATION PROMPT
# ============================================================================

confirm_action() {
    local message="$1"

    if [ "${AUTO_CONFIRM}" = "true" ]; then
        log_info "Auto-confirm enabled, proceeding with restore"
        return 0
    fi

    log_important "${message}"
    log_important "This will REPLACE existing data with backup data!"
    echo -n "Type 'yes' to confirm: "
    read response

    if [ "${response}" != "yes" ]; then
        log_error "Restore cancelled by user"
        exit 1
    fi
}

# ============================================================================
# PRE-RESTORE CHECKS
# ============================================================================

pre_restore_checks() {
    log_info "Running pre-restore checks"

    # Check if namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
        error_exit "Namespace ${NAMESPACE} does not exist"
    fi

    # Check if backup bucket is accessible
    if ! aws s3 ls "${BACKUP_BUCKET}" &>/dev/null; then
        error_exit "Cannot access backup bucket ${BACKUP_BUCKET}"
    fi

    # Check available disk space
    local available_space=$(df /tmp | awk 'NR==2 {print $4}')
    local required_space=10737418240  # 10 GB in bytes

    if [ ${available_space} -lt ${required_space} ]; then
        log_error "Insufficient disk space for restore"
        log_error "Available: $((available_space / 1024 / 1024)) MB, Required: $((required_space / 1024 / 1024)) MB"
        error_exit "Insufficient disk space"
    fi

    log_info "Pre-restore checks passed"
}

# ============================================================================
# POSTGRESQL RESTORE FUNCTION
# ============================================================================

restore_postgresql() {
    log_info "=== Starting PostgreSQL Restore ==="

    # Get database credentials
    local db_deployment="postgresql"
    local db_secret="postgresql-secret"

    if ! kubectl get statefulset -n "${NAMESPACE}" "${db_deployment}" &>/dev/null; then
        log_error "PostgreSQL statefulset not found"
        return 1
    fi

    local db_user=$(kubectl get secret -n "${NAMESPACE}" "${db_secret}" -o jsonpath='{.data.username}' 2>/dev/null | base64 -d || echo "postgres")
    local db_pass=$(kubectl get secret -n "${NAMESPACE}" "${db_secret}" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || echo "")

    if [ -z "${db_pass}" ]; then
        error_exit "Failed to retrieve PostgreSQL credentials"
    fi

    # Find backup to restore
    local backup_file=""
    if [ "${BACKUP_TIMESTAMP}" = "latest" ]; then
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/postgresql/" | tail -1 | awk '{print $4}')
        log_info "Using latest backup: ${backup_file}"
    else
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/postgresql/" | grep "${BACKUP_TIMESTAMP}" | tail -1 | awk '{print $4}')

        if [ -z "${backup_file}" ]; then
            error_exit "No backup found matching timestamp ${BACKUP_TIMESTAMP}"
        fi

        log_info "Using backup: ${backup_file}"
    fi

    # Confirm restore
    confirm_action "Restore PostgreSQL from ${backup_file}?"

    # Scale down application
    if [ "${SCALE_DOWN_APP}" = "true" ]; then
        log_info "Scaling down application to prevent writes during restore"
        kubectl scale deployment/victor-ai -n "${NAMESPACE}" --replicas=0

        log_info "Waiting for pods to terminate"
        kubectl wait --for=delete pods -l app=victor-ai -n "${NAMESPACE}" --timeout=120s || true
    fi

    # Download backup
    local temp_dir="/tmp/postgresql_restore_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    log_info "Downloading backup from S3"
    if ! aws s3 cp "${BACKUP_BUCKET}/postgresql/${backup_file}" "${temp_dir}/backup.sql.gz"; then
        error_exit "Failed to download backup from S3"
    fi

    # Verify checksum if available
    if aws s3 ls "${BACKUP_BUCKET}/postgresql/${backup_file}.sha256" &>/dev/null; then
        log_info "Verifying backup checksum"

        aws s3 cp "${BACKUP_BUCKET}/postgresql/${backup_file}.sha256" "${temp_dir}/backup.sql.gz.sha256"

        local calculated_checksum=$(sha256sum "${temp_dir}/backup.sql.gz" | awk '{print $1}')
        local stored_checksum=$(cat "${temp_dir}/backup.sql.gz.sha256" | awk '{print $1}')

        if [ "${calculated_checksum}" != "${stored_checksum}" ]; then
            error_exit "Backup checksum verification failed"
        fi

        log_info "Checksum verified: ${calculated_checksum}"
    fi

    # Extract backup
    log_info "Extracting backup"
    gunzip "${temp_dir}/backup.sql.gz"

    # Create backup of current database
    log_info "Creating backup of current database (before restore)"
    kubectl exec -n "${NAMESPACE}" statefulset/${db_deployment} -- \
        psql -U "${db_user}" -c "ALTER DATABASE victor_ai RENAME TO victor_ai_pre_restore_${TIMESTAMP};"

    # Create new database
    kubectl exec -n "${NAMESPACE}" statefulset/${db_deployment} -- \
        psql -U "${db_user}" -c "CREATE DATABASE victor_ai;"

    # Restore backup
    log_info "Restoring database (this may take a while)"
    local start_time=$(date +%s)

    cat "${temp_dir}/backup.sql" | \
        kubectl exec -i -n "${NAMESPACE}" statefulset/${db_deployment} -- \
        psql -U "${db_user}" -d victor_ai || error_exit "Database restore failed"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_info "Database restore completed in ${duration} seconds"

    # Verify restore
    log_info "Verifying restore"

    local table_count=$(kubectl exec -n "${NAMESPACE}" statefulset/${db_deployment} -- \
        psql -U "${db_user}" -d victor_ai -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d ' ')

    log_info "Restored ${table_count} tables"

    # Restart database
    log_info "Restarting database"
    kubectl rollout restart statefulset/${db_deployment} -n "${NAMESPACE}"

    log_info "Waiting for database to be ready"
    kubectl wait --for=condition=ready pod -l app=${db_deployment} -n "${NAMESPACE}" --timeout=120s

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "PostgreSQL restore completed successfully"

    # Scale up application
    if [ "${SCALE_DOWN_APP}" = "true" ]; then
        log_info "Scaling up application"
        kubectl scale deployment/victor-ai -n "${NAMESPACE}" --replicas=3
    fi
}

# ============================================================================
# CONVERSATION STORE RESTORE FUNCTION
# ============================================================================

restore_conversation_store() {
    log_info "=== Starting Conversation Store Restore ==="

    local pod_label="app=victor-ai"

    # Find backup to restore
    local backup_prefix=""
    if [ "${BACKUP_TIMESTAMP}" = "latest" ]; then
        backup_prefix=$(aws s3 ls "${BACKUP_BUCKET}/conversation-store/" | tail -1 | awk '{print $4}')
        log_info "Using latest backup: ${backup_prefix}"
    else
        backup_prefix=$(aws s3 ls "${BACKUP_BUCKET}/conversation-store/" | grep "${BACKUP_TIMESTAMP}" | head -1 | awk '{print $4}')

        if [ -z "${backup_prefix}" ]; then
            log_warn "No backup found matching timestamp ${BACKUP_TIMESTAMP}"
            return 0
        fi

        log_info "Using backup: ${backup_prefix}"
    fi

    # Confirm restore
    confirm_action "Restore conversation store from ${backup_prefix}?"

    # Get pods
    local pods=$(kubectl get pods -n "${NAMESPACE}" -l "${pod_label}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${pods}" ]; then
        log_warn "No pods found with label ${pod_label}"
        return 0
    fi

    local temp_dir="/tmp/conversation_restore_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    for pod in ${pods}; do
        log_info "Processing pod: ${pod}"

        # Check if pod has conversation database
        if kubectl exec -n "${NAMESPACE}" "${pod}" -- test -f /data/conversations.db 2>/dev/null; then
            local backup_file="conversations_backup_${pod}_${BACKUP_TIMESTAMP}.db.gz"

            if ! aws s3 ls "${BACKUP_BUCKET}/conversation-store/${backup_file}" &>/dev/null; then
                log_warn "No backup found for ${pod}"
                continue
            fi

            log_info "Downloading backup for ${pod}"
            aws s3 cp "${BACKUP_BUCKET}/conversation-store/${backup_file}" "${temp_dir}/"

            # Backup current database
            kubectl exec -n "${NAMESPACE}" "${pod}" -- \
                mv /data/conversations.db /data/conversations.db.pre_restore

            # Copy and restore backup
            gunzip "${temp_dir}/${backup_file}"
            kubectl cp "${temp_dir}/conversations_backup_${pod}_${BACKUP_TIMESTAMP}.db" \
                "${NAMESPACE}/${pod}:/data/conversations.db"

            log_info "Conversation store restored for ${pod}"
        fi
    done

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "Conversation store restore completed"
}

# ============================================================================
# REDIS RESTORE FUNCTION
# ============================================================================

restore_redis() {
    log_info "=== Starting Redis Restore ==="

    local redis_deployment="redis"

    # Check if Redis exists
    if ! kubectl get deployment -n "${NAMESPACE}" "${redis_deployment}" &>/dev/null && \
       ! kubectl get statefulset -n "${NAMESPACE}" "${redis_deployment}" &>/dev/null; then
        log_warn "Redis not found in namespace ${NAMESPACE}"
        return 0
    fi

    # Find backup to restore
    local backup_file=""
    if [ "${BACKUP_TIMESTAMP}" = "latest" ]; then
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/redis/" | tail -1 | awk '{print $4}')
        log_info "Using latest backup: ${backup_file}"
    else
        backup_file=$(aws s3 ls "${BACKUP_BUCKET}/redis/" | grep "${BACKUP_TIMESTAMP}" | tail -1 | awk '{print $4}')

        if [ -z "${backup_file}" ]; then
            log_warn "No backup found matching timestamp ${BACKUP_TIMESTAMP}"
            return 0
        fi

        log_info "Using backup: ${backup_file}"
    fi

    # Confirm restore
    confirm_action "Restore Redis from ${backup_file}?"

    # Get Redis pod
    local redis_pod=$(kubectl get pods -n "${NAMESPACE}" -l "app=${redis_deployment}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${redis_pod}" ]; then
        log_warn "No Redis pods found"
        return 0
    fi

    # Download backup
    local temp_dir="/tmp/redis_restore_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    log_info "Downloading backup from S3"
    aws s3 cp "${BACKUP_BUCKET}/redis/${backup_file}" "${temp_dir}/"

    # Extract
    gunzip "${temp_dir}/redis_backup_*.rdb.gz"

    # Stop Redis
    log_info "Stopping Redis"
    kubectl scale statefulset/${redis_deployment} -n "${NAMESPACE}" --replicas=0

    # Wait for pod to terminate
    kubectl wait --for=delete pods -l app=${redis_deployment} -n "${NAMESPACE}" --timeout=60s || true

    # Start Redis with empty data
    kubectl scale statefulset/${redis_deployment} -n "${NAMESPACE}" --replicas=1

    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod -l app=${redis_deployment} -n "${NAMESPACE}" --timeout=60s

    # Copy RDB file
    log_info "Copying RDB file to Redis"
    kubectl cp "${temp_dir}/dump.rdb" "${NAMESPACE}/${redis_pod}:/data/dump.rdb"

    # Restart Redis
    log_info "Restarting Redis"
    kubectl rollout restart statefulset/${redis_deployment} -n "${NAMESPACE}"

    log_info "Waiting for Redis to be ready"
    kubectl wait --for=condition=ready pod -l app=${redis_deployment} -n "${NAMESPACE}" --timeout=60s

    # Verify Redis
    if kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- redis-cli PING &>/dev/null; then
        log_info "Redis is responding"
    else
        log_warn "Redis is not responding after restore"
    fi

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "Redis restore completed"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Database Restore ==="
    log_info "Restore type: ${RESTORE_TYPE}"
    log_info "Backup timestamp: ${BACKUP_TIMESTAMP}"
    log_info "Namespace: ${NAMESPACE}"

    # Pre-restore checks
    pre_restore_checks

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute restores based on type
    case "${RESTORE_TYPE}" in
        postgresql|postgres)
            restore_postgresql || exit_code=$?
            ;;
        conversation|conversations)
            restore_conversation_store || exit_code=$?
            ;;
        redis)
            restore_redis || exit_code=$?
            ;;
        all)
            restore_postgresql || exit_code=$?
            restore_conversation_store || exit_code=$?
            restore_redis || exit_code=$?
            ;;
        *)
            log_error "Unknown restore type: ${RESTORE_TYPE}"
            log_error "Valid types: postgresql, conversation, redis, all"
            exit 1
            ;;
    esac

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Database Restore Completed Successfully (${total_duration}s) ==="
        send_alert "SUCCESS" "Database restore completed successfully in ${total_duration}s"
    else
        log_error "=== Database Restore Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Database restore failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
