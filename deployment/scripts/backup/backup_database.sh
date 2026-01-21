#!/bin/bash
################################################################################
# Victor AI Database Backup Script
# Purpose: Automated backup of PostgreSQL, conversation store, and Redis
# Usage: ./backup_database.sh [postgresql|conversation|redis|all]
# Dependencies: kubectl, aws cli, postgresql-client, redis-tools
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
BACKUP_BUCKET="${BACKUP_BUCKET:-s3://victor-ai-backups}"
RETENTION_DAYS="${RETENTION_DAYS:-90}"
LOG_FILE="/var/log/backups/database.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_TYPE="${1:-all}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
        if [ "${severity}" = "CRITICAL" ]; then
            emoji=":rotating_light:"
        elif [ "${severity}" = "INFO" ]; then
            emoji=":white_check_mark:"
        fi

        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"${emoji} Database Backup ${severity}\",
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

# ============================================================================
# ERROR HANDLING
# ============================================================================

error_exit() {
    log_error "$1"
    send_alert "CRITICAL" "$1"
    exit 1
}

trap 'error_exit "Database backup script failed at line $LINENO"' ERR

# ============================================================================
# POSTGRESQL BACKUP FUNCTION
# ============================================================================

backup_postgresql() {
    log_info "Starting PostgreSQL backup"

    local db_deployment="postgresql"
    local db_secret="postgresql-secret"
    local backup_file="postgresql_backup_${TIMESTAMP}.sql.gz"

    # Check if PostgreSQL is running
    if ! kubectl get statefulset -n "${NAMESPACE}" "${db_deployment}" &>/dev/null; then
        log_warn "PostgreSQL not found in namespace ${NAMESPACE}, skipping"
        return 0
    fi

    # Get database credentials
    log_info "Retrieving PostgreSQL credentials"
    local db_host=$(kubectl get svc -n "${NAMESPACE}" "${db_deployment}" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "${db_deployment}")
    local db_port=5432
    local db_name="${POSTGRES_DB:-victor_ai}"
    local db_user=$(kubectl get secret -n "${NAMESPACE}" "${db_secret}" -o jsonpath='{.data.username}' 2>/dev/null | base64 -d || echo "postgres")
    local db_pass=$(kubectl get secret -n "${NAMESPACE}" "${db_secret}" -o jsonpath='{.data.password}' 2>/dev/null | base64 -d || echo "")

    if [ -z "${db_pass}" ]; then
        log_error "Failed to retrieve PostgreSQL password"
        return 1
    fi

    # Create temporary directory
    local backup_dir="/tmp/postgresql_backup_${TIMESTAMP}"
    mkdir -p "${backup_dir}"

    # Perform logical backup
    log_info "Executing pg_dump"
    local start_time=$(date +%s)

    PGPASSWORD="${db_pass}" pg_dump \
        -h "${db_host}" \
        -p "${db_port}" \
        -U "${db_user}" \
        -d "${db_name}" \
        --format=plain \
        --no-owner \
        --no-acl \
        --verbose \
        2>&1 | tee "${backup_dir}/pg_dump.log" | gzip > "${backup_dir}/${backup_file}" || error_exit "pg_dump failed"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_info "pg_dump completed in ${duration} seconds"

    # Calculate checksum
    log_info "Calculating checksum"
    local checksum=$(sha256sum "${backup_dir}/${backup_file}" | awk '{print $1}')
    echo "${checksum}" > "${backup_dir}/${backup_file}.sha256"

    # Get backup size
    local backup_size=$(stat -c%s "${backup_dir}/${backup_file}" 2>/dev/null || stat -f%z "${backup_dir}/${backup_file}")
    local backup_size_mb=$((backup_size / 1024 / 1024))
    log_info "Backup size: ${backup_size_mb} MB"

    # Upload to S3
    log_info "Uploading to S3"
    aws s3 cp "${backup_dir}/${backup_file}" "${BACKUP_BUCKET}/postgresql/" || error_exit "S3 upload failed"
    aws s3 cp "${backup_dir}/${backup_file}.sha256" "${BACKUP_BUCKET}/postgresql/" || error_exit "Checksum upload failed"

    # Tag backup with metadata
    aws s3api put-object-tagging \
        --bucket victor-ai-backups \
        --key "postgresql/${backup_file}" \
        --tagging "TagSet=[{Key=BackupType,Value=PostgreSQL},{Key=Timestamp,Value=${TIMESTAMP}},{Key=Retention,Value=${RETENTION_DAYS}days}]" 2>/dev/null || true

    # Cleanup
    rm -rf "${backup_dir}"

    log_info "PostgreSQL backup completed successfully"

    # Log to CloudWatch/DataDog if available
    if command -v aws &>/dev/null; then
        aws cloudwatch put-metric-data \
            --namespace "VictorAI/Backups" \
            --metric-name "PostgreSQLBackupSize" \
            --value "${backup_size}" \
            --unit Bytes 2>/dev/null || true

        aws cloudwatch put-metric-data \
            --namespace "VictorAI/Backups" \
            --metric-name "PostgreSQLBackupDuration" \
            --value "${duration}" \
            --unit Seconds 2>/dev/null || true
    fi
}

# ============================================================================
# CONVERSATION STORE BACKUP FUNCTION
# ============================================================================

backup_conversation_store() {
    log_info "Starting conversation store backup"

    local pod_label="app=victor-ai"
    local backup_count=0

    # Get pods
    local pods=$(kubectl get pods -n "${NAMESPACE}" -l "${pod_label}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || "")

    if [ -z "${pods}" ]; then
        log_warn "No pods found with label ${pod_label}"
        return 0
    fi

    # Create temporary directory
    local backup_dir="/tmp/conversation_backup_${TIMESTAMP}"
    mkdir -p "${backup_dir}"

    for pod in ${pods}; do
        log_info "Processing pod: ${pod}"

        # Check for SQLite database
        if kubectl exec -n "${NAMESPACE}" "${pod}" -- test -f /data/conversations.db 2>/dev/null; then
            log_info "Found SQLite database in ${pod}"

            local backup_file="conversations_backup_${pod}_${TIMESTAMP}.db.gz"

            # Create SQLite backup using .backup command (online backup)
            kubectl exec -n "${NAMESPACE}" "${pod}" -- \
                sqlite3 /data/conversations.db ".backup /tmp/conversations_backup_${TIMESTAMP}.db" || {
                log_warn "SQLite backup failed for ${pod}, continuing"
                continue
            }

            # Copy backup locally
            kubectl cp "${NAMESPACE}/${pod}:/tmp/conversations_backup_${TIMESTAMP}.db" \
                "${backup_dir}/conversations_backup_${pod}_${TIMESTAMP}.db" || {
                log_warn "Failed to copy backup from ${pod}"
                continue
            }

            # Compress
            gzip "${backup_dir}/conversations_backup_${pod}_${TIMESTAMP}.db"

            # Upload to S3
            aws s3 cp "${backup_dir}/${backup_file}" "${BACKUP_BUCKET}/conversation-store/" || {
                log_warn "Failed to upload backup from ${pod}"
                continue
            }

            # Cleanup in pod
            kubectl exec -n "${NAMESPACE}" "${pod}" -- rm -f "/tmp/conversations_backup_${TIMESTAMP}.db" || true

            backup_count=$((backup_count + 1))
        fi

        # Check for file-based conversation store
        if kubectl exec -n "${NAMESPACE}" "${pod}" -- test -d /data/conversations 2>/dev/null; then
            log_info "Found file-based conversation store in ${pod}"

            local backup_file="conversations_${pod}_${TIMESTAMP}.tar.gz"

            # Create tar archive
            kubectl exec -n "${NAMESPACE}" "${pod}" -- \
                tar -czf "/tmp/conversations_backup_${TIMESTAMP}.tar.gz" -C /data conversations || {
                log_warn "Failed to create archive in ${pod}"
                continue
            }

            # Copy backup locally
            kubectl cp "${NAMESPACE}/${pod}:/tmp/conversations_backup_${TIMESTAMP}.tar.gz" \
                "${backup_dir}/conversations_${pod}_${TIMESTAMP}.tar.gz" || {
                log_warn "Failed to copy archive from ${pod}"
                continue
            }

            # Upload to S3
            aws s3 cp "${backup_dir}/${backup_file}" "${BACKUP_BUCKET}/conversation-store/" || {
                log_warn "Failed to upload archive from ${pod}"
                continue
            }

            # Cleanup in pod
            kubectl exec -n "${NAMESPACE}" "${pod}" -- rm -f "/tmp/conversations_backup_${TIMESTAMP}.tar.gz" || true

            backup_count=$((backup_count + 1))
        fi
    done

    # Cleanup
    rm -rf "${backup_dir}"

    if [ ${backup_count} -eq 0 ]; then
        log_warn "No conversation stores backed up"
    else
        log_info "Conversation store backup completed: ${backup_count} backups"
    fi
}

# ============================================================================
# REDIS BACKUP FUNCTION
# ============================================================================

backup_redis() {
    log_info "Starting Redis backup"

    local redis_deployment="redis"

    # Check if Redis is running
    if ! kubectl get deployment -n "${NAMESPACE}" "${redis_deployment}" &>/dev/null && \
       ! kubectl get statefulset -n "${NAMESPACE}" "${redis_deployment}" &>/dev/null; then
        log_warn "Redis not found in namespace ${NAMESPACE}, skipping"
        return 0
    fi

    # Get Redis pods
    local redis_pod=$(kubectl get pods -n "${NAMESPACE}" -l "app=${redis_deployment}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${redis_pod}" ]; then
        log_warn "No Redis pods found"
        return 0
    fi

    log_info "Triggering Redis BGSAVE on ${redis_pod}"

    # Trigger background save
    local lastsave_before=$(kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- redis-cli LASTSAVE 2>/dev/null || echo "0")
    kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- redis-cli BGSAVE 2>/dev/null || {
        log_warn "Failed to trigger BGSAVE"
        return 0
    }

    # Wait for save to complete
    local max_wait=300  # 5 minutes max wait
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        local lastsave_after=$(kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- redis-cli LASTSAVE 2>/dev/null || echo "0")
        if [ "${lastsave_after}" != "${lastsave_before}" ]; then
            break
        fi
        sleep 5
        waited=$((waited + 5))
        echo -n "."
    done
    echo ""

    if [ ${waited} -ge ${max_wait} ]; then
        log_error "Redis BGSAVE timeout"
        return 1
    fi

    # Get RDB file
    log_info "Copying RDB file from ${redis_pod}"

    local backup_file="redis_backup_${TIMESTAMP}.rdb.gz"
    local temp_dir="/tmp/redis_backup_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    kubectl cp "${NAMESPACE}/${redis_pod}:/data/dump.rdb" "${temp_dir}/dump.rdb" || {
        log_error "Failed to copy RDB file from ${redis_pod}"
        rm -rf "${temp_dir}"
        return 1
    }

    # Compress
    gzip "${temp_dir}/dump.rdb"

    # Upload to S3
    log_info "Uploading to S3"
    aws s3 cp "${temp_dir}/${backup_file}" "${BACKUP_BUCKET}/redis/" || {
        log_error "Failed to upload Redis backup"
        rm -rf "${temp_dir}"
        return 1
    }

    # Cleanup
    rm -rf "${temp_dir}"

    log_info "Redis backup completed successfully"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Database Backup ==="
    log_info "Backup type: ${BACKUP_TYPE}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Timestamp: ${TIMESTAMP}"

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute backups based on type
    case "${BACKUP_TYPE}" in
        postgresql|postgres)
            backup_postgresql || exit_code=$?
            ;;
        conversation|conversations)
            backup_conversation_store || exit_code=$?
            ;;
        redis)
            backup_redis || exit_code=$?
            ;;
        all)
            backup_postgresql || exit_code=$?
            backup_conversation_store || exit_code=$?
            backup_redis || exit_code=$?
            ;;
        *)
            log_error "Unknown backup type: ${BACKUP_TYPE}"
            log_error "Valid types: postgresql, conversation, redis, all"
            exit 1
            ;;
    esac

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Database Backup Completed Successfully (${total_duration}s) ==="
        send_alert "INFO" "Database backup completed successfully in ${total_duration}s"
    else
        log_error "=== Database Backup Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Database backup failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
