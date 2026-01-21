#!/bin/bash
################################################################################
# Victor AI Backup Cleanup Script
# Purpose: Remove old backups based on retention policies
# Usage: ./cleanup_old_backups.sh [database|config|volumes|all]
# Dependencies: kubectl, aws cli
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKUP_BUCKET="${BACKUP_BUCKET:-s3://victor-ai-backups}"
NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
RETENTION_POSTGRESQL_DAYS="${RETENTION_POSTGRESQL_DAYS:-90}"
RETENTION_REDIS_DAYS="${RETENTION_REDIS_DAYS:-30}"
RETENTION_CONFIG_DAYS="${RETENTION_CONFIG_DAYS:-365}"
RETENTION_CONVERSATION_DAYS="${RETENTION_CONVERSATION_DAYS:-90}"
RETENTION_VOLUME_DAYS="${RETENTION_VOLUME_DAYS:-30}"
LOG_FILE="/var/log/backups/cleanup.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CLEANUP_TYPE="${1:-all}"

# Dry run mode (set to "false" to actually delete files)
DRY_RUN="${DRY_RUN:-true}"

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

log_dry() {
    log "DRY-RUN" "${BLUE}[DRY RUN]${NC} $@"
}

send_alert() {
    local severity=$1
    local message=$2

    if [ -n "${ALERT_WEBHOOK}" ]; then
        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"Backup Cleanup ${severity}\", \"blocks\": [{\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": \"*${severity}*: ${message}\"}}]}" 2>/dev/null || true
    fi
}

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================

cleanup_s3_backups() {
    local prefix=$1
    local retention_days=$2

    log_info "Cleaning up ${prefix} (retention: ${retention_days} days)"

    local cutoff_date=$(date -d "${retention_days} days ago" +%Y%m%d 2>/dev/null || date -v-${retention_DAYS}d +%Y%m%d)

    # Get list of old backups
    local old_backups=$(aws s3 ls "${BACKUP_BUCKET}/${prefix}/" | awk -v cutoff="${cutoff_date}" '
        {
            # Parse date (format: YYYY-MM-DD HH:MM:SS)
            split($1, date_parts, "-")
            file_date = date_parts[1] date_parts[2] date_parts[3]

            # Compare with cutoff date
            if (file_date < cutoff) {
                print $4
            }
        }
    ')

    if [ -z "${old_backups}" ]; then
        log_info "No old backups found in ${prefix}"
        return 0
    fi

    local count=0
    local total_size=0

    for backup in ${old_backups}; do
        # Get file size
        local size=$(aws s3 ls "${BACKUP_BUCKET}/${prefix}/${backup}" | awk '{print $3}')
        total_size=$((total_size + size))

        if [ "${DRY_RUN}" = "true" ]; then
            log_dry "Would delete: ${backup} (${size} bytes)"
        else
            log_info "Deleting: ${backup} (${size} bytes)"
            aws s3 rm "${BACKUP_BUCKET}/${prefix}/${backup}" || log_warn "Failed to delete ${backup}"
        fi

        count=$((count + 1))
    done

    # Convert bytes to GB for reporting
    local size_gb=$(echo "scale=2; ${total_size} / 1024 / 1024 / 1024" | bc)

    if [ "${DRY_RUN}" = "true" ]; then
        log_dry "Would delete ${count} files (${size_gb} GB) from ${prefix}"
    else
        log_info "Deleted ${count} files (${size_gb} GB) from ${prefix}"
    fi

    return 0
}

cleanup_postgresql_backups() {
    log_info "=== Cleaning up PostgreSQL Backups ==="

    cleanup_s3_backups "postgresql" "${RETENTION_POSTGRESQL_DAYS}"
}

cleanup_redis_backups() {
    log_info "=== Cleaning up Redis Backups ==="

    cleanup_s3_backups "redis" "${RETENTION_REDIS_DAYS}"
}

cleanup_conversation_store_backups() {
    log_info "=== Cleaning up Conversation Store Backups ==="

    cleanup_s3_backups "conversation-store" "${RETENTION_CONVERSATION_DAYS}"
}

cleanup_config_backups() {
    log_info "=== Cleaning up Configuration Backups ==="

    cleanup_s3_backups "kubernetes" "${RETENTION_CONFIG_DAYS}"
    cleanup_s3_backups "helm" "${RETENTION_CONFIG_DAYS}"

    # Note: Secrets in config/ should be kept longer (365 days)
    # Only clean up if older than 1 year
    cleanup_s3_backups "secrets" "365"
}

cleanup_volume_snapshots() {
    log_info "=== Cleaning up Volume Snapshots ==="

    # Get all snapshots with Victor AI label
    local snapshots=$(kubectl get volumesnapshot -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${snapshots}" ]; then
        log_info "No volume snapshots found"
        return 0
    fi

    local cutoff_date=$(date -d "${RETENTION_VOLUME_DAYS} days ago" +%s 2>/dev/null || date -v-${RETENTION_VOLUME_DAYS}d +%s)
    local count=0

    for snapshot in ${snapshots}; do
        # Get snapshot creation time
        local creation_time=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "")

        if [ -z "${creation_time}" ]; then
            continue
        fi

        # Convert to epoch
        local creation_epoch=$(date -d "${creation_time}" +%s 2>/dev/null || date -jf "%Y-%m-%dT%H:%M:%SZ" "${creation_time}" +%s 2>/dev/null || echo "0")

        # Check if snapshot is older than retention period
        if [ ${creation_epoch} -lt ${cutoff_date} ]; then
            if [ "${DRY_RUN}" = "true" ]; then
                log_dry "Would delete snapshot: ${snapshot} (created: ${creation_time})"
            else
                log_info "Deleting snapshot: ${snapshot} (created: ${creation_time})"
                kubectl delete volumesnapshot -n "${NAMESPACE}" "${snapshot}" --timeout=60s || log_warn "Failed to delete snapshot ${snapshot}"
            fi
            count=$((count + 1))
        fi
    done

    if [ "${DRY_RUN}" = "true" ]; then
        log_dry "Would delete ${count} volume snapshots"
    else
        log_info "Deleted ${count} volume snapshots"
    fi
}

cleanup_glacier_archives() {
    log_info "=== Cleaning up Glacier Archives ==="

    # Glacier archives are typically kept for 7 years for compliance
    # This function would only be used if you need to delete archives older than 7 years

    local retention_years=7
    local cutoff_date=$(date -d "${retention_years} years ago" +%Y%m%d 2>/dev/null || date -v-${retention_years}y +%Y%m%d)

    log_info "Glacier retention: ${retention_years} years (cutoff: ${cutoff_date})"

    # List and delete old Glacier archives
    # Note: This requires special Glacier commands and may incur retrieval costs

    log_warn "Glacier cleanup not implemented (archives kept for 7 years for compliance)"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Backup Cleanup ==="
    log_info "Cleanup type: ${CLEANUP_TYPE}"
    log_info "Dry run mode: ${DRY_RUN}"
    log_info "Timestamp: ${TIMESTAMP}"

    if [ "${DRY_RUN}" = "true" ]; then
        log_info "DRY RUN MODE - No files will be deleted"
        log_info "To actually delete files, run: DRY_RUN=false $0 ${CLEANUP_TYPE}"
    fi

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute cleanup based on type
    case "${CLEANUP_TYPE}" in
        database)
            cleanup_postgresql_backups || exit_code=$?
            cleanup_redis_backups || exit_code=$?
            cleanup_conversation_store_backups || exit_code=$?
            ;;
        config)
            cleanup_config_backups || exit_code=$?
            ;;
        volumes)
            cleanup_volume_snapshots || exit_code=$?
            ;;
        all)
            cleanup_postgresql_backups || exit_code=$?
            cleanup_redis_backups || exit_code=$?
            cleanup_conversation_store_backups || exit_code=$?
            cleanup_config_backups || exit_code=$?
            cleanup_volume_snapshots || exit_code=$?
            ;;
        glacier)
            cleanup_glacier_archives || exit_code=$?
            ;;
        *)
            log_error "Unknown cleanup type: ${CLEANUP_TYPE}"
            log_error "Valid types: database, config, volumes, glacier, all"
            exit 1
            ;;
    esac

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Backup Cleanup Completed Successfully (${total_duration}s) ==="

        if [ "${DRY_RUN}" = "false" ]; then
            send_alert "INFO" "Backup cleanup completed successfully in ${total_duration}s"
        fi
    else
        log_error "=== Backup Cleanup Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Backup cleanup failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
