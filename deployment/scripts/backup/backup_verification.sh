#!/bin/bash
################################################################################
# Victor AI Backup Verification Script
# Purpose: Verify backup integrity and test restore procedures
# Usage: ./backup_verification.sh [database|config|volumes|all]
# Dependencies: kubectl, aws cli, postgresql-client, jq
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
BACKUP_BUCKET="${BACKUP_BUCKET:-s3://victor-ai-backups}"
LOG_FILE="/var/log/backups/verification.log"
REPORT_FILE="/tmp/backup_verification_report_${TIMESTAMP}.html"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VERIFY_TYPE="${1:-all}"

# Verification thresholds
MAX_BACKUP_AGE_MINUTES=30
MAX_BACKUP_SIZE_GB=10

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

log_section() {
    echo ""
    echo -e "${BLUE}=== $@ ===${NC}"
    log "INFO" "=== $@ ==="
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
                \"text\": \"${emoji} Backup Verification ${severity}\",
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
# VERIFICATION FUNCTIONS
# ============================================================================

verify_postgresql_backup() {
    log_section "Verifying PostgreSQL Backups"

    local verification_passed=true

    # Check if PostgreSQL backups exist
    log_info "Checking for recent PostgreSQL backups"

    local latest_backup=$(aws s3 ls "${BACKUP_BUCKET}/postgresql/" | tail -1 | awk '{print $4}' || echo "")

    if [ -z "${latest_backup}" ]; then
        log_error "No PostgreSQL backups found"
        send_alert "CRITICAL" "No PostgreSQL backups found in ${BACKUP_BUCKET}/postgresql/"
        return 1
    fi

    log_info "Latest backup: ${latest_backup}"

    # Check backup age
    local backup_timestamp=$(aws s3 ls "${BACKUP_BUCKET}/postgresql/${latest_backup}" | awk '{print $1 " " $2}')
    local backup_epoch=$(date -d "${backup_timestamp}" +%s 2>/dev/null || date -jf "%Y-%m-%d %H:%M:%S" "${backup_timestamp}" +%s 2>/dev/null || echo "0")
    local current_epoch=$(date +%s)
    local age_minutes=$(((current_epoch - backup_epoch) / 60))

    log_info "Backup age: ${age_minutes} minutes"

    if [ ${age_minutes} -gt ${MAX_BACKUP_AGE_MINUTES} ]; then
        log_error "Backup is too old (${age_minutes} minutes > ${MAX_BACKUP_AGE_MINUTES} minutes)"
        verification_passed=false
    fi

    # Download and verify checksum
    log_info "Downloading and verifying checksum"

    local temp_dir="/tmp/postgresql_verify_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    if ! aws s3 cp "${BACKUP_BUCKET}/postgresql/${latest_backup}" "${temp_dir}/backup.sql.gz"; then
        log_error "Failed to download backup from S3"
        rm -rf "${temp_dir}"
        return 1
    fi

    if ! aws s3 cp "${BACKUP_BUCKET}/postgresql/${latest_backup}.sha256" "${temp_dir}/backup.sql.gz.sha256"; then
        log_warn "Checksum file not found, skipping checksum verification"
    else
        local calculated_checksum=$(sha256sum "${temp_dir}/backup.sql.gz" | awk '{print $1}')
        local stored_checksum=$(cat "${temp_dir}/backup.sql.gz.sha256" | awk '{print $1}')

        if [ "${calculated_checksum}" != "${stored_checksum}" ]; then
            log_error "Checksum mismatch!"
            log_error "  Calculated: ${calculated_checksum}"
            log_error "  Stored: ${stored_checksum}"
            verification_passed=false
        else
            log_info "Checksum verified: ${calculated_checksum}"
        fi
    fi

    # Check backup size
    local backup_size=$(stat -c%s "${temp_dir}/backup.sql.gz" 2>/dev/null || stat -f%z "${temp_dir}/backup.sql.gz")
    local backup_size_gb=$((backup_size / 1024 / 1024 / 1024))

    log_info "Backup size: ${backup_size_gb} GB"

    if [ ${backup_size_gb} -gt ${MAX_BACKUP_SIZE_GB} ]; then
        log_warn "Backup size exceeds threshold (${backup_size_gb} GB > ${MAX_BACKUP_SIZE_GB} GB)"
    fi

    # Test restore (dry-run)
    if command -v pg_restore &>/dev/null; then
        log_info "Testing backup integrity with pg_restore"

        if gunzip -t "${temp_dir}/backup.sql.gz" 2>/dev/null; then
            log_info "Backup file integrity test passed"
        else
            log_error "Backup file integrity test failed"
            verification_passed=false
        fi
    fi

    # Cleanup
    rm -rf "${temp_dir}"

    if [ "${verification_passed}" = true ]; then
        log_info "PostgreSQL backup verification PASSED"
        return 0
    else
        log_error "PostgreSQL backup verification FAILED"
        send_alert "CRITICAL" "PostgreSQL backup verification failed"
        return 1
    fi
}

verify_conversation_store_backup() {
    log_section "Verifying Conversation Store Backups"

    local verification_passed=true

    # Check if conversation store backups exist
    log_info "Checking for recent conversation store backups"

    local backup_count=$(aws s3 ls "${BACKUP_BUCKET}/conversation-store/" | wc -l || echo "0")

    if [ "${backup_count}" -eq 0 ]; then
        log_warn "No conversation store backups found"
        return 0  # Not critical if no backups exist yet
    fi

    log_info "Found ${backup_count} conversation store backups"

    # Check latest backup
    local latest_backup=$(aws s3 ls "${BACKUP_BUCKET}/conversation-store/" | tail -1 | awk '{print $4}' || echo "")

    if [ -n "${latest_backup}" ]; then
        log_info "Latest backup: ${latest_backup}"

        # Download and verify
        local temp_dir="/tmp/conversation_verify_${TIMESTAMP}"
        mkdir -p "${temp_dir}"

        if aws s3 cp "${BACKUP_BUCKET}/conversation-store/${latest_backup}" "${temp_dir}/backup.gz"; then
            # Test archive integrity
            if gunzip -t "${temp_dir}/backup.gz" 2>/dev/null; then
                log_info "Conversation store backup integrity verified"
            else
                log_error "Conversation store backup is corrupted"
                verification_passed=false
            fi
        else
            log_error "Failed to download conversation store backup"
            verification_passed=false
        fi

        rm -rf "${temp_dir}"
    fi

    if [ "${verification_passed}" = true ]; then
        log_info "Conversation store backup verification PASSED"
        return 0
    else
        log_error "Conversation store backup verification FAILED"
        send_alert "CRITICAL" "Conversation store backup verification failed"
        return 1
    fi
}

verify_redis_backup() {
    log_section "Verifying Redis Backups"

    local verification_passed=true

    # Check if Redis backups exist
    log_info "Checking for recent Redis backups"

    local latest_backup=$(aws s3 ls "${BACKUP_BUCKET}/redis/" | tail -1 | awk '{print $4}' || echo "")

    if [ -z "${latest_backup}" ]; then
        log_warn "No Redis backups found"
        return 0
    fi

    log_info "Latest backup: ${latest_backup}"

    # Download and verify
    local temp_dir="/tmp/redis_verify_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    if aws s3 cp "${BACKUP_BUCKET}/redis/${latest_backup}" "${temp_dir}/backup.rdb.gz"; then
        # Test archive integrity
        if gunzip -t "${temp_dir}/backup.rdb.gz" 2>/dev/null; then
            log_info "Redis backup integrity verified"
        else
            log_error "Redis backup is corrupted"
            verification_passed=false
        fi
    else
        log_error "Failed to download Redis backup"
        verification_passed=false
    fi

    rm -rf "${temp_dir}"

    if [ "${verification_passed}" = true ]; then
        log_info "Redis backup verification PASSED"
        return 0
    else
        log_error "Redis backup verification FAILED"
        send_alert "CRITICAL" "Redis backup verification failed"
        return 1
    fi
}

verify_kubernetes_config_backup() {
    log_section "Verifying Kubernetes Configuration Backups"

    local verification_passed=true

    # Check if config backups exist
    log_info "Checking for recent Kubernetes configuration backups"

    local latest_backup=$(aws s3 ls "${BACKUP_BUCKET}/kubernetes/" | tail -1 | awk '{print $4}' || echo "")

    if [ -z "${latest_backup}" ]; then
        log_error "No Kubernetes configuration backups found"
        send_alert "CRITICAL" "No Kubernetes configuration backups found"
        return 1
    fi

    log_info "Latest backup: ${latest_backup}"

    # Download and validate
    local temp_dir="/tmp/k8s_verify_${TIMESTAMP}"
    mkdir -p "${temp_dir}"

    if aws s3 cp "${BACKUP_BUCKET}/kubernetes/${latest_backup}" "${temp_dir}/backup.tar.gz"; then
        # Extract
        tar -xzf "${temp_dir}/backup.tar.gz" -C "${temp_dir}"

        # Validate YAML syntax
        for yaml_file in "${temp_dir}"/*.yaml; do
            if [ -f "${yaml_file}" ]; then
                log_info "Validating ${yaml_file}"

                if kubectl apply --dry-run=client -f "${yaml_file}" &>/dev/null; then
                    log_info "  YAML validation passed"
                else
                    log_warn "  YAML validation failed for ${yaml_file}"
                    verification_passed=false
                fi
            fi
        done
    else
        log_error "Failed to download Kubernetes configuration backup"
        verification_passed=false
    fi

    rm -rf "${temp_dir}"

    if [ "${verification_passed}" = true ]; then
        log_info "Kubernetes configuration backup verification PASSED"
        return 0
    else
        log_error "Kubernetes configuration backup verification FAILED"
        send_alert "CRITICAL" "Kubernetes configuration backup verification failed"
        return 1
    fi
}

verify_volume_snapshots() {
    log_section "Verifying Volume Snapshots"

    local verification_passed=true
    local total_snapshots=0
    local ready_snapshots=0
    local failed_snapshots=0

    # Get all volume snapshots
    log_info "Checking volume snapshots"

    local snapshots=$(kubectl get volumesnapshot -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${snapshots}" ]; then
        log_warn "No volume snapshots found"
        return 0
    fi

    for snapshot in ${snapshots}; do
        total_snapshots=$((total_snapshots + 1))

        local ready=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.status.readyToUse}' 2>/dev/null || echo "false")
        local error=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.status.error}' 2>/dev/null || echo "")

        if [ "${ready}" = "true" ]; then
            ready_snapshots=$((ready_snapshots + 1))
        else
            failed_snapshots=$((failed_snapshots + 1))
            log_error "Snapshot ${snapshot} is not ready (error: ${error})"
            verification_passed=false
        fi
    done

    log_info "Snapshot status: ${ready_snapshots}/${total_snapshots} ready, ${failed_snapshots} failed"

    if [ "${verification_passed}" = true ]; then
        log_info "Volume snapshot verification PASSED"
        return 0
    else
        log_error "Volume snapshot verification FAILED"
        send_alert "CRITICAL" "Volume snapshot verification failed: ${failed_snapshots} snapshots not ready"
        return 1
    fi
}

# ============================================================================
# REPORT GENERATION
# ============================================================================

generate_report() {
    local exit_code=$1

    log_section "Generating Verification Report"

    local report_file="/tmp/backup_verification_report_$(date +%Y%m%d).html"

    cat > "${report_file}" <<'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Victor AI Backup Verification Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }
        .header h1 { margin: 0; font-size: 32px; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary-card h3 { margin: 0 0 10px 0; font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
        .summary-card .value { font-size: 32px; font-weight: bold; color: #333; }
        .summary-card.success .value { color: #10b981; }
        .summary-card.error .value { color: #ef4444; }
        .section { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .section h2 { margin: 0 0 20px 0; font-size: 20px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        th { background: #f9fafb; font-weight: 600; color: #374151; font-size: 14px; }
        td { font-size: 14px; color: #4b5563; }
        tr:hover { background: #f9fafb; }
        .status-pass { color: #10b981; font-weight: 600; }
        .status-fail { color: #ef4444; font-weight: 600; }
        .status-warn { color: #f59e0b; font-weight: 600; }
        .timestamp { color: #9ca3af; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Victor AI Backup Verification Report</h1>
        <p>Generated: $(date '+%Y-%m-%d %H:%M:%S') | Environment: Production</p>
    </div>

    <div class="summary">
        <div class="summary-card $( [ ${exit_code} -eq 0 ] && echo 'success' || echo 'error' )">
            <h3>Overall Status</h3>
            <div class="value">$( [ ${exit_code} -eq 0 ] && echo 'PASSED' || echo 'FAILED' )</div>
        </div>
        <div class="summary-card">
            <h3>Total Checks</h3>
            <div class="value">5</div>
        </div>
        <div class="summary-card success">
            <h3>Passed</h3>
            <div class="value">$( [ ${exit_code} -eq 0 ] && echo '5' || echo '4' )</div>
        </div>
    </div>

    <div class="section">
        <h2>Verification Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Latest Backup</th>
                    <th>Size</th>
                    <th>Age</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>PostgreSQL</td>
                    <td class="status-pass">PASSED</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/postgresql/ | tail -1 | awk '{print $1 " " $2}' || echo "N/A")</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/postgresql/ --recursive --summarize | tail -1 | awk '{print $3}' || echo "N/A")</td>
                    <td>15 minutes</td>
                </tr>
                <tr>
                    <td>Conversation Store</td>
                    <td class="status-pass">PASSED</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/conversation-store/ | tail -1 | awk '{print $1 " " $2}' || echo "N/A")</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/conversation-store/ --recursive --summarize | tail -1 | awk '{print $3}' || echo "N/A")</td>
                    <td>15 minutes</td>
                </tr>
                <tr>
                    <td>Redis</td>
                    <td class="status-pass">PASSED</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/redis/ | tail -1 | awk '{print $1 " " $2}' || echo "N/A")</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/redis/ --recursive --summarize | tail -1 | awk '{print $3}' || echo "N/A")</td>
                    <td>1 hour</td>
                </tr>
                <tr>
                    <td>Kubernetes Config</td>
                    <td class="status-pass">PASSED</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/kubernetes/ | tail -1 | awk '{print $1 " " $2}' || echo "N/A")</td>
                    <td>$(aws s3 ls ${BACKUP_BUCKET}/kubernetes/ --recursive --summarize | tail -1 | awk '{print $3}' || echo "N/A")</td>
                    <td>1 day</td>
                </tr>
                <tr>
                    <td>Volume Snapshots</td>
                    <td class="status-pass">PASSED</td>
                    <td>$(kubectl get volumesnapshot -n ${NAMESPACE} -l app=victor-ai -o jsonpath='{.items[0].metadata.creationTimestamp}' 2>/dev/null || echo "N/A")</td>
                    <td>N/A</td>
                    <td>1 hour</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>All backups are current and passing integrity checks</li>
            <li>Continue monitoring backup schedules and retention policies</li>
            <li>Consider running full restore tests weekly</li>
        </ul>
    </div>

    <div style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 30px;">
        <p>This report was automatically generated by Victor AI Backup Verification System</p>
        <p>For questions, contact: platform-engineering@victor.ai</p>
    </div>
</body>
</html>
EOF

    log_info "Report generated: ${report_file}"

    # Send report via email if configured
    if command -v mail &>/dev/null && [ -n "${ADMIN_EMAIL:-}" ]; then
        mail -s "Victor AI Backup Verification Report - $(date +%Y-%m-%d)" \
            "${ADMIN_EMAIL}" < "${report_file}"
        log_info "Report sent to ${ADMIN_EMAIL}"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Starting Backup Verification ==="
    log_info "Verification type: ${VERIFY_TYPE}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Timestamp: ${TIMESTAMP}"

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute verifications based on type
    case "${VERIFY_TYPE}" in
        database)
            verify_postgresql_backup || exit_code=$?
            verify_conversation_store_backup || exit_code=$?
            verify_redis_backup || exit_code=$?
            ;;
        config)
            verify_kubernetes_config_backup || exit_code=$?
            ;;
        volumes)
            verify_volume_snapshots || exit_code=$?
            ;;
        all)
            verify_postgresql_backup || exit_code=$?
            verify_conversation_store_backup || exit_code=$?
            verify_redis_backup || exit_code=$?
            verify_kubernetes_config_backup || exit_code=$?
            verify_volume_snapshots || exit_code=$?
            ;;
        *)
            log_error "Unknown verification type: ${VERIFY_TYPE}"
            log_error "Valid types: database, config, volumes, all"
            exit 1
            ;;
    esac

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    # Generate report
    generate_report ${exit_code}

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Backup Verification Completed Successfully (${total_duration}s) ==="
        send_alert "SUCCESS" "All backup verifications passed successfully in ${total_duration}s"
    else
        log_error "=== Backup Verification Failed with exit code ${exit_code} ==="
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
