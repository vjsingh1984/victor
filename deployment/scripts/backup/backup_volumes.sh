#!/bin/bash
################################################################################
# Victor AI Volume Snapshot Script
# Purpose: Create and manage Kubernetes volume snapshots
# Usage: ./backup_volumes.sh [create|list|cleanup|verify]
# Dependencies: kubectl, aws cli
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
SNAPSHOT_CLASS="${VOLUMESNAPSHOT_CLASS:-victor-ai-snapshot-class}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
RETENTION_COUNT="${RETENTION_COUNT:-72}"  # Keep last 72 hourly snapshots (3 days)
LOG_FILE="/var/log/backups/volumes.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ACTION="${1:-create}"

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
            -d "{\"text\": \"${emoji} Volume Backup ${severity}\", \"blocks\": [{\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": \"*${severity}*: ${message}\"}}]}" 2>/dev/null || true
    fi
}

error_exit() {
    log_error "$1"
    send_alert "CRITICAL" "$1"
    exit 1
}

trap 'error_exit "Volume backup script failed at line $LINENO"' ERR

# ============================================================================
# CREATE SNAPSHOT FUNCTION
# ============================================================================

create_snapshots() {
    log_info "=== Starting Volume Snapshot Creation ==="

    # Get all PVCs in namespace
    log_info "Discovering PVCs in namespace ${NAMESPACE}"

    local pvcs=$(kubectl get pvc -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${pvcs}" ]; then
        log_warn "No PVCs found in namespace ${NAMESPACE}"
        return 0
    fi

    local pvc_count=0
    local snapshot_count=0

    for pvc in ${pvcs}; do
        log_info "Processing PVC: ${pvc}"

        # Check if PVC is bound
        local phase=$(kubectl get pvc -n "${NAMESPACE}" "${pvc}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")

        if [ "${phase}" != "Bound" ]; then
            log_warn "PVC ${pvc} is not bound (phase: ${phase}), skipping"
            continue
        fi

        # Get PVC size for metadata
        local storage=$(kubectl get pvc -n "${NAMESPACE}" "${pvc}" -o jsonpath='{.spec.resources.requests.storage}' 2>/dev/null || echo "Unknown")

        # Create snapshot name
        local snapshot_name="${pvc}-snapshot-${TIMESTAMP}"

        # Check if snapshot class exists
        if ! kubectl get volumesnapshotclass "${SNAPSHOT_CLASS}" &>/dev/null; then
            log_error "VolumeSnapshotClass ${SNAPSHOT_CLASS} not found, cannot create snapshot"
            continue
        fi

        # Create VolumeSnapshot manifest
        cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: ${snapshot_name}
  namespace: ${NAMESPACE}
  labels:
    app: victor-ai
    backup-timestamp: "${TIMESTAMP}"
    backup-schedule: "hourly"
    source-pvc: "${pvc}"
  annotations:
    backup-size: "${storage}"
    backup-retention: "${RETENTION_DAYS}days"
spec:
  volumeSnapshotClassName: ${SNAPSHOT_CLASS}
  source:
    persistentVolumeClaimName: ${pvc}
EOF

        if [ $? -eq 0 ]; then
            log_info "Created VolumeSnapshot: ${snapshot_name} (PVC: ${pvc}, Size: ${storage})"
            snapshot_count=$((snapshot_count + 1))
        else
            log_error "Failed to create snapshot for PVC ${pvc}"
        fi

        pvc_count=$((pvc_count + 1))
    done

    log_info "Processed ${pvc_count} PVCs, created ${snapshot_count} snapshots"

    # Wait for snapshots to be ready
    if [ ${snapshot_count} -gt 0 ]; then
        log_info "Waiting for snapshots to become ready (timeout: 10 minutes)"

        local max_wait=600
        local waited=0

        while [ ${waited} -lt ${max_wait} ]; do
            local ready_count=0
            local total_count=0

            for pvc in ${pvcs}; do
                local snapshot_name="${pvc}-snapshot-${TIMESTAMP}"
                if kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot_name}" &>/dev/null; then
                    total_count=$((total_count + 1))
                    local ready=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot_name}" -o jsonpath='{.status.readyToUse}' 2>/dev/null || echo "false")

                    if [ "${ready}" = "true" ]; then
                        ready_count=$((ready_count + 1))
                    fi
                fi
            done

            log_info "Snapshot progress: ${ready_count}/${total_count} ready"

            if [ ${ready_count} -eq ${total_count} ] && [ ${total_count} -gt 0 ]; then
                log_info "All snapshots are ready"
                break
            fi

            sleep 30
            waited=$((waited + 30))
        done

        if [ ${waited} -ge ${max_wait} ]; then
            log_error "Timeout waiting for snapshots to become ready"
        fi
    fi

    log_info "=== Volume Snapshot Creation Completed ==="

    # Log metrics
    if command -v aws &>/dev/null; then
        aws cloudwatch put-metric-data \
            --namespace "VictorAI/Backups" \
            --metric-name "VolumeSnapshotsCreated" \
            --value "${snapshot_count}" \
            --unit Count 2>/dev/null || true
    fi
}

# ============================================================================
# LIST SNAPSHOTS FUNCTION
# ============================================================================

list_snapshots() {
    log_info "=== Listing Volume Snapshots ==="

    # Get all snapshots with Victor AI label
    local snapshots=$(kubectl get volumesnapshot -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${snapshots}" ]; then
        log_info "No snapshots found"
        return 0
    fi

    echo ""
    printf "%-50s %-20s %-10s %-15s\n" "SNAPSHOT NAME" "SOURCE PVC" "STATUS" "AGE"
    printf "%-50s %-20s %-10s %-15s\n" "--------------" "----------" "------" "---"

    for snapshot in ${snapshots}; do
        local source_pvc=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.metadata.labels.source-pvc}' 2>/dev/null || echo "N/A")
        local ready=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.status.readyToUse}' 2>/dev/null || echo "false")
        local creation_time=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "N/A")

        # Calculate age
        local age="N/A"
        if [ "${creation_time}" != "N/A" ]; then
            local creation_epoch=$(date -d "${creation_time}" +%s 2>/dev/null || date -jf "%Y-%m-%dT%H:%M:%SZ" "${creation_time}" +%s 2>/dev/null || echo "0")
            local current_epoch=$(date +%s)
            local age_seconds=$((current_epoch - creation_epoch))
            local age_days=$((age_seconds / 86400))
            local age_hours=$((age_seconds / 3600))

            if [ ${age_days} -gt 0 ]; then
                age="${age_days}d"
            elif [ ${age_hours} -gt 0 ]; then
                age="${age_hours}h"
            else
                age="$((age_seconds / 60))m"
            fi
        fi

        local status="NotReady"
        [ "${ready}" = "true" ] && status="Ready"

        printf "%-50s %-20s %-10s %-15s\n" "${snapshot}" "${source_pvc}" "${status}" "${age}"
    done

    echo ""

    # Count snapshots
    local snapshot_count=$(echo "${snapshots}" | wc -w)
    log_info "Total snapshots: ${snapshot_count}"
}

# ============================================================================
# CLEANUP OLD SNAPSHOTS FUNCTION
# ============================================================================

cleanup_snapshots() {
    log_info "=== Starting Cleanup of Old Snapshots ==="
    log_info "Retention: ${RETENTION_DAYS} days, or last ${RETENTION_COUNT} snapshots per PVC"

    # Get all snapshots with Victor AI label
    local snapshots=$(kubectl get volumesnapshot -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${snapshots}" ]; then
        log_info "No snapshots to clean up"
        return 0
    fi

    local deleted_count=0
    local kept_count=0
    local current_date=$(date +%s)

    # Group snapshots by PVC
    declare -A pvc_snapshots

    for snapshot in ${snapshots}; do
        local source_pvc=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.metadata.labels.source-pvc}' 2>/dev/null || echo "unknown")

        if [ "${source_pvc}" != "unknown" ]; then
            pvc_snapshots["${source_pvc}"]="${pvc_snapshots[${source_pvc}]} ${snapshot}"
        fi
    done

    # Clean up old snapshots for each PVC
    for pvc in "${!pvc_snapshots[@]}"; do
        log_info "Processing snapshots for PVC: ${pvc}"

        # Get all snapshots for this PVC, sorted by creation time (oldest first)
        local pvc_snapshot_list=$(for snap in ${pvc_snapshots[${pvc}]}; do
            local creation_time=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snap}" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "")
            local creation_epoch=$(date -d "${creation_time}" +%s 2>/dev/null || date -jf "%Y-%m-%dT%H:%M:%SZ" "${creation_time}" +%s 2>/dev/null || echo "0")
            echo "${creation_epoch} ${snap}"
        done | sort -n | awk '{print $2}')

        local snapshot_count=$(echo "${pvc_snapshot_list}" | wc -w)
        local delete_count=$((snapshot_count - RETENTION_COUNT))

        if [ ${delete_count} -le 0 ]; then
            log_info "  Keeping all ${snapshot_count} snapshots for ${pvc} (below retention count)"
            continue
        fi

        # Delete oldest snapshots
        local count=0
        for snapshot in ${pvc_snapshot_list}; do
            count=$((count + 1))

            # Check age
            local creation_time=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.metadata.creationTimestamp}' 2>/dev/null || echo "")
            local creation_epoch=$(date -d "${creation_time}" +%s 2>/dev/null || date -jf "%Y-%m-%dT%H:%M:%SZ" "${creation_time}" +%s 2>/dev/null || echo "0")
            local age_days=$(((current_date - creation_epoch) / 86400))

            # Delete if older than retention period or if we have too many snapshots
            if [ ${age_days} -gt ${RETENTION_DAYS} ] || [ ${count} -le ${delete_count} ]; then
                log_info "  Deleting snapshot: ${snapshot} (${age_days} days old, #${count} of ${snapshot_count})"

                kubectl delete volumesnapshot -n "${NAMESPACE}" "${snapshot}" --timeout=60s 2>/dev/null || {
                    log_warn "  Failed to delete snapshot ${snapshot}"
                    continue
                }

                deleted_count=$((deleted_count + 1))
            else
                kept_count=$((kept_count + 1))
            fi
        done
    done

    log_info "Cleanup completed: deleted ${deleted_count} snapshots, kept ${kept_count} snapshots"

    # Log metrics
    if command -v aws &>/dev/null; then
        aws cloudwatch put-metric-data \
            --namespace "VictorAI/Backups" \
            --metric-name "VolumeSnapshotsDeleted" \
            --value "${deleted_count}" \
            --unit Count 2>/dev/null || true
    fi
}

# ============================================================================
# VERIFY SNAPSHOTS FUNCTION
# ============================================================================

verify_snapshots() {
    log_info "=== Starting Snapshot Verification ==="

    # Get all snapshots with Victor AI label
    local snapshots=$(kubectl get volumesnapshot -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${snapshots}" ]; then
        log_warn "No snapshots found to verify"
        return 0
    fi

    local total_count=0
    local ready_count=0
    local error_count=0
    local errors=""

    for snapshot in ${snapshots}; do
        total_count=$((total_count + 1))

        local ready=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.status.readyToUse}' 2>/dev/null || echo "false")
        local error=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.status.error}' 2>/dev/null || echo "")

        if [ "${ready}" = "true" ]; then
            ready_count=$((ready_count + 1))
        else
            error_count=$((error_count + 1))
            errors="${errors}\n  - ${snapshot}: ${error}"
        fi
    done

    log_info "Verification Results:"
    log_info "  Total snapshots: ${total_count}"
    log_info "  Ready: ${ready_count}"
    log_info "  Errors: ${error_count}"

    if [ ${error_count} -gt 0 ]; then
        log_error "Snapshots with errors:${errors}"
        send_alert "CRITICAL" "${error_count} volume snapshots have errors"
        return 1
    else
        log_info "All snapshots are healthy"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Victor AI Volume Backup Script ==="
    log_info "Action: ${ACTION}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Timestamp: ${TIMESTAMP}"

    local exit_code=0

    case "${ACTION}" in
        create)
            create_snapshots || exit_code=$?
            ;;
        list)
            list_snapshots || exit_code=$?
            ;;
        cleanup)
            cleanup_snapshots || exit_code=$?
            ;;
        verify)
            verify_snapshots || exit_code=$?
            ;;
        *)
            log_error "Unknown action: ${ACTION}"
            log_error "Valid actions: create, list, cleanup, verify"
            exit 1
            ;;
    esac

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Volume Backup Script Completed Successfully ==="
    else
        log_error "=== Volume Backup Script Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Volume backup script failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
