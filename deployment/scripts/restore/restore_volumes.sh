#!/bin/bash
################################################################################
# Victor AI Volume Restore Script
# Purpose: Restore persistent volumes from snapshots
# Usage: ./restore_volumes.sh <pvc-name> <snapshot-name>
# Example: ./restore_volumes.sh victor-ai-data victor-ai-data-snapshot-20260120_143000
# Dependencies: kubectl
################################################################################

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

NAMESPACE="${VICTOR_NAMESPACE:-victor-ai-prod}"
LOG_FILE="/var/log/backups/restore_volumes.log"
ALERT_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PVC_NAME="${1:-}"
SNAPSHOT_NAME="${2:-}"

# Safety checks
AUTO_CONFIRM="${AUTO_CONFIRM:-false}"
CREATE_NEW_PVC="${CREATE_NEW_PVC:-true}"

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

        curl -X POST "${ALERT_WEBHOOK}" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"${emoji} Volume Restore ${severity}\", \"blocks\": [{\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": \"*${severity}*: ${message}\"}}]}" 2>/dev/null || true
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
# VALIDATION FUNCTIONS
# ============================================================================

validate_inputs() {
    if [ -z "${PVC_NAME}" ]; then
        error_exit "PVC name is required. Usage: $0 <pvc-name> <snapshot-name>"
    fi

    if [ -z "${SNAPSHOT_NAME}" ]; then
        error_exit "Snapshot name is required. Usage: $0 <pvc-name> <snapshot-name>"
    fi

    log_info "PVC: ${PVC_NAME}"
    log_info "Snapshot: ${SNAPSHOT_NAME}"
}

# ============================================================================
# SNAPSHOT RESTORE FUNCTION
# ============================================================================

restore_volume_from_snapshot() {
    log_info "=== Starting Volume Restore from Snapshot ==="

    # Check if snapshot exists
    if ! kubectl get volumesnapshot -n "${NAMESPACE}" "${SNAPSHOT_NAME}" &>/dev/null; then
        error_exit "Snapshot ${SNAPSHOT_NAME} not found in namespace ${NAMESPACE}"
    fi

    # Get snapshot details
    local snapshot_ready=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${SNAPSHOT_NAME}" -o jsonpath='{.status.readyToUse}' 2>/dev/null || echo "false")

    if [ "${snapshot_ready}" != "true" ]; then
        log_error "Snapshot ${SNAPSHOT_NAME} is not ready"
        error_exit "Cannot restore from unready snapshot"
    fi

    log_info "Snapshot ${SNAPSHOT_NAME} is ready for restore"

    # Check if PVC exists
    local pvc_exists=false
    if kubectl get pvc -n "${NAMESPACE}" "${PVC_NAME}" &>/dev/null; then
        pvc_exists=true
        log_info "PVC ${PVC_NAME} exists"

        # Get PVC details
        local pvc_phase=$(kubectl get pvc -n "${NAMESPACE}" "${PVC_NAME}" -o jsonpath='{.status.phase}')
        local pvc_bound=$(kubectl get pvc -n "${NAMESPACE}" "${PVC_NAME}" -o jsonpath='{.status.phase}')

        if [ "${pvc_phase}" = "Bound" ]; then
            log_warn "PVC ${PVC_NAME} is currently bound"

            if [ "${CREATE_NEW_PVC}" = "true" ]; then
                # Create new PVC from snapshot
                local new_pvc_name="${PVC_NAME}-restored-${TIMESTAMP}"
                log_info "Creating new PVC ${new_pvc_name} from snapshot"

                create_pvc_from_snapshot "${new_pvc_name}"
            else
                # Replace existing PVC
                confirm_action "Replace existing PVC ${PVC_NAME} with snapshot ${SNAPSHOT_NAME}?"
                replace_pvc_with_snapshot
            fi
        else
            log_warn "PVC ${PVC_NAME} is not bound, recreating"
            recreate_pvc_from_snapshot
        fi
    else
        log_info "PVC ${PVC_NAME} does not exist, creating new"
        create_pvc_from_snapshot "${PVC_NAME}"
    fi
}

# ============================================================================
# CREATE NEW PVC FROM SNAPSHOT
# ============================================================================

create_pvc_from_snapshot() {
    local new_pvc_name="$1"

    log_info "Creating PVC ${new_pvc_name} from snapshot ${SNAPSHOT_NAME}"

    # Get storage class and size from snapshot
    local snapshot_source_pvc=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${SNAPSHOT_NAME}" -o jsonpath='{.spec.source.persistentVolumeClaimName}')
    local storage_class=$(kubectl get pvc -n "${NAMESPACE}" "${snapshot_source_pvc}" -o jsonpath='{.spec.storageClassName}')
    local storage_request=$(kubectl get pvc -n "${NAMESPACE}" "${snapshot_source_pvc}" -o jsonpath='{.spec.resources.requests.storage}')

    if [ -z "${storage_class}" ]; then
        log_warn "Could not determine storage class, using default"
        storage_class="gp3"
    fi

    if [ -z "${storage_request}" ]; then
        log_warn "Could not determine storage size, using 10Gi"
        storage_request="10Gi"
    fi

    log_info "Storage class: ${storage_class}"
    log_info "Storage size: ${storage_request}"

    # Create PVC manifest
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${new_pvc_name}
  namespace: ${NAMESPACE}
  labels:
    app: victor-ai
    restored-from: "${SNAPSHOT_NAME}"
    restore-timestamp: "${TIMESTAMP}"
spec:
  storageClassName: ${storage_class}
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: ${storage_request}
  dataSource:
    name: ${SNAPSHOT_NAME}
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF

    if [ $? -eq 0 ]; then
        log_info "PVC ${new_pvc_name} created successfully"

        # Wait for PVC to be bound
        log_info "Waiting for PVC to be bound"
        kubectl wait --for=jsonpath='{.status.phase}'=Bound \
            pvc/${new_pvc_name} \
            -n "${NAMESPACE}" \
            --timeout=300s

        log_info "PVC ${new_pvc_name} is bound and ready"
    else
        error_exit "Failed to create PVC ${new_pvc_name}"
    fi
}

# ============================================================================
# REPLACE EXISTING PVC WITH SNAPSHOT
# ============================================================================

replace_pvc_with_snapshot() {
    log_info "Replacing PVC ${PVC_NAME} with snapshot ${SNAPSHOT_NAME}"

    # Get pods using this PVC
    local pods_using_pvc=$(kubectl get pods -n "${NAMESPACE}" -o json | \
        jq -r ".items[] | select(.spec.volumes[]?.persistentVolumeClaim.claimName == \"${PVC_NAME}\") | .metadata.name")

    if [ -n "${pods_using_pvc}" ]; then
        log_warn "The following pods are using PVC ${PVC_NAME}:"
        echo "${pods_using_pvc}"

        log_info "Pods must be deleted and recreated to use the restored PVC"
        confirm_action "Delete and recreate pods using ${PVC_NAME}?"
    fi

    # Backup PVC metadata
    local storage_class=$(kubectl get pvc -n "${NAMESPACE}" "${PVC_NAME}" -o jsonpath='{.spec.storageClassName}')
    local storage_request=$(kubectl get pvc -n "${NAMESPACE}" "${PVC_NAME}" -o jsonpath='{.spec.resources.requests.storage}')
    local access_modes=$(kubectl get pvc -n "${NAMESPACE}" "${PVC_NAME}" -o jsonpath='{.spec.accessModes}' | tr -d '[]')

    # Delete existing PVC
    log_info "Deleting existing PVC ${PVC_NAME}"
    kubectl delete pvc -n "${NAMESPACE}" "${PVC_NAME}" --timeout=60s

    # Create new PVC from snapshot
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: victor-ai
    restored-from: "${SNAPSHOT_NAME}"
    restore-timestamp: "${TIMESTAMP}"
spec:
  storageClassName: ${storage_class}
  accessModes:
$(echo "${access_modes}" | jq -R 'split(" ") | map({mode: .})')
  resources:
    requests:
      storage: ${storage_request}
  dataSource:
    name: ${SNAPSHOT_NAME}
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF

    if [ $? -eq 0 ]; then
        log_info "PVC ${PVC_NAME} recreated from snapshot"

        # Wait for PVC to be bound
        kubectl wait --for=jsonpath='{.status.phase}'=Bound \
            pvc/${PVC_NAME} \
            -n "${NAMESPACE}" \
            --timeout=300s

        log_info "PVC ${PVC_NAME} is bound and ready"

        # Restart pods that were using the PVC
        if [ -n "${pods_using_pvc}" ]; then
            log_info "Restarting pods that were using ${PVC_NAME}"

            for pod in ${pods_using_pvc}; do
                log_info "Deleting pod ${pod}"
                kubectl delete pod -n "${NAMESPACE}" "${pod}" --timeout=60s

                # Wait for pod to be recreated
                kubectl wait --for=condition=ready \
                    pod/${pod} \
                    -n "${NAMESPACE}" \
                    --timeout=300s || true
            done
        fi
    else
        error_exit "Failed to recreate PVC ${PVC_NAME}"
    fi
}

# ============================================================================
# RECREATE PVC FROM SNAPSHOT
# ============================================================================

recreate_pvc_from_snapshot() {
    log_info "Recreating PVC ${PVC_NAME} from snapshot ${SNAPSHOT_NAME}"

    # Delete existing PVC
    log_info "Deleting existing PVC ${PVC_NAME}"
    kubectl delete pvc -n "${NAMESPACE}" "${PVC_NAME}" --timeout=60s || true

    # Create new PVC from snapshot
    create_pvc_from_snapshot "${PVC_NAME}"
}

# ============================================================================
# LIST AVAILABLE SNAPSHOTS
# ============================================================================

list_snapshots() {
    log_info "=== Available Volume Snapshots ==="

    local snapshots=$(kubectl get volumesnapshot -n "${NAMESPACE}" -l app=victor-ai -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -z "${snapshots}" ]; then
        log_info "No snapshots found"
        return 0
    fi

    printf "%-50s %-20s %-10s %-15s\n" "SNAPSHOT NAME" "SOURCE PVC" "STATUS" "AGE"
    printf "%-50s %-20s %-10s %-15s\n" "--------------" "----------" "------" "---"

    for snapshot in ${snapshots}; do
        local source_pvc=$(kubectl get volumesnapshot -n "${NAMESPACE}" "${snapshot}" -o jsonpath='{.spec.source.persistentVolumeClaimName}' 2>/dev/null || echo "N/A")
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
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_info "=== Victor AI Volume Restore Script ==="
    log_info "Namespace: ${NAMESPACE}"
    log_info "Timestamp: ${TIMESTAMP}"

    # Special case: list snapshots
    if [ "${1:-}" = "list" ]; then
        list_snapshots
        exit 0
    fi

    # Validate inputs
    validate_inputs

    local overall_start_time=$(date +%s)
    local exit_code=0

    # Execute restore
    restore_volume_from_snapshot || exit_code=$?

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    if [ ${exit_code} -eq 0 ]; then
        log_info "=== Volume Restore Completed Successfully (${total_duration}s) ==="
        send_alert "INFO" "Volume restore completed successfully in ${total_duration}s"
    else
        log_error "=== Volume Restore Failed with exit code ${exit_code} ==="
        send_alert "CRITICAL" "Volume restore failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pvc-name> <snapshot-name>"
    echo "   or: $0 list"
    echo ""
    echo "Examples:"
    echo "  $0 victor-ai-data victor-ai-data-snapshot-20260120_143000"
    echo "  $0 list  # List available snapshots"
    echo ""
    echo "Environment Variables:"
    echo "  NAMESPACE        Kubernetes namespace (default: victor-ai-prod)"
    echo "  AUTO_CONFIRM     Skip confirmation prompts (default: false)"
    echo "  CREATE_NEW_PVC   Create new PVC instead of replacing (default: true)"
    exit 1
fi

# Run main function
main "$@"
