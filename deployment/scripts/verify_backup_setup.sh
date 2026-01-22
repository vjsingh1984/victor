#!/bin/bash
#############################################################################
# verify_backup_setup.sh - Verify Velero Backup Automation Setup
#
# Description:
#   Verifies that all backup components are correctly installed and configured
#
# Usage:
#   ./verify_backup_setup.sh [options]
#
# Options:
#   --namespace <ns>    Velero namespace (default: velero)
#   --verbose           Show detailed output
#   --fix               Attempt to fix common issues
#
# Examples:
#   ./verify_backup_setup.sh
#   ./verify_backup_setup.sh --verbose
#   ./verify_backup_setup.sh --namespace velero --fix
#############################################################################

set -euo pipefail

# Default values
NAMESPACE="${NAMESPACE:-velero}"
VERBOSE=false
FIX=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
PASSED=0
FAILED=0
WARNINGS=0

#############################################################################
# Helper Functions
#############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --namespace <ns>    Velero namespace (default: velero)
  --verbose           Show detailed output
  --fix               Attempt to fix common issues
  -h, --help          Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") --verbose
  $(basename "$0") --namespace velero --fix
EOF
    exit 1
}

#############################################################################
# Verification Functions
#############################################################################

check_velero_cli() {
    print_header "Checking Velero CLI"

    if command -v velero &> /dev/null; then
        local version=$(velero version --client-only 2>/dev/null | grep "Client" | awk '{print $2}')
        log_success "Velero CLI installed: $version"

        if [ "$VERBOSE" = true ]; then
            velero version --client-only
        fi
    else
        log_error "Velero CLI not installed"
        if [ "$FIX" = true ]; then
            log_info "Attempting to install Velero CLI..."
            brew install velero 2>/dev/null || log_error "Failed to install Velero CLI"
        fi
    fi
}

check_kubernetes_access() {
    print_header "Checking Kubernetes Access"

    if kubectl cluster-info &> /dev/null; then
        local cluster=$(kubectl config current-context)
        log_success "Connected to cluster: $cluster"
    else
        log_error "Cannot access Kubernetes cluster"
        return 1
    fi
}

check_velero_namespace() {
    print_header "Checking Velero Namespace"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_success "Namespace exists: $NAMESPACE"

        if [ "$VERBOSE" = true ]; then
            kubectl get namespace "$NAMESPACE"
        fi
    else
        log_error "Namespace not found: $NAMESPACE"
        if [ "$FIX" = true ]; then
            log_info "Creating namespace: $NAMESPACE"
            kubectl create namespace "$NAMESPACE"
        fi
    fi
}

check_velero_deployment() {
    print_header "Checking Velero Deployment"

    if kubectl get deployment -n "$NAMESPACE" velero &> /dev/null; then
        log_success "Velero deployment found"

        local ready=$(kubectl get deployment -n "$NAMESPACE" velero -o json | \
            jq -r '.status.readyReplicas // 0')
        local desired=$(kubectl get deployment -n "$NAMESPACE" velero -o json | \
            jq -r '.spec.replicas // 1')

        if [ "$ready" -eq "$desired" ]; then
            log_success "Velero pods ready: $ready/$desired"
        else
            log_error "Velero pods not ready: $ready/$desired"
        fi

        if [ "$VERBOSE" = true ]; then
            kubectl get deployment -n "$NAMESPACE" velero
            kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=velero
        fi
    else
        log_error "Velero deployment not found"
    fi
}

check_backup_storage_location() {
    print_header "Checking Backup Storage Locations"

    local locations=$(kubectl get backupstoragelocations -n "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.items | length' 2>/dev/null || echo "0")

    if [ "$locations" -gt 0 ]; then
        log_success "Found $locations backup storage location(s)"

        kubectl get backupstoragelocations -n "$NAMESPACE" -o json | \
            jq -r '.items[] | "\(.metadata.name): \(.status.phase // "Unknown"}"' | \
            while read -r line; do
                local name=$(echo "$line" | cut -d: -f1)
                local status=$(echo "$line" | cut -d: -f2)

                if [ "$status" = "Available" ]; then
                    log_success "Storage location $name: $status"
                else
                    log_error "Storage location $name: $status"
                fi
            done

        if [ "$VERBOSE" = true ]; then
            kubectl get backupstoragelocations -n "$NAMESPACE"
        fi
    else
        log_error "No backup storage locations found"
    fi
}

check_volume_snapshot_locations() {
    print_header "Checking Volume Snapshot Locations"

    local locations=$(kubectl get volumesnapshotlocations -n "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.items | length' 2>/dev/null || echo "0")

    if [ "$locations" -gt 0 ]; then
        log_success "Found $locations volume snapshot location(s)"

        kubectl get volumesnapshotlocations -n "$NAMESPACE" -o json | \
            jq -r '.items[] | "\(.metadata.name): \(.status.phase // "Unknown"}"' | \
            while read -r line; do
                local name=$(echo "$line" | cut -d: -f1)
                local status=$(echo "$line" | cut -d: -f2)

                if [ "$status" = "Available" ]; then
                    log_success "Snapshot location $name: $status"
                else
                    log_warning "Snapshot location $name: $status"
                fi
            done

        if [ "$VERBOSE" = true ]; then
            kubectl get volumesnapshotlocations -n "$NAMESPACE"
        fi
    else
        log_warning "No volume snapshot locations found (optional)"
    fi
}

check_schedules() {
    print_header "Checking Backup Schedules"

    local schedules=$(kubectl get schedules -n "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.items | length' 2>/dev/null || echo "0")

    if [ "$schedules" -gt 0 ]; then
        log_success "Found $chedules schedule(s)"

        kubectl get schedules -n "$NAMESPACE" | tail -n +2 | \
            while read -r name age schedule status; do
                if [ "$status" = "Enabled" ]; then
                    log_success "Schedule $name: $schedule ($status)"
                else
                    log_warning "Schedule $name: $schedule ($status)"
                fi
            done

        if [ "$VERBOSE" = true ]; then
            kubectl get schedules -n "$NAMESPACE"
        fi
    else
        log_warning "No backup schedules found"
    fi
}

check_cronjobs() {
    print_header "Checking CronJobs"

    local cronjobs=$(kubectl get cronjobs -n "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.items | length' 2>/dev/null || echo "0")

    if [ "$cronjobs" -gt 0 ]; then
        log_success "Found $cronjobs CronJob(s)"

        kubectl get cronjobs -n "$NAMESPACE" -o json | \
            jq -r '.items[] | "\(.metadata.name): \(.spec.schedule) (suspend: \(.spec.suspend)}"' | \
            while read -r line; do
                local name=$(echo "$line" | cut -d: -f1)
                local schedule=$(echo "$line" | cut -d: -f2 | cut -d' ' -f1-5)
                local suspend=$(echo "$line" | grep -o "suspend: [^)]*" | cut -d' ' -f2)

                if [ "$suspend" = "false" ]; then
                    log_success "CronJob $name: $schedule"
                else
                    log_warning "CronJob $name: $schedule (suspended)"
                fi
            done

        if [ "$VERBOSE" = true ]; then
            kubectl get cronjobs -n "$NAMESPACE"
        fi
    else
        log_warning "No CronJobs found"
    fi
}

check_recent_backups() {
    print_header "Checking Recent Backups"

    local backups=$(velero backup get -n "$NAMESPACE" -o json 2>/dev/null | \
        jq -r '.items | length' 2>/dev/null || echo "0")

    if [ "$backups" -gt 0 ]; then
        log_success "Found $backups backup(s)"

        # Check latest backup
        local latest=$(velero backup get -n "$NAMESPACE" -o json 2>/dev/null | \
            jq -r '.items | sort_by(.status.startTimestamp) | reverse | [0] | .[0].name')

        if [ -n "$latest" ] && [ "$latest" != "null" ]; then
            local status=$(velero backup get "$latest" -n "$NAMESPACE" -o json | \
                jq -r '.status.phase')

            local timestamp=$(velero backup get "$latest" -n "$NAMESPACE" -o json | \
                jq -r '.status.startTimestamp')

            if [ "$status" = "Completed" ]; then
                log_success "Latest backup: $latest ($status at $timestamp)"
            else
                log_error "Latest backup: $latest ($status at $timestamp)"
            fi
        fi

        # Check for failed backups
        local failed=$(velero backup get -n "$NAMESPACE" -o json 2>/dev/null | \
            jq -r '[.items[] | select(.status.phase != "Completed")] | length')

        if [ "$failed" -gt 0 ]; then
            log_warning "Found $failed failed backup(s)"
        fi

        if [ "$VERBOSE" = true ]; then
            velero backup get -n "$NAMESPACE"
        fi
    else
        log_warning "No backups found yet"
    fi
}

check_permissions() {
    print_header "Checking Velero Permissions"

    local sa_exists=$(kubectl get serviceaccount -n "$NAMESPACE" velero-server &> /dev/null && echo "true" || echo "false")

    if [ "$sa_exists" = "true" ]; then
        log_success "Velero service account exists"

        # Check cluster role binding
        local crb_exists=$(kubectl get clusterrolebinding velero-server &> /dev/null && echo "true" || echo "false")

        if [ "$crb_exists" = "true" ]; then
            log_success "Velero cluster role binding exists"
        else
            log_error "Velero cluster role binding not found"
        fi

        if [ "$VERBOSE" = true ]; then
            kubectl get serviceaccount -n "$NAMESPACE" velero-server
            kubectl get clusterrolebinding velero-server
        fi
    else
        log_error "Velero service account not found"
    fi
}

check_monitoring() {
    print_header "Checking Monitoring Setup"

    # Check for ServiceMonitor
    if kubectl get servicemonitor -n "$NAMESPACE" velero &> /dev/null; then
        log_success "ServiceMonitor found"

        if [ "$VERBOSE" = true ]; then
            kubectl get servicemonitor -n "$NAMESPACE" velero
        fi
    else
        log_warning "ServiceMonitor not found (optional)"
    fi

    # Check for PrometheusRule
    if kubectl get prometheusrule -n "$NAMESPACE" velero-alerts &> /dev/null; then
        log_success "PrometheusRule found"

        if [ "$VERBOSE" = true ]; then
            kubectl get prometheusrule -n "$NAMESPACE" velero-alerts
        fi
    else
        log_warning "PrometheusRule not found (optional)"
    fi

    # Check metrics endpoint
    if kubectl get svc -n "$NAMESPACE" velero &> /dev/null; then
        log_success "Velero service found (metrics available)"
    else
        log_error "Velero service not found"
    fi
}

check_credentials() {
    print_header "Checking Backup Credentials"

    # Check for secret if using access keys
    if kubectl get secret -n "$NAMESPACE" cloud-credentials &> /dev/null; then
        log_success "Cloud credentials secret found"
    else
        log_info "No cloud credentials secret found (using IAM role or instance profile)"
    fi

    # Check storage location configuration
    local location=$(kubectl get backupstoragelocation -n "$NAMESPACE" default -o json 2>/dev/null | \
        jq -r '.spec.provider // ""')

    if [ -n "$location" ]; then
        log_success "Storage provider: $location"
    fi
}

run_test_backup() {
    print_header "Running Test Backup"

    local test_backup="verify-test-$(date +%Y%m%d-%H%M%S)"

    log_info "Creating test backup: $test_backup"

    if velero backup create "$test_backup" \
        -n "$NAMESPACE" \
        --include-namespaces="$NAMESPACE" \
        --wait \
        --timeout=300s 2>/dev/null; then

        log_success "Test backup created successfully"

        # Describe backup
        if [ "$VERBOSE" = true ]; then
            velero backup describe "$test_backup" -n "$NAMESPACE" --details
        fi

        # Cleanup
        log_info "Cleaning up test backup..."
        velero backup delete "$test_backup" -n "$NAMESPACE" --confirm &>/dev/null || true
        log_success "Test backup deleted"

    else
        log_error "Test backup failed"
        return 1
    fi
}

print_summary() {
    print_header "Verification Summary"

    echo "Tests Passed: $PASSED"
    echo "Tests Failed: $FAILED"
    echo "Warnings: $WARNINGS"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All critical checks passed!${NC}"
        return 0
    else
        echo -e "${RED}Some checks failed. Please review the errors above.${NC}"
        return 1
    fi
}

#############################################################################
# Main Script
#############################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --fix)
                FIX=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done

    echo "========================================"
    echo "Velero Backup Setup Verification"
    echo "========================================"
    echo "Namespace: $NAMESPACE"
    echo "Verbose: $VERBOSE"
    echo "Auto-fix: $FIX"
    echo ""

    # Run verification checks
    check_velero_cli
    check_kubernetes_access
    check_velero_namespace
    check_velero_deployment
    check_backup_storage_location
    check_volume_snapshot_locations
    check_schedules
    check_cronjobs
    check_recent_backups
    check_permissions
    check_monitoring
    check_credentials

    # Optionally run test backup
    if [ $FAILED -eq 0 ]; then
        run_test_backup || true
    fi

    # Print summary
    print_summary
}

main "$@"
