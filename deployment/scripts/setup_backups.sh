#!/bin/bash
#############################################################################
# setup_backups.sh - Comprehensive Velero Backup Automation Setup
#
# Description:
#   Installs and configures Velero for Kubernetes backup automation with:
#   - Velero CLI and server installation
#   - S3 bucket configuration
#   - Automated daily backup CronJobs
#   - Backup verification
#   - Monitoring and alerting integration
#
# Usage:
#   ./setup_backups.sh [options]
#
# Options:
#   --bucket <name>        S3 bucket name for backups (required)
#   --region <region>      AWS region (default: us-east-1)
#   --schedule <cron>      Backup schedule (default: 0 2 * * *)
#   --retention <days>     Backup retention in days (default: 30)
#   --namespace <ns>       Velero namespace (default: velero)
#   --provider <provider>  Storage provider (default: aws)
#   --skip-monitoring      Skip monitoring setup
#   --dry-run              Show commands without executing
#   --uninstall            Remove Velero and all backups
#
# Environment Variables:
#   VELERO_VERSION         Velero version (default: v1.13.0)
#   AWS_ACCESS_KEY_ID      AWS access key (optional, uses IAM role if not set)
#   AWS_SECRET_ACCESS_KEY  AWS secret key (optional)
#   BUCKET_NAME            S3 bucket name
#
# Examples:
#   # Basic setup with IAM role
#   ./setup_backups.sh --bucket my-k8s-backups
#
#   # Custom schedule and retention
#   ./setup_backups.sh --bucket my-backups --schedule "0 3 * * *" --retention 60
#
#   # With AWS credentials
#   AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy \
#     ./setup_backups.sh --bucket my-backups --region us-west-2
#
#   # Dry run
#   ./setup_backups.sh --bucket my-backups --dry-run
#
#   # Uninstall
#   ./setup_backups.sh --bucket my-backups --uninstall
#############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
VELERO_VERSION="${VELERO_VERSION:-v1.13.0}"
NAMESPACE="${NAMESPACE:-velero}"
PROVIDER="${PROVIDER:-aws}"
REGION="${REGION:-us-east-1}"
SCHEDULE="${SCHEDULE:-0 2 * * *}"  # Daily at 2 AM
RETENTION="${RETENTION:-30}"
DRY_RUN=false
UNINSTALL=false
SKIP_MONITORING=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#############################################################################
# Helper Functions
#############################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    cat << "EOF"

╔═══════════════════════════════════════════════════════════════╗
║  Victor Backup Automation Setup with Velero                   ║
║  Comprehensive Kubernetes Backup & Restore Solution           ║
╚═══════════════════════════════════════════════════════════════╝

EOF
}

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --bucket <name>        S3 bucket name for backups (required)
  --region <region>      AWS region (default: us-east-1)
  --schedule <cron>      Backup schedule in cron format (default: 0 2 * * *)
  --retention <days>     Backup retention period in days (default: 30)
  --namespace <ns>       Velero namespace (default: velero)
  --provider <provider>  Storage provider (default: aws)
  --skip-monitoring      Skip monitoring and alerting setup
  --dry-run              Show commands without executing
  --uninstall            Remove Velero and all configurations
  -h, --help             Show this help message

Environment Variables:
  VELERO_VERSION         Velero version to install (default: v1.13.0)
  AWS_ACCESS_KEY_ID      AWS access key (optional)
  AWS_SECRET_ACCESS_KEY  AWS secret key (optional)
  BUCKET_NAME            S3 bucket name (alternative to --bucket)

Examples:
  # Basic setup
  $(basename "$0") --bucket my-k8s-backups

  # Custom configuration
  $(basename "$0") --bucket my-backups --region us-west-2 --retention 60

  # Dry run
  $(basename "$0") --bucket my-backups --dry-run

EOF
    exit 1
}

#############################################################################
# Validation Functions
#############################################################################

check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    # Check for required commands
    for cmd in kubectl jq; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Install missing dependencies:"
        echo "  brew install kubectl jq"
        exit 1
    fi

    log_success "All required dependencies are installed"
}

check_kubernetes_access() {
    log_info "Checking Kubernetes access..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        log_info "Please configure kubectl to access your cluster"
        exit 1
    fi

    local cluster_name=$(kubectl config current-context)
    log_success "Connected to cluster: $cluster_name"
}

check_bucket_access() {
    local bucket="$1"

    log_info "Checking S3 bucket access..."

    if [ -n "$AWS_ACCESS_KEY_ID" ]; then
        export AWS_ACCESS_KEY_ID
        export AWS_SECRET_ACCESS_KEY
    fi

    # Check if AWS CLI is available
    if command -v aws &> /dev/null; then
        if aws s3 ls "s3://${bucket}" --region "$REGION" &> /dev/null; then
            log_success "Bucket access verified: s3://${bucket}"
        else
            log_warning "Cannot access bucket, will proceed anyway"
            log_info "Ensure bucket exists or IAM permissions are correct"
        fi
    else
        log_warning "AWS CLI not found, skipping bucket verification"
    fi
}

#############################################################################
# Velero Installation
#############################################################################

install_velero_cli() {
    log_info "Checking Velero CLI installation..."

    if command -v velero &> /dev/null; then
        local version=$(velero version --client-only 2>/dev/null | grep "Client" | awk '{print $2}')
        log_success "Velero CLI already installed: $version"
        return 0
    fi

    log_info "Installing Velero CLI ${VELERO_VERSION}..."

    if [ "$DRY_RUN" = true ]; then
        echo "Would download and install Velero CLI ${VELERO_VERSION}"
        return 0
    fi

    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    # Convert architecture to Velero's naming
    if [ "$arch" = "x86_64" ]; then
        arch="amd64"
    elif [ "$arch" = "arm64" ]; then
        arch="arm64"
    fi

    local velero_url="https://github.com/vmware-tanzu/velero/releases/download/${VELERO_VERSION}/velero-${VELERO_VERSION}-${os}-${arch}.tar.gz"

    log_info "Downloading Velero from: $velero_url"

    # Download and extract
    local tmp_dir=$(mktemp -d)
    cd "$tmp_dir"

    if ! curl -fsSL -o velero.tar.gz "$velero_url"; then
        log_error "Failed to download Velero"
        rm -rf "$tmp_dir"
        exit 1
    fi

    tar -xzf velero.tar.gz
    sudo mv velero-${VELERO_VERSION}-${os}-${arch}/velero /usr/local/bin/

    rm -rf "$tmp_dir"
    cd - > /dev/null

    # Verify installation
    if command -v velero &> /dev/null; then
        local version=$(velero version --client-only 2>/dev/null | grep "Client" | awk '{print $2}')
        log_success "Velero CLI installed: $version"
    else
        log_error "Failed to install Velero CLI"
        exit 1
    fi
}

install_velero_server() {
    local bucket="$1"

    log_info "Installing Velero server in namespace: ${NAMESPACE}"

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: ${NAMESPACE}"
        if [ "$DRY_RUN" = false ]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi

    # Prepare credentials secret if using access keys
    local credentials_arg=""
    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        log_info "Creating AWS credentials secret..."

        if [ "$DRY_RUN" = false ]; then
            # Create credentials file
            cat > /tmp/velero-credentials << EOF
[default]
aws_access_key_id=${AWS_ACCESS_KEY_ID}
aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}
EOF

            # Create secret
            kubectl create secret generic \
                cloud-credentials \
                --namespace "$NAMESPACE" \
                --from-file=cloud=/tmp/velero-credentials \
                --dry-run=client -o yaml | kubectl apply -f -

            rm -f /tmp/velero-credentials
        fi

        credentials_arg="--secret-file -"
    fi

    # Install Velero
    local velero_install_cmd="velero install \
        --provider ${PROVIDER} \
        --plugins velero/velero-plugin-for-aws:${VELERO_VERSION} \
        --bucket ${bucket} \
        --prefix velero \
        --backup-location-config region=${REGION} \
        --snapshot-location-config region=${REGION} \
        --namespace ${NAMESPACE} \
        --wait"

    if [ -n "$credentials_arg" ]; then
        velero_install_cmd="${velero_install_cmd} ${credentials_arg}"
    fi

    log_info "Running: $velero_install_cmd"

    if [ "$DRY_RUN" = true ]; then
        echo "$velero_install_cmd"
    else
        if $velero_install_cmd; then
            log_success "Velero server installed successfully"
        else
            log_error "Failed to install Velero server"
            exit 1
        fi
    fi

    # Verify installation
    if [ "$DRY_RUN" = false ]; then
        log_info "Waiting for Velero pods to be ready..."
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=velero -n "$NAMESPACE" --timeout=300s
        log_success "Velero is ready"
    fi
}

#############################################################################
# Backup Configuration
#############################################################################

create_backup_schedule() {
    local bucket="$1"

    log_info "Creating backup schedule: ${SCHEDULE}"

    local schedule_name="daily-backup"
    local ttl="${RETENTION}d"

    # Create schedule
    local schedule_cmd="velero schedule create ${schedule_name} \
        --schedule '${SCHEDULE}' \
        --namespace ${NAMESPACE} \
        --ttl ${ttl} \
        --storage-location default \
        --include-cluster-resources=true \
        --default-volumes-to-fs-backup=true \
        --wait"

    log_info "Running: $schedule_cmd"

    if [ "$DRY_RUN" = true ]; then
        echo "$schedule_cmd"
    else
        if $schedule_cmd 2>/dev/null; then
            log_success "Backup schedule created: $schedule_name"
        else
            log_warning "Schedule might already exist, updating..."
            velero schedule delete "$schedule_name" --namespace "$NAMESPACE" --confirm 2>/dev/null || true
            $schedule_cmd
        fi
    fi
}

create_backup_retention_policy() {
    log_info "Configuring backup retention policy..."

    # Velero uses TTL for retention, which we already set in the schedule
    # Here we create a backup repository maintenance config
    cat > /tmp/velero-retention-policy.yaml << 'EOF'
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: default
  namespace: velero
spec:
  provider: aws
  objectStorage:
    bucket: ${BUCKET_NAME}
    prefix: velero
  config:
    region: ${REGION}
  backupSyncPeriod: 1m
  retentionPolicy:
    # Keep daily backups for 7 days
    - type: Daily
      days: 7
    # Keep weekly backups for 4 weeks
    - type: Weekly
      weeks: 4
    # Keep monthly backups for 3 months
    - type: Monthly
      months: 3
EOF

    if [ "$DRY_RUN" = false ]; then
        # Apply retention policy
        envsubst < /tmp/velero-retention-policy.yaml | kubectl apply -f -
        log_success "Retention policy configured"
    else
        echo "Would apply retention policy"
    fi

    rm -f /tmp/velero-retention-policy.yaml
}

#############################################################################
# CronJob Creation
#############################################################################

create_backup_cronjob() {
    log_info "Creating Kubernetes CronJob for daily backups..."

    cat > /tmp/backup-cronjob.yaml << EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: velero-daily-backup
  namespace: ${NAMESPACE}
  labels:
    app: velero-backup
    component: backup
spec:
  schedule: "${SCHEDULE}"
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  concurrencyPolicy: Forbid
  jobTemplate:
    metadata:
      labels:
        app: velero-backup
        component: backup
    spec:
      template:
        metadata:
          labels:
            app: velero-backup
            component: backup
        spec:
          serviceAccountName: velero-server
          restartPolicy: OnFailure
          containers:
          - name: velero-cli
            image: velero/velero:${VELERO_VERSION}
            command:
            - /velero
            args:
            - backup
            - create
            - victor-backup-$(date +%Y%m%d-%H%M%S)
            - --namespace=${NAMESPACE}
            - --selector=app.kubernetes.io/instance=victor
            - --include-cluster-resources=true
            - --default-volumes-to-fs-backup=true
            - --wait=true
            env:
            - name: VELERO_NAMESPACE
              value: ${NAMESPACE}
            resources:
              requests:
                cpu: 100m
                memory: 128Mi
              limits:
                cpu: 500m
                memory: 512Mi
          backoffLimit: 3
EOF

    if [ "$DRY_RUN" = false ]; then
        kubectl apply -f /tmp/backup-cronjob.yaml
        log_success "Backup CronJob created"
    else
        echo "Would create backup CronJob"
    fi

    rm -f /tmp/backup-cronjob.yaml
}

create_verification_cronjob() {
    log_info "Creating backup verification CronJob..."

    cat > /tmp/verification-cronjob.yaml << EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: velero-verification
  namespace: ${NAMESPACE}
  labels:
    app: velero-backup
    component: verification
spec:
  schedule: "0 6 * * *"  # Daily at 6 AM (after backup)
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  concurrencyPolicy: Forbid
  jobTemplate:
    metadata:
      labels:
        app: velero-backup
        component: verification
    spec:
      template:
        metadata:
          labels:
            app: velero-backup
            component: verification
        spec:
          serviceAccountName: velero-server
          restartPolicy: OnFailure
          containers:
          - name: velero-verify
            image: velero/velero:${VELERO_VERSION}
            command:
            - /bin/bash
            args:
            - -c
            - |
              set -e
              echo "Starting backup verification..."

              # Get latest backup
              LATEST_BACKUP=\$(velero backup get -o json | jq -r '.items | sort_by(.status.startTimestamp) | reverse | [.[0].name] | .[0]')

              if [ -z "\$LATEST_BACKUP" ]; then
                echo "No backups found!"
                exit 1
              fi

              echo "Checking backup: \$LATEST_BACKUP"

              # Get backup details
              STATUS=\$(velero backup get \$LATEST_BACKUP -o json | jq -r '.status.phase')

              if [ "\$STATUS" != "Completed" ]; then
                echo "Backup failed with status: \$STATUS"
                exit 1
              fi

              # Check for errors
              ERRORS=\$(velero backup get \$LATEST_BACKUP -o json | jq -r '.status.errors')
              if [ "\$ERRORS" != "null" ] && [ "\$ERRORORS" != "0" ]; then
                echo "Backup completed with errors: \$ERRORS"
                exit 1
              fi

              echo "Backup verification successful: \$LATEST_BACKUP"

              # List backup statistics
              velero backup describe \$LATEST_BACKUP --details
            env:
            - name: VELERO_NAMESPACE
              value: ${NAMESPACE}
            resources:
              requests:
                cpu: 100m
                memory: 128Mi
              limits:
                cpu: 500m
                memory: 512Mi
          backoffLimit: 2
EOF

    if [ "$DRY_RUN" = false ]; then
        kubectl apply -f /tmp/verification-cronjob.yaml
        log_success "Verification CronJob created"
    else
        echo "Would create verification CronJob"
    fi

    rm -f /tmp/verification-cronjob.yaml
}

#############################################################################
# Monitoring and Alerting
#############################################################################

setup_monitoring() {
    if [ "$SKIP_MONITORING" = true ]; then
        log_warning "Skipping monitoring setup"
        return 0
    fi

    log_info "Setting up monitoring and alerting..."

    # Check if Prometheus is installed
    if ! kubectl get namespace monitoring &> /dev/null; then
        log_warning "Monitoring namespace not found, skipping Prometheus setup"
        return 0
    fi

    # Create ServiceMonitor for Velero
    cat > /tmp/velero-servicemonitor.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: velero
  namespace: ${NAMESPACE}
  labels:
    app: velero
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: velero
  endpoints:
  - port: metrics
    interval: 30s
    scheme: http
EOF

    if [ "$DRY_RUN" = false ]; then
        kubectl apply -f /tmp/velero-servicemonitor.yaml 2>/dev/null || \
            log_warning "Prometheus operator not found, skipping ServiceMonitor"
    fi

    rm -f /tmp/velero-servicemonitor.yaml

    # Create Prometheus rules
    cat > /tmp/velero-prometheus-rules.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: velero-alerts
  namespace: ${NAMESPACE}
  labels:
    app: velero
    prometheus: kube-prometheus
spec:
  groups:
  - name: velero.rules
    interval: 30s
    rules:
    - alert: VeleroBackupFailed
      expr: velero_backup_last_status{status="Failed"} == 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Velero backup has failed"
        description: "Backup {{ $labels.schedule }} has failed for namespace {{ \$labels.namespace }}"

    - alert: VeleroBackupOlderThan24h
      expr: time() - velero_backup_last_success_timestamp_seconds > 86400
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Velero backup is older than 24 hours"
        description: "Last successful backup was {{ humanizeDuration \$value }} ago"

    - alert: VeleroRestoreFailed
      expr: velero_restore_last_status{status="Failed"} == 1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Velero restore has failed"
        description: "Restore {{ $labels.restore }} has failed"

    - alert: VeleroBackupStorageLocationNotReady
      expr: velero_backup_storage_location_ready == 0
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Velero backup storage location not ready"
        description: "Backup storage location {{ $labels.name }} is not ready"

    - alert: VeleroVolumeSnapshotNotReady
      expr: velero_volume_snapshot_ready == 0
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Velero volume snapshot not ready"
        description: "Volume snapshot location {{ $labels.name }} is not ready"
EOF

    if [ "$DRY_RUN" = false ]; then
        kubectl apply -f /tmp/velero-prometheus-rules.yaml 2>/dev/null || \
            log_warning "Prometheus operator not found, skipping alert rules"
        log_success "Monitoring rules configured"
    else
        echo "Would configure monitoring rules"
    fi

    rm -f /tmp/velero-prometheus-rules.yaml
}

#############################################################################
# Testing
#############################################################################

test_backup() {
    log_info "Creating test backup..."

    local test_backup_name="test-backup-$(date +%Y%m%d-%H%M%S)"

    if [ "$DRY_RUN" = true ]; then
        echo "Would create test backup: $test_backup_name"
        return 0
    fi

    # Create test backup
    if velero backup create "$test_backup_name" \
        --namespace "$NAMESPACE" \
        --wait \
        --include-cluster-resources=false \
        --selector="app.kubernetes.io/name=victor"; then

        log_success "Test backup created successfully: $test_backup_name"

        # Describe backup
        velero backup describe "$test_backup_name" --namespace "$NAMESPACE"

        # Get backup stats
        log_info "Backup statistics:"
        velero backup get "$test_backup_name" -n "$NAMESPACE"

        return 0
    else
        log_error "Test backup failed"
        return 1
    fi
}

test_restore() {
    local backup_name="$1"

    log_info "Creating test restore from backup: $backup_name"

    local test_restore_name="test-restore-$(date +%Y%m%d-%H%M%S)"

    if [ "$DRY_RUN" = true ]; then
        echo "Would create test restore: $test_restore_name"
        return 0
    fi

    # Create test restore (dry-run to verify)
    if velero restore create "$test_restore_name" \
        --namespace "$NAMESPACE" \
        --from-backup "$backup_name" \
        --dry-run \
        --wait; then

        log_success "Test restore (dry-run) completed successfully"

        # Describe restore
        velero restore describe "$test_restore_name" --namespace "$NAMESPACE"

        # Delete test restore
        velero restore delete "$test_restore_name" --namespace "$NAMESPACE" --confirm

        return 0
    else
        log_error "Test restore failed"
        return 1
    fi
}

cleanup_test_backup() {
    local backup_name="$1"

    log_info "Cleaning up test backup: $backup_name"

    if [ "$DRY_RUN" = true ]; then
        echo "Would delete test backup: $backup_name"
        return 0
    fi

    velero backup delete "$backup_name" --namespace "$NAMESPACE" --confirm
    log_success "Test backup deleted"
}

#############################################################################
# Uninstall
#############################################################################

uninstall_velero() {
    log_warning "Uninstalling Velero and all backups..."

    if [ "$DRY_RUN" = true ]; then
        echo "Would uninstall Velero"
        return 0
    fi

    # Delete CronJobs
    kubectl delete cronjob -n "$NAMESPACE" -l app=velero-backup --ignore-not-found=true

    # Delete all backups
    log_info "Deleting all backups..."
    velero backup delete --all --namespace "$NAMESPACE" --confirm || true

    # Uninstall Velero
    log_info "Removing Velero server..."
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true --wait=false

    # Delete CRDs
    log_info "Deleting Velero CRDs..."
    kubectl delete crd \
        backups.velero.io \
        backupstoragelocations.velero.io \
        deletebackuprequests.velero.io \
        downloadrequests.velero.io \
        podvolumebackups.velero.io \
        podvolumerestores.velero.io \
        resticrepositories.velero.io \
        restores.velero.io \
        schedules.velero.io \
        serverstatusrequests.velero.io \
        volumesnapshotlocations.velero.io \
        --ignore-not-found=true || true

    log_success "Velero uninstalled"
}

#############################################################################
# Main Script
#############################################################################

main() {
    print_banner

    # Parse arguments
    local bucket=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --bucket)
                bucket="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --schedule)
                SCHEDULE="$2"
                shift 2
                ;;
            --retention)
                RETENTION="$2"
                shift 2
                ;;
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --provider)
                PROVIDER="$2"
                shift 2
                ;;
            --skip-monitoring)
                SKIP_MONITORING=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --uninstall)
                UNINSTALL=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # Check for bucket from environment
    if [ -z "$bucket" ]; then
        bucket="${BUCKET_NAME:-}"
    fi

    # Handle uninstall
    if [ "$UNINSTALL" = true ]; then
        uninstall_velero
        exit 0
    fi

    # Validate required parameters
    if [ -z "$bucket" ]; then
        log_error "Bucket name is required (use --bucket or BUCKET_NAME)"
        usage
    fi

    # Show configuration
    cat << EOF

Configuration:
  Bucket:       ${bucket}
  Region:       ${REGION}
  Schedule:     ${SCHEDULE}
  Retention:    ${RETENTION} days
  Namespace:    ${NAMESPACE}
  Provider:     ${PROVIDER}
  Velero:       ${VELERO_VERSION}
  Dry Run:      ${DRY_RUN}

EOF

    # Pre-flight checks
    check_dependencies
    check_kubernetes_access
    check_bucket_access "$bucket"

    # Installation
    install_velero_cli
    install_velero_server "$bucket"
    create_backup_schedule "$bucket"
    create_backup_retention_policy

    # CronJobs
    create_backup_cronjob
    create_verification_cronjob

    # Monitoring
    setup_monitoring

    # Testing
    log_info "Running backup tests..."
    test_backup_name="test-backup-$(date +%Y%m%d-%H%M%S)"
    if test_backup; then
        test_restore "$test_backup_name"
        cleanup_test_backup "$test_backup_name"
    fi

    # Summary
    cat << EOF

========================================
Backup Setup Complete!
========================================

Velero Installation:
  - CLI: $(command -v velero)
  - Server: Namespace ${NAMESPACE}
  - Version: ${VELERO_VERSION}

Backup Configuration:
  - Schedule: ${SCHEDULE}
  - Retention: ${RETENTION} days
  - Storage: s3://${bucket}

Next Steps:
  1. Verify backups:
     kubectl get schedules -n ${NAMESPACE}
     velero schedule get -n ${NAMESPACE}

  2. Monitor backups:
     kubectl logs -n ${NAMESPACE} deployment/velero -f
     velero backup get -n ${NAMESPACE}

  3. Manual backup:
     velero backup create my-backup -n ${NAMESPACE} --wait

  4. Manual restore:
     velero restore create my-restore --from-backup my-backup -n ${NAMESPACE}

  5. View backup details:
     velero backup describe <backup-name> -n ${NAMESPACE} --details

Monitoring:
  - Prometheus ServiceMonitor: velero
  - Alert rules: velero-alerts
  - Metrics: http://velero.${NAMESPACE}:8085/metrics

Backup Locations:
  kubectl get backupstoragelocations -n ${NAMESPACE}

Schedules:
  kubectl get schedules -n ${NAMESPACE}
  velero schedule get -n ${NAMESPACE}

Useful Commands:
  # List backups
  velero backup get -n ${NAMESPACE}

  # Describe backup
  velero backup describe <name> -n ${NAMESPACE} --details

  # Create on-demand backup
  velero backup create urgent-backup -n ${NAMESPACE} --wait

  # Restore from backup
  velero restore create restore-1 --from-backup <backup-name> -n ${NAMESPACE}

  # Check backup status
  kubectl get pod -n ${NAMESPACE} -l app.kubernetes.io/name=velero

  # View Velero logs
  kubectl logs -n ${NAMESPACE} deployment/velero -f

For more information:
  https://velero.io/docs/
  https://github.com/vmware-tanzu/velero

EOF
}

# Run main function
main "$@"
