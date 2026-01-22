#!/bin/bash
#############################################################################
# configure_backup.sh - Backup Configuration Helper
#
# Description:
#   Interactive script to configure backup settings before running setup
#
# Usage:
#   ./configure_backup.sh [options]
#
# Options:
#   --config <file>     Config file (default: backup-config.yaml)
#   --interactive       Interactive mode (default)
#   --generate-only     Only generate config, don't run setup
#   --dry-run           Show commands without executing
#
# Examples:
#   ./configure_backup.sh
#   ./configure_backup.sh --config my-backup-config.yaml
#   ./configure_backup.sh --generate-only
#############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
CONFIG_FILE="backup-config.yaml"
INTERACTIVE=true
GENERATE_ONLY=false
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --config <file>     Config file path (default: backup-config.yaml)
  --interactive       Interactive mode (default)
  --generate-only     Only generate config, don't run setup
  --dry-run           Show commands without executing
  -h, --help          Show this help message

Environment Variables:
  BUCKET_NAME         S3 bucket name
  AWS_REGION          AWS region
  BACKUP_SCHEDULE     Backup schedule (cron format)
  BACKUP_RETENTION    Backup retention in days

Examples:
  # Interactive mode
  $(basename "$0")

  # Generate config only
  $(basename "$0") --generate-only

  # Use existing config
  $(basename "$0") --config prod-backup-config.yaml

EOF
    exit 1
}

prompt_value() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"

    if [ -n "$default" ]; then
        prompt="$prompt [$default]"
    fi

    read -p "$prompt: " value

    if [ -z "$value" ] && [ -n "$default" ]; then
        value="$default"
    fi

    eval "$var_name='$value'"
}

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-n}"

    while true; do
        read -p "$prompt [y/n]: " answer
        answer=${answer:-$default}

        case "$answer" in
            [Yy]*)
                return 0
                ;;
            [Nn]*)
                return 1
                ;;
            *)
                echo "Please answer yes or no."
                ;;
        esac
    done
}

validate_bucket_name() {
    local bucket="$1"

    # Check if bucket name is valid
    if [[ ! "$bucket" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]]; then
        log_error "Invalid bucket name: $bucket"
        log_info "Bucket names must:"
        log_info "  - Be 3-63 characters long"
        log_info "  - Contain only lowercase letters, numbers, dots, and hyphens"
        log_info "  - Start and end with a letter or number"
        return 1
    fi

    return 0
}

validate_region() {
    local region="$1"

    # List of valid AWS regions (common ones)
    local valid_regions=(
        "us-east-1" "us-east-2" "us-west-1" "us-west-2"
        "eu-west-1" "eu-west-2" "eu-west-3" "eu-central-1"
        "ap-southeast-1" "ap-southeast-2" "ap-northeast-1" "ap-northeast-2"
        "ap-south-1" "ca-central-1" "sa-east-1"
    )

    for valid_region in "${valid_regions[@]}"; do
        if [ "$region" = "$valid_region" ]; then
            return 0
        fi
    done

    log_warning "Region $region might not be valid"
    return 0
}

validate_schedule() {
    local schedule="$1"

    # Basic cron format validation: 5 fields
    if [[ ! "$schedule" =~ ^[0-9,\-\*/]+\ [0-9,\-\*/]+\ [0-9,\-\*/]+\ [0-9,\-\*/]+\ [0-9,\-\*/]+$ ]]; then
        log_error "Invalid cron schedule: $schedule"
        log_info "Format: minute hour day month weekday"
        log_info "Example: 0 2 * * * (daily at 2 AM)"
        return 1
    fi

    return 0
}

generate_config() {
    local bucket="$1"
    local region="$2"
    local schedule="$3"
    local retention="$4"
    local output_file="$5"

    cat > "$output_file" << EOF
# Velero Backup Configuration
# Generated on: $(date)

s3:
  bucket: "$bucket"
  region: "$region"
  prefix: "velero"

schedule:
  daily: "$schedule"
  verification: "0 6 * * *"
  cleanup: "0 3 * * 0"
  test: "0 4 * * 0"

retention:
  default_ttl: "${retention}d"
  daily: 7
  weekly: 4
  monthly: 3

backup:
  all_namespaces: true
  include_cluster_resources: true
  default_volumes_to_fs_backup: true
  timeout: "14400s"

velero:
  namespace: "velero"
  version: "v1.13.0"

monitoring:
  enabled: true

cronjobs:
  backup:
    enabled: true
    schedule: "$schedule"
    suspend: false
  verification:
    enabled: true
    suspend: false
  cleanup:
    enabled: true
    retention_days: $retention
  test:
    enabled: true
    suspend: false

credentials:
  use_iam_role: true
EOF

    log_success "Configuration saved to: $output_file"
}

#############################################################################
# Main Script
#############################################################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --interactive)
                INTERACTIVE=true
                shift
                ;;
            --generate-only)
                GENERATE_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
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

    cat << "EOF"

========================================
Velero Backup Configuration
========================================

This script will help you configure Velero backups for your Victor application.

Press Ctrl+C at any time to cancel.

EOF

    # Check if config file already exists
    if [ -f "$CONFIG_FILE" ]; then
        log_warning "Config file already exists: $CONFIG_FILE"

        if prompt_yes_no "Do you want to overwrite it?" "n"; then
            log_info "Will overwrite existing config"
        else
            log_info "Using existing config: $CONFIG_FILE"
            INTERACTIVE=false
        fi
    fi

    # Collect configuration
    if [ "$INTERACTIVE" = true ]; then
        echo ""
        echo "=== S3 Configuration ==="
        echo ""

        # S3 bucket
        local bucket="${BUCKET_NAME:-}"
        while [ -z "$bucket" ]; do
            prompt_value "S3 bucket name for backups" "" bucket
            if ! validate_bucket_name "$bucket"; then
                bucket=""
            fi
        done

        # AWS region
        local region="${AWS_REGION:-us-east-1}"
        prompt_value "AWS region" "$region" region
        validate_region "$region"

        echo ""
        echo "=== Backup Schedule ==="
        echo ""

        # Backup schedule
        local schedule="${BACKUP_SCHEDULE:-0 2 * * *}"
        prompt_value "Backup schedule (cron format)" "$schedule" schedule
        while ! validate_schedule "$schedule"; do
            prompt_value "Backup schedule (cron format)" "0 2 * * *" schedule
        done

        # Retention period
        local retention="${BACKUP_RETENTION:-30}"
        prompt_value "Backup retention period (days)" "$retention" retention

        echo ""
        echo "=== Advanced Options ==="
        echo ""

        # Monitoring
        local monitoring_enabled="true"
        if ! prompt_yes_no "Enable monitoring and alerting?" "y"; then
            monitoring_enabled="false"
        fi

        # CronJobs
        local cronjobs_enabled="true"
        if ! prompt_yes_no "Enable automated CronJobs?" "y"; then
            cronjobs_enabled="false"
        fi

        # Generate config
        generate_config "$bucket" "$region" "$schedule" "$retention" "$CONFIG_FILE"

        # Update config with advanced options
        if [ "$monitoring_enabled" = "false" ]; then
            sed -i.bak 's/enabled: true/enabled: false/' "$CONFIG_FILE" 2>/dev/null || \
                sed -i.bak 's/enabled: true/enabled: false/' "$CONFIG_FILE"
        fi

        if [ "$cronjobs_enabled" = "false" ]; then
            sed -i.bak '/cronjobs:/,/^[^ ]/ s/enabled: true/enabled: false/' "$CONFIG_FILE" 2>/dev/null || true
        fi

        # Cleanup backup
        rm -f "${CONFIG_FILE}.bak"

    elif [ -f "$CONFIG_FILE" ]; then
        log_info "Using existing configuration: $CONFIG_FILE"

        # Extract values from config
        bucket=$(grep 'bucket:' "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')
        region=$(grep 'region:' "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')
        schedule=$(grep 'daily:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
        retention=$(grep 'retention_days:' "$CONFIG_FILE" | awk '{print $2}')

        # Use defaults if not found
        bucket="${bucket:-my-k8s-backups}"
        region="${region:-us-east-1}"
        schedule="${schedule:-0 2 * * *}"
        retention="${retention:-30}"

    else
        log_error "Config file not found: $CONFIG_FILE"
        log_info "Run with --interactive to create one"
        exit 1
    fi

    # Show summary
    cat << EOF

========================================
Configuration Summary
========================================

S3 Bucket:        ${bucket}
Region:           ${region}
Schedule:         ${schedule}
Retention:        ${retention} days
Config File:      ${CONFIG_FILE}

EOF

    # Generate setup command
    local setup_cmd="./setup_backups.sh --bucket ${bucket} --region ${region} --schedule '${schedule}' --retention ${retention}"

    if [ "$DRY_RUN" = true ]; then
        setup_cmd="${setup_cmd} --dry-run"
    fi

    # Run setup or just generate config
    if [ "$GENERATE_ONLY" = true ]; then
        log_success "Configuration file generated: $CONFIG_FILE"
        echo ""
        echo "To run setup:"
        echo "  cd ${SCRIPT_DIR}"
        echo "  ${setup_cmd}"
        exit 0
    fi

    if prompt_yes_no "Do you want to run setup now?" "y"; then
        log_info "Running setup..."

        # Change to script directory
        cd "$SCRIPT_DIR" || exit 1

        # Run setup
        if [ "$DRY_RUN" = true ]; then
            echo "$setup_cmd"
        else
            bash -c "$setup_cmd"
        fi

        log_success "Setup complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Verify installation: ./verify_backup_setup.sh"
        echo "  2. Check backups: velero backup get -n velero"
        echo "  3. Monitor logs: kubectl logs -n velero deployment/velero -f"
    else
        log_info "Setup skipped. You can run it later with:"
        echo "  cd ${SCRIPT_DIR}"
        echo "  ${setup_cmd}"
    fi
}

main "$@"
