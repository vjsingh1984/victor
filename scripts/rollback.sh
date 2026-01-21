#!/bin/bash
#############################################################################
# Rollback Script for Victor AI
#
# Features:
# - Version detection
# - Graceful shutdown
# - Previous version restoration
# - Data rollback if needed
#
# Usage: ./rollback.sh [--version VERSION] [--force] [--keep-db]
#############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/backups"
VENV_DIR="${PROJECT_ROOT}/.venv"
LOG_FILE="${PROJECT_ROOT}/logs/rollback_$(date +%Y%m%d_%H%M%S).log"

# Flags
TARGET_VERSION=""
FORCE=false
KEEP_DATABASE=false
DRY_RUN=false

# Ensure backup and logs directories exist
mkdir -p "${BACKUP_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}"
}

log_dry_run() {
    echo -e "${YELLOW}[DRY RUN]${NC} $1" | tee -a "${LOG_FILE}"
}

# Error handler
error_exit() {
    log_error "$1"
    log_warning "Rollback failed! Check logs at ${LOG_FILE}"
    exit 1
}

# Confirm action
confirm_action() {
    if [ "$FORCE" = true ]; then
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        return 0
    fi

    echo ""
    read -p "$(echo -e ${YELLOW}Are you sure you want to rollback? This will stop the current deployment. [y/N] ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
}

#############################################################################
# Parse Arguments
#############################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                TARGET_VERSION="$2"
                shift 2
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --keep-db)
                KEEP_DATABASE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --version VERSION    Rollback to specific version (backup name)"
                echo "  --force              Skip confirmation prompt"
                echo "  --keep-db            Keep current database (don't rollback)"
                echo "  --dry-run            Simulate rollback without making changes"
                echo "  --help               Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                              # Interactive rollback to latest backup"
                echo "  $0 --version backup_20240120    # Rollback to specific backup"
                echo "  $0 --force --keep-db            # Forced rollback, keep current database"
                echo "  $0 --dry-run                    # Simulate rollback"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

#############################################################################
# List Available Backups
#############################################################################
list_backups() {
    log_info "Available backups:"

    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]; then
        log_error "No backups found in ${BACKUP_DIR}"
        exit 1
    fi

    local index=1
    declare -A backup_map

    echo ""
    printf "%-5s %-30s %-15s %-15s\n" "No." "Backup Name" "Date" "Git Commit"
    printf "%-5s %-30s %-15s %-15s\n" "-----" "------------------------------" "---------------" "---------------"

    for backup in $(ls -t "$BACKUP_DIR"); do
        local backup_path="${BACKUP_DIR}/${backup}"
        local git_commit="unknown"
        local backup_date="unknown"

        if [ -f "${backup_path}/git_commit.txt" ]; then
            git_commit=$(cat "${backup_path}/git_commit.txt" 2>/dev/null || echo "unknown")
        fi

        # Extract date from backup name if possible
        if [[ "$backup" =~ backup_([0-9]{8})_ ]]; then
            backup_date=$(echo "$backup" | sed -n 's/backup_\([0-9]\{8\}\)_.*/\1/p')
            backup_date=$(date -j -f "%Y%m%d" "$backup_date" "+%Y-%m-%d" 2>/dev/null || echo "$backup_date")
        fi

        printf "%-5s %-30s %-15s %-15s\n" "$index" "$backup" "$backup_date" "$git_commit"
        backup_map[$index]="$backup"
        ((index++))
    done

    echo ""

    # If no specific version requested, prompt user
    if [ -z "$TARGET_VERSION" ]; then
        if [ "$FORCE" = false ]; then
            read -p "Select backup number to rollback to (or 'q' to cancel): " selection

            if [[ "$selection" == "q" ]] || [[ "$selection" == "Q" ]]; then
                log_info "Rollback cancelled by user"
                exit 0
            fi

            if [[ -z "${backup_map[$selection]:-}" ]]; then
                error_exit "Invalid selection: $selection"
            fi

            TARGET_VERSION="${backup_map[$selection]}"
        else
            # Default to latest backup
            TARGET_VERSION=$(ls -t "$BACKUP_DIR" | head -n 1)
            log_info "No version specified, using latest backup: ${TARGET_VERSION}"
        fi
    fi

    # Verify backup exists
    if [ ! -d "${BACKUP_DIR}/${TARGET_VERSION}" ]; then
        error_exit "Backup not found: ${TARGET_VERSION}"
    fi

    log_success "Selected backup: ${TARGET_VERSION}"
}

#############################################################################
# Stop Services
#############################################################################
stop_services() {
    log_info "Stopping services..."

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would stop services"
        return
    fi

    # Stop systemd service if exists
    if systemctl list-unit-files | grep -q victor-api; then
        sudo systemctl stop victor-api || log_warning "Failed to stop victor-api service"
        log_success "Stopped victor-api service"
    fi

    # Stop screen/tmux sessions
    if screen -list | grep -q "victor-api"; then
        screen -S victor-api -X quit || log_warning "Failed to stop screen session"
        log_success "Stopped screen session"
    fi

    # Stop tmux sessions
    if tmux list-sessions 2>/dev/null | grep -q "victor-api"; then
        tmux kill-session -t victor-api || log_warning "Failed to stop tmux session"
        log_success "Stopped tmux session"
    fi

    # Kill any remaining processes
    if pgrep -f "victor.api.server" > /dev/null; then
        pkill -f "victor.api.server" || log_warning "Failed to kill API processes"
        log_success "Stopped API processes"
    fi

    log_success "All services stopped"
}

#############################################################################
# Backup Current State
#############################################################################
backup_current_state() {
    log_info "Backing up current state before rollback..."

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would backup current state"
        return
    fi

    local pre_rollback_backup="pre_rollback_$(date +%Y%m%d_%H%M%S)"
    local backup_path="${BACKUP_DIR}/${pre_rollback_backup}"

    mkdir -p "$backup_path"

    # Backup current virtual environment
    if [ -d "${VENV_DIR}" ]; then
        cp -r "${VENV_DIR}" "${backup_path}/venv" 2>/dev/null || true
    fi

    # Backup current configuration
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        cp "${PROJECT_ROOT}/.env" "${backup_path}/"
    fi

    # Backup current database
    if [ -f "${PROJECT_ROOT}/victor.db" ]; then
        cp "${PROJECT_ROOT}/victor.db" "${backup_path}/"
    fi

    # Store current git commit
    git rev-parse HEAD > "${backup_path}/git_commit.txt" 2>/dev/null || true

    log_success "Current state backed up to: ${backup_path}"
}

#############################################################################
# Restore Virtual Environment
#############################################################################
restore_virtual_environment() {
    log_info "Restoring virtual environment..."

    local backup_path="${BACKUP_DIR}/${TARGET_VERSION}"

    if [ ! -d "${backup_path}/venv" ]; then
        log_warning "Virtual environment not found in backup. Skipping..."
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would restore virtual environment from ${backup_path}/venv"
        return
    fi

    # Remove current virtual environment
    if [ -d "${VENV_DIR}" ]; then
        rm -rf "${VENV_DIR}"
    fi

    # Restore backup
    cp -r "${backup_path}/venv" "${VENV_DIR}"
    log_success "Virtual environment restored"
}

#############################################################################
# Restore Configuration
#############################################################################
restore_configuration() {
    log_info "Restoring configuration..."

    local backup_path="${BACKUP_DIR}/${TARGET_VERSION}"

    if [ ! -f "${backup_path}/.env" ]; then
        log_warning "Configuration file not found in backup. Skipping..."
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would restore configuration from ${backup_path}/.env"
        return
    fi

    cp "${backup_path}/.env" "${PROJECT_ROOT}/.env"
    log_success "Configuration restored"
}

#############################################################################
# Restore Database
#############################################################################
restore_database() {
    if [ "$KEEP_DATABASE" = true ]; then
        log_info "Keeping current database (skipping rollback)"
        return
    fi

    log_info "Restoring database..."

    local backup_path="${BACKUP_DIR}/${TARGET_VERSION}"

    if [ ! -f "${backup_path}/victor.db" ]; then
        log_warning "Database not found in backup. Skipping..."
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would restore database from ${backup_path}/victor.db"
        return
    fi

    # Backup current database first
    if [ -f "${PROJECT_ROOT}/victor.db" ]; then
        cp "${PROJECT_ROOT}/victor.db" "${PROJECT_ROOT}/victor.db.pre_rollback"
        log_info "Current database backed up to victor.db.pre_rollback"
    fi

    cp "${backup_path}/victor.db" "${PROJECT_ROOT}/victor.db"
    log_success "Database restored"
}

#############################################################################
# Restore Git State
#############################################################################
restore_git_state() {
    log_info "Restoring Git state..."

    local backup_path="${BACKUP_DIR}/${TARGET_VERSION}"

    if [ ! -f "${backup_path}/git_commit.txt" ]; then
        log_warning "Git commit information not found in backup. Skipping..."
        return
    fi

    local git_commit
    git_commit=$(cat "${backup_path}/git_commit.txt" 2>/dev/null || echo "unknown")

    if [ "$git_commit" = "unknown" ]; then
        log_warning "Git commit unknown. Skipping..."
        return
    fi

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would restore Git state to commit ${git_commit}"
        return
    fi

    cd "${PROJECT_ROOT}"

    # Check if commit exists
    if ! git cat-file -e "$git_commit" 2>/dev/null; then
        log_warning "Git commit ${git_commit} not found in repository. Skipping..."
        return
    fi

    # Checkout commit
    git checkout "$git_commit"
    log_success "Git state restored to commit ${git_commit}"
}

#############################################################################
# Start Services
#############################################################################
start_services() {
    log_info "Starting services..."

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would start services"
        return
    fi

    source "${VENV_DIR}/bin/activate"
    cd "${PROJECT_ROOT}"

    # Start with systemd if available
    if systemctl list-unit-files | grep -q victor-api; then
        sudo systemctl start victor-api
        sudo systemctl enable victor-api
        log_success "Services started via systemd"
    else
        # Start with screen or tmux
        if command -v screen &> /dev/null; then
            screen -dmS victor-api bash -c "cd ${PROJECT_ROOT} && source ${VENV_DIR}/bin/activate && uvicorn victor.api.server:app --host 0.0.0.0 --port 8000"
            log_success "Service started in screen session 'victor-api'"
        elif command -v tmux &> /dev/null; then
            tmux new-session -d -s victor-api "cd ${PROJECT_ROOT} && source ${VENV_DIR}/bin/activate && uvicorn victor.api.server:app --host 0.0.0.0 --port 8000"
            log_success "Service started in tmux session 'victor-api'"
        else
            log_warning "No screen or tmux available. Start service manually:"
            log_info "  source ${VENV_DIR}/bin/activate"
            log_info "  uvicorn victor.api.server:app --host 0.0.0.0 --port 8000"
        fi
    fi
}

#############################################################################
# Verify Rollback
#############################################################################
verify_rollback() {
    log_info "Verifying rollback..."

    if [ "$DRY_RUN" = true ]; then
        log_dry_run "Would verify rollback"
        return
    fi

    # Run health checks
    local health_script="${SCRIPT_DIR}/health_check.sh"

    if [ -f "$health_script" ]; then
        if bash "$health_script" --no-exit-on-failure; then
            log_success "Health checks passed"
        else
            log_warning "Some health checks failed. Please verify the deployment manually."
        fi
    else
        log_warning "Health check script not found. Skipping verification."
    fi
}

#############################################################################
# Display Rollback Summary
#############################################################################
display_summary() {
    echo ""
    echo "=========================================="
    echo "Rollback Summary"
    echo "=========================================="
    echo "Target Version: ${TARGET_VERSION}"
    echo "Timestamp: $(date)"
    echo "Database Rollback: $([ "$KEEP_DATABASE" = true ] && echo 'No' || echo 'Yes')"
    echo "=========================================="
    echo ""

    log_success "Rollback completed successfully!"

    if [ "$DRY_RUN" = true ]; then
        log_info "This was a dry run. No changes were made."
        log_info "Run without --dry-run to perform actual rollback."
    else
        echo ""
        echo "To manage the service:"
        if systemctl list-unit-files | grep -q victor-api; then
            echo "  sudo systemctl status victor-api"
            echo "  sudo systemctl restart victor-api"
        else
            echo "  screen -r victor-api  # or tmux attach -t victor-api"
        fi
        echo ""
        echo "To check logs:"
        echo "  tail -f ${LOG_FILE}"
        echo ""
    fi
}

#############################################################################
# Main Rollback Flow
#############################################################################
main() {
    log_info "Starting rollback for Victor AI..."
    log_info "Rollback started at $(date)"

    # Parse arguments
    parse_args "$@"

    # List and select backup
    list_backups

    # Confirm action
    confirm_action

    # Stop services
    stop_services

    # Backup current state
    backup_current_state

    # Restore components
    restore_virtual_environment
    restore_configuration
    restore_database
    restore_git_state

    # Start services
    start_services

    # Verify rollback
    verify_rollback

    # Display summary
    display_summary

    log_info "Rollback finished at $(date)"
}

# Run main rollback
main "$@"
