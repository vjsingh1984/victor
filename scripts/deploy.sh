#!/bin/bash
#############################################################################
# Production Deployment Script for Victor AI
#
# Features:
# - Environment validation
# - Dependency installation
# - Configuration setup
# - Database migration
# - Health checks
# - Rollback support
#
# Usage: ./deploy.sh [staging|production]
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
DEPLOYMENT_ENV="${1:-staging}"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_FILE="${PROJECT_ROOT}/logs/deploy_$(date +%Y%m%d_%H%M%S).log"
VENV_DIR="${PROJECT_ROOT}/.venv"

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

# Error handler
error_exit() {
    log_error "$1"
    log_warning "Deployment failed! Check logs at ${LOG_FILE}"
    exit 1
}

# Trap errors
trap 'error_exit "Deployment failed at line $LINENO"' ERR

#############################################################################
# Environment Validation
#############################################################################
validate_environment() {
    log_info "Validating environment..."

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error_exit "Python 3 is not installed"
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    REQUIRED_VERSION="3.9"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        error_exit "Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    fi

    log_success "Python version validated: $PYTHON_VERSION"

    # Check required commands
    local required_commands=("git" "pip")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command '$cmd' is not available"
        fi
    done

    log_success "All required commands are available"
}

#############################################################################
# Configuration Validation
#############################################################################
validate_configuration() {
    log_info "Validating configuration..."

    # Load environment file if exists
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        source "${PROJECT_ROOT}/.env"
        log_success "Loaded environment variables from .env"
    else
        log_warning ".env file not found. Using default configuration."
    fi

    # Check required environment variables based on deployment environment
    local required_vars=()

    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        required_vars=(
            "VICTOR_PROFILE"
            "VICTOR_LOG_LEVEL"
        )
    fi

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_warning "Missing environment variables: ${missing_vars[*]}"
        log_info "These will use default values"
    else
        log_success "All required environment variables are set"
    fi
}

#############################################################################
# Backup Current Deployment
#############################################################################
backup_deployment() {
    log_info "Backing up current deployment..."

    if [ -d "${VENV_DIR}" ]; then
        BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
        BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

        mkdir -p "${BACKUP_PATH}"

        # Backup virtual environment
        if [ -d "${VENV_DIR}" ]; then
            cp -r "${VENV_DIR}" "${BACKUP_PATH}/venv" 2>/dev/null || true
        fi

        # Backup configuration
        if [ -f "${PROJECT_ROOT}/.env" ]; then
            cp "${PROJECT_ROOT}/.env" "${BACKUP_PATH}/"
        fi

        # Backup database if exists
        if [ -f "${PROJECT_ROOT}/victor.db" ]; then
            cp "${PROJECT_ROOT}/victor.db" "${BACKUP_PATH}/"
        fi

        # Store current git commit
        git rev-parse HEAD > "${BACKUP_PATH}/git_commit.txt" 2>/dev/null || true

        log_success "Backup created at ${BACKUP_PATH}"

        # Keep only last 5 backups
        ls -t "${BACKUP_DIR}" | tail -n +6 | xargs -I {} rm -rf "${BACKUP_DIR}/{}" 2>/dev/null || true
    else
        log_info "No existing deployment to backup"
    fi
}

#############################################################################
# Install Dependencies
#############################################################################
install_dependencies() {
    log_info "Installing/upgrading dependencies..."

    # Create or update virtual environment
    if [ ! -d "${VENV_DIR}" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    # Install dependencies
    log_info "Installing project dependencies..."
    cd "${PROJECT_ROOT}"

    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        pip install -e ".[api]" --no-deps
        pip install -e ".[api]"
    else
        pip install -e ".[dev]"
    fi

    log_success "Dependencies installed successfully"
}

#############################################################################
# Run Database Migrations
#############################################################################
run_migrations() {
    log_info "Running database migrations..."

    source "${VENV_DIR}/bin/activate"
    cd "${PROJECT_ROOT}"

    # Check if migrations directory exists
    if [ -d "${PROJECT_ROOT}/migrations" ]; then
        python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
try:
    from victor.core.migrations import run_migrations
    run_migrations()
    print('Migrations completed successfully')
except Exception as e:
    print(f'Migration error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE}"

        log_success "Database migrations completed"
    else
        log_info "No migrations to run"
    fi
}

#############################################################################
# Start Services
#############################################################################
start_services() {
    log_info "Starting services..."

    source "${VENV_DIR}/bin/activate"
    cd "${PROJECT_ROOT}"

    # Check if service manager is available
    if command -v systemctl &> /dev/null && [ "$DEPLOYMENT_ENV" = "production" ]; then
        # Using systemd
        sudo systemctl restart victor-api
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
# Health Checks
#############################################################################
run_health_checks() {
    log_info "Running health checks..."

    source "${VENV_DIR}/bin/activate"
    cd "${PROJECT_ROOT}"

    local max_attempts=30
    local attempt=1
    local health_url="http://localhost:8000/health"

    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt $attempt/$max_attempts..."

        if curl -f -s "${health_url}" > /dev/null 2>&1; then
            log_success "Service is responding"

            # Run comprehensive health check
            python -c "
import sys
import json
import urllib.request

try:
    response = urllib.request.urlopen('${health_url}', timeout=10)
    data = json.loads(response.read())

    print('Health Status:', data.get('status', 'unknown'))
    print('Version:', data.get('version', 'unknown'))

    # Check critical components
    if data.get('status') != 'healthy':
        sys.exit(1)

    # Check components if available
    components = data.get('components', {})
    for component, status in components.items():
        if status != 'healthy':
            print(f'Component {component} is not healthy: {status}', file=sys.stderr)
            sys.exit(1)

    print('All health checks passed')

except Exception as e:
    print(f'Health check failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE}"

            log_success "All health checks passed"
            return 0
        fi

        log_info "Service not ready yet, waiting..."
        sleep 2
        ((attempt++))
    done

    error_exit "Health checks failed after $max_attempts attempts"
}

#############################################################################
# Smoke Tests
#############################################################################
run_smoke_tests() {
    log_info "Running smoke tests..."

    source "${VENV_DIR}/bin/activate"
    cd "${PROJECT_ROOT}"

    # Run basic smoke tests
    python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

try:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.core.container import ServiceContainer

    # Test container initialization
    container = ServiceContainer()
    print('✓ ServiceContainer initialized')

    # Test basic imports
    from victor.tools.filesystem import ReadFileTool
    print('✓ Tools can be imported')

    # Test provider loading
    from victor.providers.registry import ProviderRegistry
    providers = ProviderRegistry.list_providers()
    print(f'✓ {len(providers)} providers loaded')

    print('\\nAll smoke tests passed')

except Exception as e:
    print(f'Smoke test failed: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a "${LOG_FILE}"

    log_success "Smoke tests completed"
}

#############################################################################
# Rollback
#############################################################################
rollback() {
    log_warning "Initiating rollback..."

    # Find latest backup
    LATEST_BACKUP=$(ls -t "${BACKUP_DIR}" | head -n 1)

    if [ -z "$LATEST_BACKUP" ]; then
        error_exit "No backup found for rollback"
    fi

    log_info "Rolling back to backup: ${LATEST_BACKUP}"

    BACKUP_PATH="${BACKUP_DIR}/${LATEST_BACKUP}"

    # Stop services
    if command -v systemctl &> /dev/null; then
        sudo systemctl stop victor-api || true
    fi

    # Restore virtual environment
    if [ -d "${BACKUP_PATH}/venv" ]; then
        rm -rf "${VENV_DIR}"
        cp -r "${BACKUP_PATH}/venv" "${VENV_DIR}"
        log_success "Virtual environment restored"
    fi

    # Restore configuration
    if [ -f "${BACKUP_PATH}/.env" ]; then
        cp "${BACKUP_PATH}/.env" "${PROJECT_ROOT}/.env"
        log_success "Configuration restored"
    fi

    # Restore database
    if [ -f "${BACKUP_PATH}/victor.db" ]; then
        cp "${BACKUP_PATH}/victor.db" "${PROJECT_ROOT}/victor.db"
        log_success "Database restored"
    fi

    # Restore git state
    if [ -f "${BACKUP_PATH}/git_commit.txt" ]; then
        GIT_COMMIT=$(cat "${BACKUP_PATH}/git_commit.txt")
        git checkout "$GIT_COMMIT" 2>/dev/null || true
        log_success "Git state restored to $GIT_COMMIT"
    fi

    # Restart services
    start_services

    log_success "Rollback completed"
}

#############################################################################
# Main Deployment Flow
#############################################################################
main() {
    log_info "Starting deployment to ${DEPLOYMENT_ENV}..."
    log_info "Deployment started at $(date)"

    # Validate environment
    validate_configuration

    # Backup current deployment
    backup_deployment

    # Install dependencies
    install_dependencies

    # Run migrations
    run_migrations

    # Start services
    start_services

    # Health checks
    if ! run_health_checks; then
        log_error "Health checks failed!"
        read -p "Do you want to rollback? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rollback
        fi
        exit 1
    fi

    # Smoke tests
    run_smoke_tests

    log_success "Deployment completed successfully!"
    log_info "Deployment finished at $(date)"

    # Print deployment info
    echo ""
    echo "=========================================="
    echo "Deployment Summary"
    echo "=========================================="
    echo "Environment: ${DEPLOYMENT_ENV}"
    echo "Timestamp: $(date)"
    echo "Virtual Environment: ${VENV_DIR}"
    echo "Logs: ${LOG_FILE}"
    echo "=========================================="
    echo ""
    echo "To manage the service:"
    if command -v systemctl &> /dev/null; then
        echo "  sudo systemctl status victor-api"
        echo "  sudo systemctl restart victor-api"
    else
        echo "  screen -r victor-api  # or tmux attach -t victor-api"
    fi
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        staging|production)
            DEPLOYMENT_ENV="$1"
            shift
            ;;
        --rollback)
            rollback
            exit 0
            ;;
        --help)
            echo "Usage: $0 [staging|production] [--rollback] [--help]"
            echo ""
            echo "Options:"
            echo "  staging       Deploy to staging environment (default)"
            echo "  production    Deploy to production environment"
            echo "  --rollback    Rollback to previous version"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main deployment
main
