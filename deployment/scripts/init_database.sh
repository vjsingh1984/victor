#!/bin/bash
################################################################################
# Database Initialization Script
#
# This script initializes the PostgreSQL database with required schemas,
# extensions, and initial data for Victor AI.
#
# Usage:
#   ./init_database.sh [--namespace NAMESPACE] [--migrations-path PATH]
#                     [--skip-extensions] [--dry-run]
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MIGRATIONS_PATH="${PROJECT_ROOT}/migrations"

# Default values
NAMESPACE="victor"
SKIP_EXTENSIONS=false
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Initialize PostgreSQL database with schemas and extensions.

Options:
  --namespace NAMESPACE      Kubernetes namespace (default: victor)
  --migrations-path PATH     Path to migration files
  --skip-extensions          Skip installing PostgreSQL extensions
  --dry-run                  Show what would be done without executing
  --help                     Show this help message

Examples:
  $0
  $0 --namespace production
  $0 --migrations-path ./sql/migrations
  $0 --dry-run
EOF
}

################################################################################
# Argument Parsing
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --migrations-path)
                MIGRATIONS_PATH="$2"
                shift 2
                ;;
            --skip-extensions)
                SKIP_EXTENSIONS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

################################################################################
# Database Connection
################################################################################

get_postgres_pod() {
    kubectl get pod -n "${NAMESPACE}" -l app=postgres \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo ""
}

get_db_password() {
    kubectl get secret postgres-secret -n "${NAMESPACE}" \
        -o jsonpath='{.data.POSTGRES_PASSWORD}' 2>/dev/null | base64 -d || echo ""
}

execute_sql() {
    local sql=$1

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would execute SQL:"
        echo "${sql}"
        return 0
    fi

    local pod
    pod=$(get_postgres_pod)

    if [[ -z "${pod}" ]]; then
        log_error "PostgreSQL pod not found"
        return 1
    fi

    kubectl exec -n "${NAMESPACE}" "${pod}" -- psql -U victor -d victor -c "${sql}"
}

################################################################################
# Extensions
################################################################################

install_extensions() {
    log_info "Installing PostgreSQL extensions..."

    if [[ "${SKIP_EXTENSIONS}" == "true" ]]; then
        log_warning "Skipping extension installation"
        return 0
    fi

    # Required extensions for Victor AI
    local extensions=(
        "uuid-ossp"        # UUID generation
        "pg_stat_statements"  # Query statistics
        "pg_trgm"          # Trigram matching for full-text search
        "btree_gin"        # GIN indexes for B-Tree
        "btree_gist"       # GiST indexes for B-Tree
    )

    for ext in "${extensions[@]}"; do
        log_info "Installing extension: ${ext}"
        execute_sql "CREATE EXTENSION IF NOT EXISTS \"${ext}\" CASCADE;" || true
    done

    log_success "Extensions installed"
}

################################################################################
# Schemas
################################################################################

create_schemas() {
    log_info "Creating database schemas..."

    # Create schemas for different components
    local schemas=(
        "event_store"      # Event sourcing
        "workflows"        # Workflow state
        "checkpoints"      # Checkpoint data
        "cache"            # Cache tables
        "sessions"         # Session management
    )

    for schema in "${schemas[@]}"; do
        log_info "Creating schema: ${schema}"
        execute_sql "CREATE SCHEMA IF NOT EXISTS ${schema};" || true
        execute_sql "GRANT ALL PRIVILEGES ON SCHEMA ${schema} TO victor;" || true
    done

    log_success "Schemas created"
}

################################################################################
# Tables
################################################################################

create_tables() {
    log_info "Creating database tables..."

    # Event store table
    execute_sql "
CREATE TABLE IF NOT EXISTS event_store.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    aggregate_type VARCHAR(255) NOT NULL,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(255) NOT NULL,
    event_data JSONB NOT NULL,
    version BIGINT NOT NULL,
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    INDEX (aggregate_type, aggregate_id, version)
);
" || true

    # Workflow state table
    execute_sql "
CREATE TABLE IF NOT EXISTS workflows.state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) UNIQUE NOT NULL,
    state JSONB NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
" || true

    # Checkpoint table
    execute_sql "
CREATE TABLE IF NOT EXISTS checkpoints.checkpoint (
    thread_id VARCHAR(255) PRIMARY KEY,
    checkpoint JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
" || true

    # Session table
    execute_sql "
CREATE TABLE IF NOT EXISTS sessions.session (
    id VARCHAR(255) PRIMARY KEY,
    user_id UUID,
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);
" || true

    log_success "Tables created"
}

################################################################################
# Indexes
################################################################################

create_indexes() {
    log_info "Creating indexes..."

    # Event store indexes
    execute_sql "CREATE INDEX IF NOT EXISTS idx_events_aggregate ON event_store.events(aggregate_type, aggregate_id);" || true
    execute_sql "CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON event_store.events(occurred_at DESC);" || true

    # Workflow state indexes
    execute_sql "CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflows.state(status);" || true
    execute_sql "CREATE INDEX IF NOT EXISTS idx_workflow_updated ON workflows.state(updated_at DESC);" || true

    # Session indexes
    execute_sql "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions.session(user_id);" || true
    execute_sql "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions.session(expires_at);" || true

    log_success "Indexes created"
}

################################################################################
# Migrations
################################################################################

run_migrations() {
    log_info "Running database migrations..."

    if [[ ! -d "${MIGRATIONS_PATH}" ]]; then
        log_warning "Migrations directory not found: ${MIGRATIONS_PATH}"
        log_info "Skipping migrations"
        return 0
    fi

    # Find migration files
    local migrations
    migrations=$(find "${MIGRATIONS_PATH}" -name "*.sql" -type f | sort)

    if [[ -z "${migrations}" ]]; then
        log_info "No migration files found"
        return 0
    fi

    # Create migrations tracking table
    execute_sql "
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
" || true

    # Run each migration
    while IFS= read -r migration_file; do
        local version
        version=$(basename "${migration_file}" .sql)

        log_info "Applying migration: ${version}"

        # Check if migration already applied
        local applied
        applied=$(kubectl exec -n "${NAMESPACE}" "$(get_postgres_pod)" -- psql -U victor -d victor -tAc \
            "SELECT COUNT(*) FROM schema_migrations WHERE version = '${version}'" 2>/dev/null || echo "0")

        if [[ "${applied}" -gt 0 ]]; then
            log_info "Migration ${version} already applied, skipping"
            continue
        fi

        # Read and execute migration
        local sql
        sql=$(cat "${migration_file}")

        if [[ "${DRY_RUN}" == "false" ]]; then
            if kubectl exec -n "${NAMESPACE}" "$(get_postgres_pod)" -- psql -U victor -d victqor <<< "${sql}"; then
                execute_sql "INSERT INTO schema_migrations (version) VALUES ('${version}');" || true
                log_success "Migration ${version} applied"
            else
                log_error "Migration ${version} failed"
                return 1
            fi
        fi
    done <<< "${migrations}"

    log_success "Migrations completed"
}

################################################################################
# Verification
################################################################################

verify_database() {
    log_info "Verifying database setup..."

    local pod
    pod=$(get_postgres_pod)

    # Check schemas exist
    local schemas
    schemas=$(kubectl exec -n "${NAMESPACE}" "${pod}" -- psql -U victor -d victor -tAc \
        "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name IN ('event_store', 'workflows', 'checkpoints', 'sessions');" 2>/dev/null || echo "0")

    if [[ "${schemas}" -ge 4 ]]; then
        log_success "All schemas created (${schemas}/4)"
    else
        log_warning "Some schemas missing (${schemas}/4)"
    fi

    # Check tables exist
    local tables
    tables=$(kubectl exec -n "${NAMESPACE}" "${pod}" -- psql -U victor -d victor -tAc \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('event_store', 'workflows', 'checkpoints', 'sessions');" 2>/dev/null || echo "0")

    if [[ "${tables}" -ge 4 ]]; then
        log_success "All tables created (${tables}/4)"
    else
        log_warning "Some tables missing (${tables}/4)"
    fi

    # Check extensions
    local extensions
    extensions=$(kubectl exec -n "${NAMESPACE}" "${pod}" -- psql -U victor -d victor -tAc \
        "SELECT COUNT(*) FROM pg_extension WHERE ext_name IN ('uuid-ossp', 'pg_stat_statements', 'pg_trgm');" 2>/dev/null || echo "0")

    if [[ "${extensions}" -ge 3 ]]; then
        log_success "All extensions installed (${extensions}/3)"
    else
        log_warning "Some extensions missing (${extensions}/3)"
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    parse_arguments "$@"

    echo "==================================================================="
    echo "Victor AI Database Initialization"
    echo "==================================================================="
    echo "Namespace:       ${NAMESPACE}"
    echo "Migrations Path: ${MIGRATIONS_PATH}"
    echo "Dry Run:         ${DRY_RUN}"
    echo "==================================================================="
    echo ""

    # Check prerequisites
    local pod
    pod=$(get_postgres_pod)

    if [[ -z "${pod}" ]]; then
        log_error "PostgreSQL pod not found in namespace ${NAMESPACE}"
        log_error "Ensure infrastructure is deployed:"
        log_error "  ./deployment/scripts/deploy_infrastructure.sh --components postgres"
        exit 1
    fi

    log_success "PostgreSQL pod found: ${pod}"

    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    local max_attempts=30
    local attempt=0

    while [[ ${attempt} -lt ${max_attempts} ]]; do
        if kubectl exec -n "${NAMESPACE}" "${pod}" -- pg_isready -U victor -d victor &>/dev/null; then
            log_success "Database is ready"
            break
        fi

        attempt=$((attempt + 1))
        sleep 2
    done

    if [[ ${attempt} -eq ${max_attempts} ]]; then
        log_error "Database not ready after ${max_attempts} attempts"
        exit 1
    fi

    # Run initialization
    install_extensions
    create_schemas
    create_tables
    create_indexes
    run_migrations

    # Verify
    if [[ "${DRY_RUN}" == "false" ]]; then
        verify_database
    fi

    echo ""
    log_success "Database initialization complete"
}

# Run main function
main "$@"
