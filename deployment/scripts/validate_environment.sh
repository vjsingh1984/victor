#!/bin/bash
# Victor AI 0.5.1 - Environment Configuration Validation Script
#
# This script validates that an environment is properly configured
# before deployment. It checks:
# - Required environment variables
# - Secret existence and validity
# - External service connectivity
# - Resource limits
# - Feature flag settings
#
# Usage: ./validate_environment.sh <environment>
#   environment: development | testing | staging | production

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
ENVIRONMENT=""
ERRORS=0
WARNINGS=0
VALIDATION_REPORT=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((ERRORS++))
}

# Print header
print_header() {
    local env=$1
    echo "================================================"
    echo "Victor AI 0.5.1 - Environment Validation"
    echo "Environment: $env"
    echo "================================================"
    echo ""
}

# Check if required command exists
check_command() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required command '$cmd' not found"
        return 1
    else
        log_success "Command '$cmd' is available"
        return 0
    fi
}

# Validate environment variables
validate_environment_variables() {
    log_info "Validating environment variables..."

    local required_vars=(
        "VICTOR_PROFILE"
        "VICTOR_LOG_LEVEL"
        "VICTOR_MAX_WORKERS"
        "VICTOR_CACHE_SIZE"
        "event-bus-backend"
        "checkpoint-backend"
        "cache-backend"
        "enable-metrics"
        "enable-tracing"
        "enable-rate-limiting"
        "tool-selection-strategy"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            log_error "Required environment variable '$var' is not set"
        else
            log_success "Environment variable '$var' is set to '${!var}'"
        fi
    done

    # Environment-specific variable validation
    case "$ENVIRONMENT" in
        development)
            if [ "${VICTOR_ENABLE_DEV_TOOLS:-}" != "true" ]; then
                log_warning "VICTOR_ENABLE_DEV_TOOLS should be 'true' in development"
            fi
            if [ "${VICTOR_HOT_RELOAD:-}" != "true" ]; then
                log_warning "VICTOR_HOT_RELOAD should be 'true' in development"
            fi
            ;;
        testing)
            if [ "${checkpoint-backend:-}" != "sqlite" ]; then
                log_warning "checkpoint-backend should be 'sqlite' in testing"
            fi
            ;;
        staging)
            if [ "${event-bus-backend:-}" != "memory" ]; then
                log_warning "event-bus-backend should be 'memory' in staging"
            fi
            if [ -z "${rate-limit-requests-per-minute:-}" ]; then
                log_error "rate-limit-requests-per-minute must be set in staging"
            fi
            ;;
        production)
            if [ "${event-bus-backend:-}" != "kafka" ]; then
                log_error "event-bus-backend must be 'kafka' in production"
            fi
            if [ "${checkpoint-backend:-}" != "postgres" ]; then
                log_error "checkpoint-backend must be 'postgres' in production"
            fi
            if [ "${cache-backend:-}" != "redis" ]; then
                log_error "cache-backend must be 'redis' in production"
            fi
            ;;
    esac
}

# Validate secrets
validate_secrets() {
    log_info "Validating secrets..."

    # Check if secret file or environment variable is set
    if [ "$ENVIRONMENT" == "production" ] || [ "$ENVIRONMENT" == "staging" ]; then
        local secret_vars=(
            "DATABASE_URL"
            "REDIS_URL"
        )

        for var in "${secret_vars[@]}"; do
            if [ -z "${!var:-}" ]; then
                log_warning "Secret '$var' is not set (may be injected by external secrets operator)"
            else
                log_success "Secret '$var' is configured"
            fi
        done
    fi

    # Check API keys for providers
    local provider_keys=(
        "ANTHROPIC_API_KEY"
        "OPENAI_API_KEY"
    )

    for key in "${provider_keys[@]}"; do
        if [ -z "${!key:-}" ]; then
            log_warning "Provider key '$key' is not set (optional)"
        else
            log_success "Provider key '$key' is configured"
        fi
    done
}

# Validate external service connectivity
validate_service_connectivity() {
    log_info "Validating external service connectivity..."

    case "$ENVIRONMENT" in
        production|staging)
            # Check PostgreSQL connectivity
            if [ -n "${DATABASE_URL:-}" ]; then
                if command -v psql &> /dev/null; then
                    if psql "$DATABASE_URL" -c "SELECT 1;" &> /dev/null; then
                        log_success "PostgreSQL is reachable"
                    else
                        log_error "Cannot connect to PostgreSQL"
                    fi
                else
                    log_warning "psql not available, skipping PostgreSQL check"
                fi
            fi

            # Check Redis connectivity
            if [ -n "${REDIS_URL:-}" ]; then
                if command -v redis-cli &> /dev/null; then
                    if redis-cli -u "$REDIS_URL" ping &> /dev/null; then
                        log_success "Redis is reachable"
                    else
                        log_error "Cannot connect to Redis"
                    fi
                else
                    log_warning "redis-cli not available, skipping Redis check"
                fi
            fi

            # Check Kafka connectivity
            if [ "$ENVIRONMENT" == "production" ] && [ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]; then
                if command -v kafka-broker-api-versions &> /dev/null; then
                    if kafka-broker-api-versions --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" &> /dev/null; then
                        log_success "Kafka is reachable"
                    else
                        log_error "Cannot connect to Kafka"
                    fi
                else
                    log_warning "kafka-broker-api-versions not available, skipping Kafka check"
                fi
            fi
            ;;
        development|testing)
            log_success "No external services required for $ENVIRONMENT"
            ;;
    esac
}

# Validate resource limits
validate_resource_limits() {
    log_info "Validating resource limits..."

    local expected_cpu_request expected_cpu_limit
    local expected_memory_request expected_memory_limit

    case "$ENVIRONMENT" in
        development)
            expected_cpu_request="250m"
            expected_cpu_limit="1000m"
            expected_memory_request="256Mi"
            expected_memory_limit="1Gi"
            ;;
        testing)
            expected_cpu_request="250m"
            expected_cpu_limit="1000m"
            expected_memory_request="256Mi"
            expected_memory_limit="1Gi"
            ;;
        staging)
            expected_cpu_request="500m"
            expected_cpu_limit="2000m"
            expected_memory_request="512Mi"
            expected_memory_limit="2Gi"
            ;;
        production)
            expected_cpu_request="1000m"
            expected_cpu_limit="4000m"
            expected_memory_request="1Gi"
            expected_memory_limit="4Gi"
            ;;
    esac

    # Check if Kubernetes deployment exists
    if command -v kubectl &> /dev/null; then
        local namespace="victor-ai-${ENVIRONMENT}"
        if kubectl get namespace "$namespace" &> /dev/null; then
            log_success "Kubernetes namespace '$namespace' exists"

            # Check deployment
            if kubectl get deployment -n "$namespace" victor-ai &> /dev/null; then
                log_success "Kubernetes deployment 'victor-ai' exists"

                # Validate resource limits
                local actual_cpu_request
                actual_cpu_request=$(kubectl get deployment -n "$namespace" victor-ai -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')
                local actual_cpu_limit
                actual_cpu_limit=$(kubectl get deployment -n "$namespace" victor-ai -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}')
                local actual_memory_request
                actual_memory_request=$(kubectl get deployment -n "$namespace" victor-ai -o jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}')
                local actual_memory_limit
                actual_memory_limit=$(kubectl get deployment -n "$namespace" victor-ai -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}')

                if [ "$actual_cpu_request" == "$expected_cpu_request" ]; then
                    log_success "CPU request matches: $actual_cpu_request"
                else
                    log_warning "CPU request is '$actual_cpu_request', expected '$expected_cpu_request'"
                fi

                if [ "$actual_cpu_limit" == "$expected_cpu_limit" ]; then
                    log_success "CPU limit matches: $actual_cpu_limit"
                else
                    log_warning "CPU limit is '$actual_cpu_limit', expected '$expected_cpu_limit'"
                fi

                if [ "$actual_memory_request" == "$expected_memory_request" ]; then
                    log_success "Memory request matches: $actual_memory_request"
                else
                    log_warning "Memory request is '$actual_memory_request', expected '$expected_memory_request'"
                fi

                if [ "$actual_memory_limit" == "$expected_memory_limit" ]; then
                    log_success "Memory limit matches: $actual_memory_limit"
                else
                    log_warning "Memory limit is '$actual_memory_limit', expected '$expected_memory_limit'"
                fi
            else
                log_warning "Kubernetes deployment 'victor-ai' does not exist yet"
            fi
        else
            log_warning "Kubernetes namespace '$namespace' does not exist yet"
        fi
    else
        log_warning "kubectl not available, skipping Kubernetes checks"
    fi
}

# Validate feature flags
validate_feature_flags() {
    log_info "Validating feature flags..."

    # Validate profile
    if [ "${VICTOR_PROFILE:-}" != "$ENVIRONMENT" ]; then
        log_error "VICTOR_PROFILE is '${VICTOR_PROFILE:-not set}', expected '$ENVIRONMENT'"
    else
        log_success "VICTOR_PROFILE is correctly set to '$ENVIRONMENT'"
    fi

    # Validate log level
    local expected_log_level
    case "$ENVIRONMENT" in
        development|testing|staging)
            expected_log_level="DEBUG"
            ;;
        production)
            expected_log_level="INFO"
            ;;
    esac

    if [ "${VICTOR_LOG_LEVEL:-}" == "$expected_log_level" ]; then
        log_success "VICTOR_LOG_LEVEL is correctly set to '$expected_log_level'"
    else
        log_warning "VICTOR_LOG_LEVEL is '${VICTOR_LOG_LEVEL:-not set}', expected '$expected_log_level'"
    fi

    # Validate tool selection strategy
    local expected_tool_selection
    case "$ENVIRONMENT" in
        development|testing)
            expected_tool_selection="keyword"
            ;;
        staging|production)
            expected_tool_selection="hybrid"
            ;;
    esac

    if [ "${tool-selection-strategy:-}" == "$expected_tool_selection" ]; then
        log_success "tool-selection-strategy is correctly set to '$expected_tool_selection'"
    else
        log_warning "tool-selection-strategy is '${tool-selection-strategy:-not set}', expected '$expected_tool_selection'"
    fi

    # Validate rate limiting
    case "$ENVIRONMENT" in
        development|testing)
            if [ "${enable-rate-limiting:-}" == "true" ]; then
                log_warning "Rate limiting should be disabled in $ENVIRONMENT"
            else
                log_success "Rate limiting is correctly disabled in $ENVIRONMENT"
            fi
            ;;
        staging|production)
            if [ "${enable-rate-limiting:-}" != "true" ]; then
                log_error "Rate limiting must be enabled in $ENVIRONMENT"
            else
                log_success "Rate limiting is correctly enabled in $ENVIRONMENT"
            fi

            if [ "$ENVIRONMENT" == "staging" ]; then
                if [ "${rate-limit-requests-per-minute:-}" == "500" ]; then
                    log_success "Rate limit is correctly set to 500/min"
                else
                    log_warning "Rate limit is '${rate-limit-requests-per-minute:-not set}', expected '500'"
                fi
            elif [ "$ENVIRONMENT" == "production" ]; then
                if [ "${rate-limit-requests-per-minute:-}" == "1000" ]; then
                    log_success "Rate limit is correctly set to 1000/min"
                else
                    log_warning "Rate limit is '${rate-limit-requests-per-minute:-not set}', expected '1000'"
                fi
            fi
            ;;
    esac
}

# Validate Kubernetes configuration files
validate_kubernetes_config() {
    log_info "Validating Kubernetes configuration files..."

    local overlay_dir="/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/$ENVIRONMENT"

    if [ ! -d "$overlay_dir" ]; then
        log_error "Kubernetes overlay directory does not exist: $overlay_dir"
        return
    fi

    # Check kustomization.yaml
    local kustomization_file="$overlay_dir/kustomization.yaml"
    if [ -f "$kustomization_file" ]; then
        log_success "kustomization.yaml exists"

        # Validate YAML syntax
        if command -v yamllint &> /dev/null; then
            if yamllint "$kustomization_file" &> /dev/null; then
                log_success "kustomization.yaml has valid YAML syntax"
            else
                log_error "kustomization.yaml has YAML syntax errors"
            fi
        fi
    else
        log_error "kustomization.yaml not found in $overlay_dir"
    fi

    # Check deployment-patch.yaml
    local deployment_patch="$overlay_dir/deployment-patch.yaml"
    if [ -f "$deployment_patch" ]; then
        log_success "deployment-patch.yaml exists"

        if command -v yamllint &> /dev/null; then
            if yamllint "$deployment_patch" &> /dev/null; then
                log_success "deployment-patch.yaml has valid YAML syntax"
            else
                log_error "deployment-patch.yaml has YAML syntax errors"
            fi
        fi
    else
        log_warning "deployment-patch.yaml not found in $overlay_dir"
    fi

    # Check HPA patch for production
    if [ "$ENVIRONMENT" == "production" ]; then
        local hpa_patch="$overlay_dir/hpa-patch.yaml"
        if [ -f "$hpa_patch" ]; then
            log_success "hpa-patch.yaml exists"

            if command -v yamllint &> /dev/null; then
                if yamllint "$hpa_patch" &> /dev/null; then
                    log_success "hpa-patch.yaml has valid YAML syntax"
                else
                    log_error "hpa-patch.yaml has YAML syntax errors"
                fi
            fi
        else
            log_error "hpa-patch.yaml not found in $overlay_dir (required for production)"
        fi
    fi
}

# Generate validation report
generate_report() {
    local report_file="/tmp/victor_environment_validation_${ENVIRONMENT}_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "================================================"
        echo "Victor AI 0.5.1 - Environment Validation Report"
        echo "Environment: $ENVIRONMENT"
        echo "Timestamp: $(date)"
        echo "================================================"
        echo ""
        echo "Errors: $ERRORS"
        echo "Warnings: $WARNINGS"
        echo ""
        echo "================================================"
        echo "Validation Summary"
        echo "================================================"
        echo ""
        if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
            echo "✓ All validations passed successfully"
            echo ""
            echo "Environment $ENVIRONMENT is ready for deployment."
        elif [ $ERRORS -eq 0 ]; then
            echo "⚠ Validations passed with $WARNINGS warning(s)"
            echo ""
            echo "Review warnings before proceeding with deployment."
        else
            echo "✗ Validations failed with $ERRORS error(s) and $WARNINGS warning(s)"
            echo ""
            echo "Fix all errors before proceeding with deployment."
        fi
        echo ""
        echo "================================================"
    } > "$report_file"

    VALIDATION_REPORT="$report_file"
    log_info "Validation report saved to: $report_file"
}

# Main validation function
main() {
    # Check arguments
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <environment>"
        echo "  environment: development | testing | staging | production"
        exit 1
    fi

    ENVIRONMENT=$1

    # Validate environment name
    case "$ENVIRONMENT" in
        development|testing|staging|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            echo "Valid environments: development, testing, staging, production"
            exit 1
            ;;
    esac

    # Print header
    print_header "$ENVIRONMENT"

    # Check required commands
    check_command "kubectl" || true
    check_command "yamllint" || true

    # Run validations
    validate_environment_variables
    echo ""
    validate_secrets
    echo ""
    validate_service_connectivity
    echo ""
    validate_resource_limits
    echo ""
    validate_feature_flags
    echo ""
    validate_kubernetes_config
    echo ""

    # Generate report
    generate_report

    # Print summary
    echo "================================================"
    echo "Validation Summary"
    echo "================================================"
    echo ""
    echo "Errors:   $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""

    # Exit with appropriate code
    if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
        log_success "All validations passed!"
        exit 0
    elif [ $ERRORS -eq 0 ]; then
        log_warning "Validations passed with warnings"
        exit 0
    else
        log_error "Validations failed"
        exit 1
    fi
}

# Run main function
main "$@"
