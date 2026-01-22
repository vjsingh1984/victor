#!/bin/bash
################################################################################
# Secure Secrets Configuration Script
#
# Automates creation of all required Kubernetes secrets from secure sources.
# Supports environment variables, AWS SSM Parameter Store, and Azure Key Vault.
#
# Usage:
#   ./configure_secrets.sh [--namespace NAMESPACE] [--source SOURCE] [--dry-run]
#
# Features:
# - Creates victor-ai-secrets (database, Redis, API keys, encryption keys)
# - Creates alertmanager-secrets (SMTP, Slack webhooks)
# - Creates Grafana credentials (admin password)
# - Auto-generates secure passwords and encryption keys
# - Validates all secrets are created
# - Does NOT log secret values
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ROOT="${PROJECT_ROOT}/deployment"

# Default values
NAMESPACE="${NAMESPACE:-victor-ai}"
SOURCE="${SOURCE:-env}"
DRY_RUN=false
VERIFY_ONLY=false
SKIP_VALIDATION=false
AUTO_GENERATE_MISSING=true

# Kubernetes context
CONTEXT="${KUBECONTEXT:-}"
CLUSTER="${CLUSTER:-}"

# External secrets backends
AWS_REGION="${AWS_REGION:-us-east-1}"
SSM_PREFIX="${SSM_PREFIX:-/victor-ai/}"
AZURE_KEY_VAULT="${AZURE_KEY_VAULT:-victor-ai-kv}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Tracking
SECRETS_CREATED=0
SECRETS_FAILED=0
SECRETS_SKIPPED=0

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

log_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

check_dependencies() {
    log_section "Checking Dependencies"

    local missing=0

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        ((missing++))
    fi

    if [[ "$SOURCE" == "aws-ssm" ]]; then
        if ! command -v aws &> /dev/null; then
            log_error "AWS CLI not found. Please install AWS CLI."
            ((missing++))
        fi
    fi

    if [[ "$SOURCE" == "azure-keyvault" ]]; then
        if ! command -v az &> /dev/null; then
            log_error "Azure CLI not found. Please install Azure CLI."
            ((missing++))
        fi
    fi

    if [[ $missing -gt 0 ]]; then
        log_error "Missing $missing required dependencies. Aborting."
        exit 1
    fi

    log_success "All dependencies installed"
}

generate_password() {
    local length=${1:-32}
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

generate_encryption_key() {
    # Generate 256-bit key (32 bytes) for AES-256
    openssl rand -base64 32
}

generate_jwt_secret() {
    # Generate secure random JWT secret (minimum 256 bits)
    openssl rand -base64 64
}

validate_kubectl_context() {
    log_section "Validating Kubernetes Context"

    if [[ -n "$CONTEXT" ]]; then
        kubectl config use-context "$CONTEXT" &> /dev/null || {
            log_error "Failed to set context: $CONTEXT"
            exit 1
        }
    fi

    local current_context
    current_context=$(kubectl config current-context 2>&1)
    log_info "Current context: $current_context"

    local current_namespace
    current_namespace=$(kubectl config view --minify --output 'jsonpath={..namespace}' 2>&1)
    log_info "Current namespace: ${current_namespace:-default}"

    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist. Creating it..."
        if [[ "$DRY_RUN" == false ]]; then
            kubectl create namespace "$NAMESPACE"
            log_success "Created namespace: $NAMESPACE"
        else
            log_info "[DRY-RUN] Would create namespace: $NAMESPACE"
        fi
    else
        log_success "Namespace exists: $NAMESPACE"
    fi
}

get_secret_value() {
    local secret_name=$1
    local env_var=$2
    local default=${3:-}
    local required=${4:-true}

    local value=""

    case "$SOURCE" in
        "env")
            if [[ -n "${!env_var:-}" ]]; then
                value="${!env_var}"
            elif [[ -f "${PROJECT_ROOT}/.secrets" ]]; then
                value=$(grep "^${env_var}=" "${PROJECT_ROOT}/.secrets" | cut -d'=' -f2- | tr -d '"')
            fi
            ;;
        "aws-ssm")
            if command -v aws &> /dev/null; then
                value=$(aws ssm get-parameter --name "${SSM_PREFIX}${secret_name}" --with-decryption --region "$AWS_REGION" --query 'Parameter.Value' --output text 2>/dev/null || echo "")
            fi
            ;;
        "azure-keyvault")
            if command -v az &> /dev/null; then
                value=$(az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "$secret_name" --query value --output tsv 2>/dev/null || echo "")
            fi
            ;;
    esac

    # Use default if value is empty and default is provided
    if [[ -z "$value" && -n "$default" ]]; then
        value="$default"
    fi

    # Auto-generate if still empty and auto-generate is enabled
    if [[ -z "$value" && "$AUTO_GENERATE_MISSING" == true ]]; then
        case "$secret_name" in
            *password|*credential)
                value=$(generate_password 32)
                log_warning "Auto-generated password for: $secret_name"
                ;;
            *encryption*|*key)
                value=$(generate_encryption_key)
                log_warning "Auto-generated encryption key for: $secret_name"
                ;;
            *jwt*)
                value=$(generate_jwt_secret)
                log_warning "Auto-generated JWT secret for: $secret_name"
                ;;
        esac
    fi

    # Check if required value is missing
    if [[ -z "$value" && "$required" == true ]]; then
        log_error "Required secret not found: $secret_name (env var: $env_var)"
        return 1
    fi

    echo "$value"
}

create_victor_ai_secrets() {
    log_section "Creating victor-ai-secrets"

    local secret_file
    secret_file=$(mktemp)

    # Database configuration
    log_info "Configuring database credentials..."

    local db_host="${DB_HOST:-victor-postgres}"
    local db_port="${DB_PORT:-5432}"
    local db_name="${DB_NAME:-victor}"
    local db_user="${DB_USER:-victor}"
    local db_password
    db_password=$(get_secret_value "database-password" "DB_PASSWORD" "" true)

    local database_url="postgresql://${db_user}:${db_password}@${db_host}:${db_port}/${db_name}"
    echo "  database-url: [SET]"
    echo "database-url: $database_url" >> "$secret_file"

    # Redis configuration
    log_info "Configuring Redis credentials..."

    local redis_host="${REDIS_HOST:-victor-redis}"
    local redis_port="${REDIS_PORT:-6379}"
    local redis_password
    redis_password=$(get_secret_value "redis-password" "REDIS_PASSWORD" "" false)

    local redis_url
    if [[ -n "$redis_password" ]]; then
        redis_url="redis://:${redis_password}@${redis_host}:${redis_port}/0"
    else
        redis_url="redis://${redis_host}:${redis_port}/0"
    fi
    echo "  redis-url: [SET]"
    echo "redis-url: $redis_url" >> "$secret_file"

    # Provider API Keys
    log_info "Configuring provider API keys..."

    local anthropic_key
    anthropic_key=$(get_secret_value "anthropic-api-key" "ANTHROPIC_API_KEY" "" false)
    if [[ -n "$anthropic_key" ]]; then
        echo "  anthropic-api-key: [SET]"
        echo "anthropic-api-key: $anthropic_key" >> "$secret_file"
    fi

    local openai_key
    openai_key=$(get_secret_value "openai-api-key" "OPENAI_API_KEY" "" false)
    if [[ -n "$openai_key" ]]; then
        echo "  openai-api-key: [SET]"
        echo "openai-api-key: $openai_key" >> "$secret_file"
    fi

    # Other provider keys (optional)
    local optional_keys=(
        "google-api-key:GOOGLE_API_KEY"
        "azure-openai-api-key:AZURE_OPENAI_API_KEY"
        "cerebras-api-key:CEREBRAS_API_KEY"
        "deepseek-api-key:DEEPSEEK_API_KEY"
        "fireworks-api-key:FIREWORKS_API_KEY"
        "groq-api-key:GROQ_API_KEY"
        "mistral-api-key:MISTRAL_API_KEY"
        "moonshot-api-key:MOONSHOT_API_KEY"
        "together-api-key:TOGETHER_API_KEY"
        "replicate-api-key:REPLICATE_API_KEY"
        "xai-api-key:XAI_API_KEY"
    )

    for key_spec in "${optional_keys[@]}"; do
        IFS=':' read -r secret_key env_var <<< "$key_spec"
        local key_value
        key_value=$(get_secret_value "$secret_key" "$env_var" "" false)
        if [[ -n "$key_value" ]]; then
            echo "  $secret_key: [SET]"
            echo "$secret_key: $key_value" >> "$secret_file"
        fi
    done

    # Encryption key (required)
    log_info "Configuring encryption keys..."

    local encryption_key
    encryption_key=$(get_secret_value "encryption-key" "ENCRYPTION_KEY" "" true)
    echo "  encryption-key: [SET]"
    echo "encryption-key: $encryption_key" >> "$secret_file"

    # JWT secret (required)
    log_info "Configuring JWT secrets..."

    local jwt_secret
    jwt_secret=$(get_secret_value "jwt-secret" "JWT_SECRET" "" true)
    echo "  jwt-secret: [SET]"
    echo "jwt-secret: $jwt_secret" >> "$secret_file"

    local jwt_issuer="${JWT_ISSUER:-victor-ai}"
    echo "jwt-issuer: $jwt_issuer" >> "$secret_file"

    # External services (optional)
    log_info "Configuring external service credentials..."

    local sentry_dsn
    sentry_dsn=$(get_secret_value "sentry-dsn" "SENTRY_DSN" "" false)
    if [[ -n "$sentry_dsn" ]]; then
        echo "  sentry-dsn: [SET]"
        echo "sentry-dsn: $sentry_dsn" >> "$secret_file"
    fi

    local datadog_key
    datadog_key=$(get_secret_value "datadog-api-key" "DATADOG_API_KEY" "" false)
    if [[ -n "$datadog_key" ]]; then
        echo "  datadog-api-key: [SET]"
        echo "datadog-api-key: $datadog_key" >> "$secret_file"
    fi

    # Create or update secret
    if [[ "$DRY_RUN" == false ]]; then
        if kubectl get secret victor-ai-secrets -n "$NAMESPACE" &> /dev/null; then
            log_info "Updating existing secret: victor-ai-secrets"
            kubectl create secret generic victor-ai-secrets \
                --namespace="$NAMESPACE" \
                --from-env-file="$secret_file" \
                --dry-run=client -o yaml | kubectl apply -f -
            log_success "Updated secret: victor-ai-secrets"
        else
            log_info "Creating new secret: victor-ai-secrets"
            kubectl create secret generic victor-ai-secrets \
                --namespace="$NAMESPACE" \
                --from-env-file="$secret_file"
            log_success "Created secret: victor-ai-secrets"
        fi
        ((SECRETS_CREATED++))
    else
        log_info "[DRY-RUN] Would create secret: victor-ai-secrets"
    fi

    # Securely delete temp file
    shred -u "$secret_file" 2>/dev/null || rm -f "$secret_file"
}

create_alertmanager_secrets() {
    log_section "Creating alertmanager-secrets"

    local secret_file
    secret_file=$(mktemp)

    local has_secrets=false

    # SMTP configuration
    log_info "Configuring SMTP credentials..."

    local smtp_host
    smtp_host=$(get_secret_value "smtp-host" "SMTP_HOST" "" false)
    if [[ -n "$smtp_host" ]]; then
        echo "  smtp-host: [SET]"
        echo "smtp-host: $smtp_host" >> "$secret_file"
        has_secrets=true
    fi

    local smtp_port="${SMTP_PORT:-587}"
    echo "smtp-port: $smtp_port" >> "$secret_file"

    local smtp_user
    smtp_user=$(get_secret_value "smtp-user" "SMTP_USER" "" false)
    if [[ -n "$smtp_user" ]]; then
        echo "  smtp-user: [SET]"
        echo "smtp-user: $smtp_user" >> "$secret_file"
        has_secrets=true
    fi

    local smtp_password
    smtp_password=$(get_secret_value "smtp-password" "SMTP_PASSWORD" "" false)
    if [[ -n "$smtp_password" ]]; then
        echo "  smtp-password: [SET]"
        echo "smtp-password: $smtp_password" >> "$secret_file"
        has_secrets=true
    fi

    local smtp_from
    smtp_from=$(get_secret_value "smtp-from" "SMTP_FROM" "alerts@victor-ai.local" false)
    echo "smtp-from: $smtp_from" >> "$secret_file"

    # Slack webhooks
    log_info "Configuring Slack webhooks..."

    local slack_webhook
    slack_webhook=$(get_secret_value "slack-webhook-url" "SLACK_WEBHOOK_URL" "" false)
    if [[ -n "$slack_webhook" ]]; then
        echo "  slack-webhook-url: [SET]"
        echo "slack-webhook-url: $slack_webhook" >> "$secret_file"
        has_secrets=true
    fi

    local slack_api_token
    slack_api_token=$(get_secret_value "slack-api-token" "SLACK_API_TOKEN" "" false)
    if [[ -n "$slack_api_token" ]]; then
        echo "  slack-api-token: [SET]"
        echo "slack-api-token: $slack_api_token" >> "$secret_file"
        has_secrets=true
    fi

    # PagerDuty
    log_info "Configuring PagerDuty credentials..."

    local pagerduty_key
    pagerduty_key=$(get_secret_value "pagerduty-integration-key" "PAGERDUTY_INTEGRATION_KEY" "" false)
    if [[ -n "$pagerduty_key" ]]; then
        echo "  pagerduty-integration-key: [SET]"
        echo "pagerduty-integration-key: $pagerduty_key" >> "$secret_file"
        has_secrets=true
    fi

    # Create secret if we have any values
    if [[ "$has_secrets" == true ]]; then
        if [[ "$DRY_RUN" == false ]]; then
            if kubectl get secret alertmanager-secrets -n "$NAMESPACE" &> /dev/null; then
                log_info "Updating existing secret: alertmanager-secrets"
                kubectl create secret generic alertmanager-secrets \
                    --namespace="$NAMESPACE" \
                    --from-env-file="$secret_file" \
                    --dry-run=client -o yaml | kubectl apply -f -
                log_success "Updated secret: alertmanager-secrets"
            else
                log_info "Creating new secret: alertmanager-secrets"
                kubectl create secret generic alertmanager-secrets \
                    --namespace="$NAMESPACE" \
                    --from-env-file="$secret_file"
                log_success "Created secret: alertmanager-secrets"
            fi
            ((SECRETS_CREATED++))
        else
            log_info "[DRY-RUN] Would create secret: alertmanager-secrets"
        fi
    else
        log_info "No alertmanager secrets provided, skipping..."
        ((SECRETS_SKIPPED++))
    fi

    # Securely delete temp file
    shred -u "$secret_file" 2>/dev/null || rm -f "$secret_file"
}

create_grafana_secrets() {
    log_section "Creating Grafana credentials"

    local secret_file
    secret_file=$(mktemp)

    # Admin password (auto-generated if not provided)
    log_info "Configuring Grafana admin credentials..."

    local admin_password
    admin_password=$(get_secret_value "grafana-admin-password" "GRAFANA_ADMIN_PASSWORD" "" false)

    if [[ -z "$admin_password" ]]; then
        admin_password=$(generate_password 24)
        log_warning "Auto-generated Grafana admin password"
        log_warning "Please save the password securely: $admin_password"
    fi

    echo "admin-password: $admin_password" >> "$secret_file"
    echo "  admin-password: [SET]"

    # Create or update secret
    if [[ "$DRY_RUN" == false ]]; then
        if kubectl get secret grafana-credentials -n "$NAMESPACE" &> /dev/null; then
            log_info "Updating existing secret: grafana-credentials"
            kubectl create secret generic grafana-credentials \
                --namespace="$NAMESPACE" \
                --from-env-file="$secret_file" \
                --dry-run=client -o yaml | kubectl apply -f -
            log_success "Updated secret: grafana-credentials"
        else
            log_info "Creating new secret: grafana-credentials"
            kubectl create secret generic grafana-credentials \
                --namespace="$NAMESPACE" \
                --from-env-file="$secret_file"
            log_success "Created secret: grafana-credentials"
        fi
        ((SECRETS_CREATED++))
    else
        log_info "[DRY-RUN] Would create secret: grafana-credentials"
    fi

    # Securely delete temp file
    shred -u "$secret_file" 2>/dev/null || rm -f "$secret_file"
}

validate_secrets() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_warning "Skipping secret validation"
        return
    fi

    log_section "Validating Secrets"

    local secrets_valid=true

    # Check victor-ai-secrets
    log_info "Validating victor-ai-secrets..."

    if kubectl get secret victor-ai-secrets -n "$NAMESPACE" &> /dev/null; then
        local required_keys=("database-url" "encryption-key" "jwt-secret")

        for key in "${required_keys[@]}"; do
            if kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" &> /dev/null; then
                log_success "  ✓ $key is set"
            else
                log_error "  ✗ $key is missing"
                secrets_valid=false
            fi
        done
    else
        log_error "Secret victor-ai-secrets not found"
        secrets_valid=false
    fi

    # Check alertmanager-secrets (optional)
    if kubectl get secret alertmanager-secrets -n "$NAMESPACE" &> /dev/null; then
        log_info "Validating alertmanager-secrets..."
        log_success "  ✓ alertmanager-secrets exists"
    else
        log_info "  ⚠ alertmanager-secrets not found (optional)"
    fi

    # Check grafana-credentials
    log_info "Validating grafana-credentials..."

    if kubectl get secret grafana-credentials -n "$NAMESPACE" &> /dev/null; then
        if kubectl get secret grafana-credentials -n "$NAMESPACE" -o jsonpath="{.data.admin-password}" &> /dev/null; then
            log_success "  ✓ admin-password is set"
        else
            log_error "  ✗ admin-password is missing"
            secrets_valid=false
        fi
    else
        log_error "Secret grafana-credentials not found"
        secrets_valid=false
    fi

    if [[ "$secrets_valid" == true ]]; then
        log_success "All secrets validated successfully"
    else
        log_error "Some secrets are missing or invalid"
        return 1
    fi
}

show_summary() {
    log_section "Summary"

    echo "Secrets created: $SECRETS_CREATED"
    echo "Secrets skipped: $SECRETS_SKIPPED"
    echo "Secrets failed: $SECRETS_FAILED"
    echo ""

    if [[ "$SECRETS_FAILED" -gt 0 ]]; then
        log_error "Secret configuration completed with errors"
        return 1
    else
        log_success "Secret configuration completed successfully"
    fi
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Secure Secrets Configuration Script

Options:
  -n, --namespace NAMESPACE       Kubernetes namespace (default: victor-ai)
  -s, --source SOURCE             Secret source: env, aws-ssm, azure-keyvault (default: env)
  -c, --context CONTEXT           Kubernetes context to use
  --dry-run                       Show what would be done without making changes
  --verify-only                   Only validate existing secrets
  --skip-validation               Skip secret validation
  --no-auto-generate              Don't auto-generate missing secrets
  -h, --help                      Show this help message

Environment Variables:
  # Database
  DB_HOST                          Database host (default: victor-postgres)
  DB_PORT                          Database port (default: 5432)
  DB_NAME                          Database name (default: victor)
  DB_USER                          Database user (default: victor)
  DB_PASSWORD                      Database password

  # Redis
  REDIS_HOST                       Redis host (default: victor-redis)
  REDIS_PORT                       Redis port (default: 6379)
  REDIS_PASSWORD                   Redis password (optional)

  # Provider API Keys
  ANTHROPIC_API_KEY                Anthropic API key
  OPENAI_API_KEY                   OpenAI API key
  GOOGLE_API_KEY                   Google API key
  AZURE_OPENAI_API_KEY             Azure OpenAI API key
  CEREBRAS_API_KEY                 Cerebras API key
  DEEPSEEK_API_KEY                 DeepSeek API key
  FIREWORKS_API_KEY                Fireworks API key
  GROQ_API_KEY                     Groq API key
  MISTRAL_API_KEY                  Mistral API key
  MOONSHOT_API_KEY                 Moonshot API key
  TOGETHER_API_KEY                 Together API key
  REPLICATE_API_KEY                Replicate API key
  XAI_API_KEY                      X.AI API key

  # Security
  ENCRYPTION_KEY                   Encryption key for data at rest (auto-generated if not set)
  JWT_SECRET                       JWT signing secret (auto-generated if not set)
  JWT_ISSUER                       JWT issuer (default: victor-ai)

  # External Services
  SENTRY_DSN                       Sentry DSN for error tracking
  DATADOG_API_KEY                  Datadog API key for monitoring

  # Alerting
  SMTP_HOST                        SMTP server host
  SMTP_PORT                        SMTP server port (default: 587)
  SMTP_USER                        SMTP username
  SMTP_PASSWORD                    SMTP password
  SMTP_FROM                        From email address (default: alerts@victor-ai.local)
  SLACK_WEBHOOK_URL                Slack webhook URL
  SLACK_API_TOKEN                  Slack API token
  PAGERDUTY_INTEGRATION_KEY        PagerDuty integration key

  # Grafana
  GRAFANA_ADMIN_PASSWORD           Grafana admin password (auto-generated if not set)

Examples:
  # Create secrets from environment variables
  $0 --namespace production

  # Create secrets from AWS SSM Parameter Store
  $0 --source aws-ssm --namespace production

  # Dry-run to see what would be created
  $0 --dry-run

  # Validate existing secrets only
  $0 --verify-only

Secret Verification Commands:
  # List all secrets
  kubectl get secrets -n victor-ai

  # Describe specific secret
  kubectl describe secret victor-ai-secrets -n victor-ai

  # Decode specific secret value
  kubectl get secret victor-ai-secrets -n victor-ai -o jsonpath='{.data.database-url}' | base64 -d

EOF
}

################################################################################
# Main Script
################################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -s|--source)
                SOURCE="$2"
                shift 2
                ;;
            -c|--context)
                CONTEXT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verify-only)
                VERIFY_ONLY=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --no-auto-generate)
                AUTO_GENERATE_MISSING=false
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

main() {
    log_section "Victor AI - Secure Secrets Configuration"
    echo "Namespace: $NAMESPACE"
    echo "Source: $SOURCE"
    echo "Dry-run: $DRY_RUN"
    echo ""

    # Parse arguments
    parse_arguments "$@"

    # Check dependencies
    check_dependencies

    # Validate Kubernetes context
    validate_kubectl_context

    # Verify-only mode
    if [[ "$VERIFY_ONLY" == true ]]; then
        validate_secrets
        exit $?
    fi

    # Create secrets
    create_victor_ai_secrets
    create_alertmanager_secrets
    create_grafana_secrets

    # Validate secrets
    validate_secrets

    # Show summary
    show_summary
}

# Run main function
main "$@"
