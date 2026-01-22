#!/bin/bash
################################################################################
# Secrets Verification Script
#
# Verifies that all required Kubernetes secrets are properly configured.
# Shows secret status without revealing secret values.
#
# Usage:
#   ./verify_secrets.sh [--namespace NAMESPACE] [--detailed]
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-victor-ai}"
DETAILED=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--detailed)
            DETAILED=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [OPTIONS]

Secret Verification Script

Options:
  -n, --namespace NAMESPACE    Kubernetes namespace (default: victor-ai)
  -d, --detailed               Show detailed secret information
  -h, --help                   Show this help message

EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_section "Victor AI - Secrets Verification"
echo "Namespace: $NAMESPACE"
echo ""

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log_error "Namespace '$NAMESPACE' does not exist"
    exit 1
fi

log_success "Namespace exists: $NAMESPACE"

################################################################################
# victor-ai-secrets
################################################################################

log_section "Checking victor-ai-secrets"

if kubectl get secret victor-ai-secrets -n "$NAMESPACE" &> /dev/null; then
    log_success "Secret exists: victor-ai-secrets"

    if [[ "$DETAILED" == true ]]; then
        echo ""
        echo "Secret keys:"
        kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath='{.data}' | jq -r 'keys[]' | while read -r key; do
            local value_length
            value_length=$(kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" | base64 -d | wc -c)
            echo "  - $key (${value_length} bytes)"
        done
    fi

    echo ""
    echo "Required keys:"

    local required_keys=("database-url" "encryption-key" "jwt-secret")
    local all_present=true

    for key in "${required_keys[@]}"; do
        if kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" &> /dev/null; then
            local is_empty
            is_empty=$(kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}")

            if [[ -z "$is_empty" ]]; then
                log_error "  $key - EMPTY"
                all_present=false
            else
                log_success "  $key - SET"
            fi
        else
            log_error "  $key - MISSING"
            all_present=false
        fi
    done

    if [[ "$all_present" == true ]]; then
        log_success "All required keys are present"
    else
        log_error "Some required keys are missing"
    fi

    echo ""
    echo "Optional keys:"

    local optional_keys=("anthropic-api-key" "openai-api-key" "redis-url" "sentry-dsn" "datadog-api-key")
    for key in "${optional_keys[@]}"; do
        if kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" &> /dev/null; then
            log_success "  $key - SET"
        else
            log_warning "  $key - NOT SET"
        fi
    done

else
    log_error "Secret NOT found: victor-ai-secrets"
fi

################################################################################
# alertmanager-secrets
################################################################################

log_section "Checking alertmanager-secrets"

if kubectl get secret alertmanager-secrets -n "$NAMESPACE" &> /dev/null; then
    log_success "Secret exists: alertmanager-secrets"

    if [[ "$DETAILED" == true ]]; then
        echo ""
        echo "Secret keys:"
        kubectl get secret alertmanager-secrets -n "$NAMESPACE" -o jsonpath='{.data}' | jq -r 'keys[]' | while read -r key; do
            local value_length
            value_length=$(kubectl get secret alertmanager-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" | base64 -d | wc -c)
            echo "  - $key (${value_length} bytes)"
        done
    fi

    echo ""
    echo "Alerting keys:"

    local alerting_keys=("smtp-host" "smtp-user" "slack-webhook-url" "pagerduty-integration-key")
    local has_alerting=false

    for key in "${alerting_keys[@]}"; do
        if kubectl get secret alertmanager-secrets -n "$NAMESPACE" -o jsonpath="{.data.$key}" &> /dev/null; then
            log_success "  $key - SET"
            has_alerting=true
        fi
    done

    if [[ "$has_alerting" == false ]]; then
        log_warning "No alerting credentials configured"
    fi

else
    log_warning "Secret NOT found: alertmanager-secrets (optional)"
fi

################################################################################
# grafana-credentials
################################################################################

log_section "Checking grafana-credentials"

if kubectl get secret grafana-credentials -n "$NAMESPACE" &> /dev/null; then
    log_success "Secret exists: grafana-credentials"

    if kubectl get secret grafana-credentials -n "$NAMESPACE" -o jsonpath="{.data.admin-password}" &> /dev/null; then
        local is_empty
        is_empty=$(kubectl get secret grafana-credentials -n "$NAMESPACE" -o jsonpath="{.data.admin-password}")

        if [[ -z "$is_empty" ]]; then
            log_error "  admin-password - EMPTY"
        else
            log_success "  admin-password - SET"
        fi
    else
        log_error "  admin-password - MISSING"
    fi

else
    log_error "Secret NOT found: grafana-credentials"
fi

################################################################################
# Summary
################################################################################

log_section "Summary"

echo "Commands to inspect secrets:"
echo ""
echo "  # List all secrets"
echo "  kubectl get secrets -n $NAMESPACE"
echo ""
echo "  # Describe specific secret (without showing values)"
echo "  kubectl describe secret victor-ai-secrets -n $NAMESPACE"
echo ""
echo "  # Decode specific secret value"
echo "  kubectl get secret victor-ai-secrets -n $NAMESPACE -o jsonpath='{.data.database-url}' | base64 -d"
echo ""
echo "  # Show all secret keys (with lengths)"
echo "  kubectl get secret victor-ai-secrets -n $NAMESPACE -o jsonpath='{.data}' | jq -r 'to_entries[] | \"\(.key): \(.value | @base64d | length) bytes\"'"
echo ""

################################################################################
# Security checks
################################################################################

log_section "Security Checks"

# Check for secrets with weak passwords
log_info "Checking for weak password patterns..."

if kubectl get secret victor-ai-secrets -n "$NAMESPACE" &> /dev/null; then
    local db_url
    db_url=$(kubectl get secret victor-ai-secrets -n "$NAMESPACE" -o jsonpath='{.data.database-url}' | base64 -d 2>/dev/null || echo "")

    if [[ "$db_url" =~ (password|changeme|123456|admin|secret) ]]; then
        log_warning "Database URL may contain a weak password"
    fi
fi

# Check for default/placeholder values
log_info "Checking for placeholder values..."

local secrets_to_check=("victor-ai-secrets" "alertmanager-secrets" "grafana-credentials")

for secret in "${secrets_to_check[@]}"; do
    if kubectl get secret "$secret" -n "$NAMESPACE" &> /dev/null; then
        local has_placeholder=false
        local keys
        keys=$(kubectl get secret "$secret" -n "$NAMESPACE" -o jsonpath='{.data}' | jq -r 'keys[]')

        for key in $keys; do
            local value
            value=$(kubectl get secret "$secret" -n "$NAMESPACE" -o jsonpath="{.data.$key}" | base64 -d 2>/dev/null || echo "")

            if [[ "$value" =~ (CHANGE_THIS|TODO|placeholder|your-key-here) ]]; then
                log_warning "$secret/$key contains placeholder value"
                has_placeholder=true
            fi
        done

        if [[ "$has_placeholder" == false ]]; then
            log_success "$secret - No placeholder values found"
        fi
    fi
done

# Check for old secrets (based on age)
log_info "Checking secret age..."

local oldest_secret_age=0
for secret in $(kubectl get secrets -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'); do
    local age
    age=$(kubectl get secret "$secret" -n "$NAMESPACE" -o jsonpath='{.metadata.creationTimestamp}' | awk -F'T' '{print $1}' | xargs -I {} date -d {} +%s 2>/dev/null || echo 0)

    if [[ $age -gt $oldest_secret_age ]]; then
        oldest_secret_age=$age
    fi
done

if [[ $oldest_secret_age -gt 0 ]]; then
    local current_time
    current_time=$(date +%s)
    local age_days=$(( (current_time - oldest_secret_age) / 86400 ))

    if [[ $age_days -gt 90 ]]; then
        log_warning "Some secrets are older than 90 days. Consider rotation."
    else
        log_success "Secrets are relatively new ($age_days days old)"
    fi
fi

echo ""
log_success "Verification complete"
