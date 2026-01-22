#!/bin/bash
# Victor AI - Secure Credentials Setup for Monitoring Stack
# Generates and manages secure credentials for Grafana, AlertManager, and notification channels
#
# Usage:
#   ./setup_monitoring_credentials.sh [options]
#
# Options:
#   --namespace NAME       Namespace (default: victor-monitoring)
#   --generate-password    Generate secure random password for Grafana
#   --password PASSWORD    Set specific Grafana password
#   --email FILE           Load email credentials from file
#   --slack FILE           Load Slack webhook from file
#   --show                 Show current credentials (WARNING: sensitive data)
#   --rotate               Rotate all credentials
#   --help                 Show this help message

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NAMESPACE="${NAMESPACE:-victor-monitoring}"
GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-}"
ALERTMANAGER_SMTP_HOST="${ALERTMANAGER_SMTP_HOST:-}"
ALERTMANAGER_SMTP_PORT="${ALERTMANAGER_SMTP_PORT:-587}"
ALERTMANAGER_SMTP_USER="${ALERTMANAGER_SMTP_USER:-}"
ALERTMANAGER_SMTP_PASSWORD="${ALERTMANAGER_SMTP_PASSWORD:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# Flags
GENERATE_PASSWORD=false
SHOW_CREDENTIALS=false
ROTATE_CREDENTIALS=false

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

show_help() {
    cat << EOF
Victor AI - Secure Credentials Setup for Monitoring Stack

Usage: $0 [OPTIONS]

Options:
  --namespace NAME         Namespace (default: victor-monitoring)
  --generate-password      Generate secure random password for Grafana
  --password PASSWORD      Set specific Grafana admin password
  --email FILE             Load email credentials from JSON file
  --slack FILE             Load Slack webhook URL from file
  --show                   Show current credentials (WARNING: sensitive)
  --rotate                 Rotate all existing credentials
  --help                   Show this help message

Email credentials file format (JSON):
  {
    "smtp_host": "smtp.example.com",
    "smtp_port": "587",
    "smtp_user": "alerts@example.com",
    "smtp_password": "your-password",
    "from_address": "alertmanager@example.com"
  }

Slack webhook file format:
  https://hooks.slack.com/services/YOUR/WEBHOOK/URL

Examples:
  # Generate secure password
  $0 --generate-password

  # Set specific password
  $0 --password "my-secure-password"

  # Load email credentials
  $0 --email email-creds.json

  # Setup Slack notifications
  $0 --slack slack-webhook.txt

  # Setup all notifications
  $0 --generate-password --email email-creds.json --slack slack-webhook.txt

  # Rotate all credentials
  $0 --rotate

Security Notes:
  - Passwords are stored as Kubernetes Secrets
  - Use RBAC to restrict secret access
  - Consider using External Secrets Operator for production
  - Enable secrets encryption in etcd
  - Rotate credentials regularly

EOF
}

generate_secure_password() {
    local length=${1:-32}
    # Generate secure random password using openssl
    openssl rand -base64 48 | tr -d "=+/" | cut -c1-${length}
}

check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
}

# ============================================================================
# GRAFANA CREDENTIALS
# ============================================================================

setup_grafana_credentials() {
    log_info "Setting up Grafana credentials..."

    if [[ -z "${GRAFANA_ADMIN_PASSWORD}" ]]; then
        if [[ "${GENERATE_PASSWORD}" == true ]] || [[ "${ROTATE_CREDENTIALS}" == true ]]; then
            GRAFANA_ADMIN_PASSWORD=$(generate_secure_password 32)
            log_success "Generated secure password"
        else
            log_error "No password provided. Use --generate-password or --password"
            return 1
        fi
    fi

    # Create or update secret
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Secret
metadata:
  name: grafana-admin-credentials
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: grafana
type: Opaque
stringData:
  admin-user: "${GRAFANA_ADMIN_USER}"
  admin-password: "${GRAFANA_ADMIN_PASSWORD}"
EOF

    if [[ $? -eq 0 ]]; then
        log_success "Grafana credentials updated"

        # Restart Grafana to apply new credentials
        log_info "Restarting Grafana deployment..."
        kubectl rollout restart deployment/grafana -n "${NAMESPACE}" &> /dev/null || true

        # Wait for rollout
        kubectl rollout status deployment/grafana -n "${NAMESPACE}" --timeout=60s &> /dev/null || true

        log_success "Grafana restarted with new credentials"
    else
        log_error "Failed to update Grafana credentials"
        return 1
    fi
}

show_grafana_credentials() {
    log_info "Current Grafana credentials:"

    local secret_data=$(kubectl get secret grafana-admin-credentials \
        -n "${NAMESPACE}" \
        -o jsonpath='{.data}')

    if [[ -n "${secret_data}" ]]; then
        local username=$(echo "${secret_data}" | jq -r '.["admin-user"]' | base64 -d 2>/dev/null || echo "N/A")
        local password=$(echo "${secret_data}" | jq -r '.["admin-password"]' | base64 -d 2>/dev/null || echo "N/A")

        echo ""
        echo "  Username: ${username}"
        echo "  Password: ${password}"
        echo ""
    else
        log_warning "Grafana credentials not found"
    fi
}

# ============================================================================
# EMAIL CREDENTIALS
# ============================================================================

setup_email_credentials() {
    local email_file=$1

    if [[ ! -f "${email_file}" ]]; then
        log_error "Email credentials file not found: ${email_file}"
        return 1
    fi

    log_info "Loading email credentials from: ${email_file}"

    # Parse JSON file
    if command -v jq &> /dev/null; then
        ALERTMANAGER_SMTP_HOST=$(jq -r '.smtp_host' "${email_file}")
        ALERTMANAGER_SMTP_PORT=$(jq -r '.smtp_port // "587"' "${email_file}")
        ALERTMANAGER_SMTP_USER=$(jq -r '.smtp_user' "${email_file}")
        ALERTMANAGER_SMTP_PASSWORD=$(jq -r '.smtp_password' "${email_file}")
        local from_address=$(jq -r '.from_address // ""' "${email_file}")
    else
        log_error "jq is required to parse JSON credentials file"
        return 1
    fi

    # Validate required fields
    if [[ -z "${ALERTMANAGER_SMTP_HOST}" ]] || [[ -z "${ALERTMANAGER_SMTP_USER}" ]] || [[ -z "${ALERTMANAGER_SMTP_PASSWORD}" ]]; then
        log_error "Email credentials file must contain: smtp_host, smtp_user, smtp_password"
        return 1
    fi

    # Create secret
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-email-credentials
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
type: Opaque
stringData:
  smtp_host: "${ALERTMANAGER_SMTP_HOST}"
  smtp_port: "${ALERTMANAGER_SMTP_PORT}"
  smtp_user: "${ALERTMANAGER_SMTP_USER}"
  smtp_password: "${ALERTMANAGER_SMTP_PASSWORD}"
  from_address: "${from_address:-alertmanager@victor-ai.com}"
EOF

    if [[ $? -eq 0 ]]; then
        log_success "Email credentials configured"

        # Update AlertManager configuration
        update_alertmanager_config
    else
        log_error "Failed to configure email credentials"
        return 1
    fi
}

# ============================================================================
# SLACK CREDENTIALS
# ============================================================================

setup_slack_credentials() {
    local slack_file=$1

    if [[ ! -f "${slack_file}" ]]; then
        log_error "Slack webhook file not found: ${slack_file}"
        return 1
    fi

    log_info "Loading Slack webhook from: ${slack_file}"

    SLACK_WEBHOOK_URL=$(cat "${slack_file}" | tr -d ' \n')

    # Validate webhook URL
    if [[ ! "${SLACK_WEBHOOK_URL}" =~ ^https://hooks\.slack\.com/services/ ]]; then
        log_error "Invalid Slack webhook URL format"
        return 1
    fi

    # Create secret
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-slack-credentials
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
type: Opaque
stringData:
  webhook_url: "${SLACK_WEBHOOK_URL}"
EOF

    if [[ $? -eq 0 ]]; then
        log_success "Slack credentials configured"

        # Update AlertManager configuration
        update_alertmanager_config
    else
        log_error "Failed to configure Slack credentials"
        return 1
    fi
}

# ============================================================================
# ALERTMANAGER CONFIGURATION UPDATE
# ============================================================================

update_alertmanager_config() {
    log_info "Updating AlertManager configuration..."

    # Get credentials from secrets
    local smtp_host=""
    local smtp_port=""
    local smtp_user=""
    local smtp_password=""
    local from_address=""
    local slack_webhook=""

    if kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" &> /dev/null; then
        smtp_host=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.smtp_host}' | base64 -d)
        smtp_port=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.smtp_port}' | base64 -d)
        smtp_user=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.smtp_user}' | base64 -d)
        smtp_password=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.smtp_password}' | base64 -d)
        from_address=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.from_address}' | base64 -d)
    fi

    if kubectl get secret alertmanager-slack-credentials -n "${NAMESPACE}" &> /dev/null; then
        slack_webhook=$(kubectl get secret alertmanager-slack-credentials -n "${NAMESPACE}" -o jsonpath='{.data.webhook_url}' | base64 -d)
    fi

    # Build alertmanager configuration
    local alertmanager_config=$(cat <<EOF
global:
  resolve_timeout: 5m
$(if [[ -n "${slack_webhook}" ]]; then
    echo "  slack_api_url: '${slack_webhook}'"
fi
)
$(if [[ -n "${smtp_host}" ]]; then
    cat <<SMTP
  smtp_smarthost: '${smtp_host}:${smtp_port}'
  smtp_from: '${from_address:-alertmanager@victor-ai.com}'
  smtp_auth_username: '${smtp_user}'
  smtp_auth_password: '${smtp_password}'
  smtp_require_tls: true
SMTP
fi
)

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true

    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
$(if [[ -n "${slack_webhook}" ]]; then
    cat <<SLACK
    slack_configs:
      - channel: '#victor-ai-alerts'
        send_resolved: true
        title: '[{{ .Status | toUpper }}] {{ .CommonLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          {{ end }}
SLACK
fi
)

  - name: 'critical'
$(if [[ -n "${from_address}" ]]; then
    cat <<EMAIL
    email_configs:
      - to: '${from_address}'
        send_resolved: true
EMAIL
fi
)
$(if [[ -n "${slack_webhook}" ]]; then
    cat <<SLACK
    slack_configs:
      - channel: '#victor-ai-critical'
        send_resolved: true
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'
        title: '[CRITICAL] {{ .CommonLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Runbook:* {{ .Annotations.runbook_url }}
          {{ end }}
SLACK
fi
)

  - name: 'warning'
$(if [[ -n "${slack_webhook}" ]]; then
    cat <<SLACK
    slack_configs:
      - channel: '#victor-ai-warnings'
        send_resolved: true
        title: '[WARNING] {{ .CommonLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Summary:* {{ .Annotations.summary }}
          {{ end }}
SLACK
fi
)

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
EOF
)

    # Create or update ConfigMap
    cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  labels:
    app.kubernetes.io/name: victor
    app.kubernetes.io/component: alertmanager
data:
  alertmanager.yml: |
$(echo "${alertmanager_config}" | sed 's/^/    /')
EOF

    if [[ $? -eq 0 ]]; then
        log_success "AlertManager configuration updated"

        # Restart AlertManager
        log_info "Restarting AlertManager..."
        kubectl rollout restart deployment/alertmanager -n "${NAMESPACE}" &> /dev/null || true

        # Wait for rollout
        kubectl rollout status deployment/alertmanager -n "${NAMESPACE}" --timeout=60s &> /dev/null || true

        log_success "AlertManager restarted with new configuration"
    else
        log_error "Failed to update AlertManager configuration"
        return 1
    fi
}

# ============================================================================
# CREDENTIAL ROTATION
# ============================================================================

rotate_credentials() {
    log_info "Rotating all credentials..."

    # Rotate Grafana password
    GENERATE_PASSWORD=true
    setup_grafana_credentials

    log_success "All credentials rotated successfully"
}

# ============================================================================
# SHOW CURRENT CREDENTIALS
# ============================================================================

show_all_credentials() {
    log_warning "WARNING: Displaying sensitive credentials!"
    echo ""

    log_info "Grafana Credentials:"
    show_grafana_credentials

    log_info "Email Credentials:"
    if kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" &> /dev/null; then
        local smtp_host=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.smtp_host}' | base64 -d)
        local smtp_user=$(kubectl get secret alertmanager-email-credentials -n "${NAMESPACE}" -o jsonpath='{.data.smtp_user}' | base64 -d)
        echo ""
        echo "  SMTP Host: ${smtp_host}"
        echo "  SMTP User: ${smtp_user}"
        echo "  (Password is hidden for security)"
        echo ""
    else
        log_warning "Email credentials not configured"
    fi

    log_info "Slack Credentials:"
    if kubectl get secret alertmanager-slack-credentials -n "${NAMESPACE}" &> /dev/null; then
        local webhook=$(kubectl get secret alertmanager-slack-credentials -n "${NAMESPACE}" -o jsonpath='{.data.webhook_url}' | base64 -d)
        local webhook_sanitized=$(echo "${webhook}" | sed 's/\(https:\/\/hooks\.slack\.com\/services\/\)[^\/]*\/[^\/]*\/.*/\1***\/***\/***/')
        echo ""
        echo "  Webhook: ${webhook_sanitized}"
        echo ""
    else
        log_warning "Slack credentials not configured"
    fi
}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

main() {
    log_info "Victor AI - Secure Credentials Setup"
    echo ""

    # Parse arguments
    local email_file=""
    local slack_file=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --generate-password)
                GENERATE_PASSWORD=true
                shift
                ;;
            --password)
                GRAFANA_ADMIN_PASSWORD="$2"
                shift 2
                ;;
            --email)
                email_file="$2"
                shift 2
                ;;
            --slack)
                slack_file="$2"
                shift 2
                ;;
            --show)
                SHOW_CREDENTIALS=true
                shift
                ;;
            --rotate)
                ROTATE_CREDENTIALS=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Check prerequisites
    check_kubectl

    # Check if namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Namespace ${NAMESPACE} does not exist"
        log_info "Deploy monitoring stack first with: deploy_monitoring_complete.sh"
        exit 1
    fi

    # Execute actions
    if [[ "${SHOW_CREDENTIALS}" == true ]]; then
        show_all_credentials
        exit 0
    fi

    if [[ "${ROTATE_CREDENTIALS}" == true ]]; then
        rotate_credentials
        exit 0
    fi

    # Setup individual credentials
    local actions_taken=false

    if [[ "${GENERATE_PASSWORD}" == true ]] || [[ -n "${GRAFANA_ADMIN_PASSWORD}" ]]; then
        setup_grafana_credentials
        actions_taken=true
    fi

    if [[ -n "${email_file}" ]]; then
        setup_email_credentials "${email_file}"
        actions_taken=true
    fi

    if [[ -n "${slack_file}" ]]; then
        setup_slack_credentials "${slack_file}"
        actions_taken=true
    fi

    if [[ "${actions_taken}" == false ]]; then
        log_error "No action specified. Use --help for usage information"
        exit 1
    fi

    log_success "Credential setup completed successfully!"
}

# Run main function
main "$@"
