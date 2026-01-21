#!/bin/bash
# Victor AI Grafana Dashboard Configuration Script
# Load and configure Grafana dashboards

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GRAFANA_NAMESPACE="victor-monitoring"
GRAFANA_POD=""
GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="changeme123"

# Functions
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

get_grafana_pod() {
    log_info "Finding Grafana pod..."
    GRAFANA_POD=$(kubectl get pods -n "${GRAFANA_NAMESPACE}" -l app.kubernetes.io/component=grafana -o jsonpath='{.items[0].metadata.name}')

    if [[ -z "${GRAFANA_POD}" ]]; then
        log_error "Grafana pod not found in namespace ${GRAFANA_NAMESPACE}"
        exit 1
    fi

    log_success "Found Grafana pod: ${GRAFANA_POD}"
}

setup_port_forward() {
    log_info "Setting up port forwarding to Grafana..."

    # Kill any existing port-forward on 3000
    pkill -f "kubectl.*port-forward.*3000" || true
    sleep 2

    # Start port-forward in background
    kubectl port-forward -n "${GRAFANA_NAMESPACE}" "${GRAFANA_POD}" 3000:3000 > /dev/null 2>&1 &
    PORT_FORWARD_PID=$!

    # Wait for port-forward to be ready
    local max_wait=10
    local waited=0
    while [[ ${waited} -lt ${max_wait} ]]; do
        if curl -s "${GRAFANA_URL}" > /dev/null 2>&1; then
            log_success "Port forwarding established"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    log_error "Failed to establish port forward"
    return 1
}

cleanup_port_forward() {
    if [[ -n "${PORT_FORWARD_PID:-}" ]]; then
        log_info "Cleaning up port forwarding..."
        kill "${PORT_FORWARD_PID}" 2>/dev/null || true
        log_success "Port forwarding cleaned up"
    fi
}

get_grafana_token() {
    log_info "Authenticating with Grafana..."

    local response=$(curl -s "${GRAFANA_URL}/api/login" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"user\":\"${GRAFANA_USER}\",\"password\":\"${GRAFANA_PASSWORD}\"}")

    GRAFANA_TOKEN=$(echo "${response}" | jq -r '.message // empty')

    if [[ -n "${GRAFANA_TOKEN}" && "${GRAFANA_TOKEN}" != "null" ]]; then
        log_success "Authenticated with Grafana"
    else
        # Try basic auth
        GRAFANA_TOKEN=$(echo -n "${GRAFANA_USER}:${GRAFANA_PASSWORD}" | base64)
        log_success "Using basic authentication"
    fi
}

create_datasource() {
    log_info "Creating Prometheus datasource..."

    local response=$(curl -s "${GRAFANA_URL}/api/datasources" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "access": "proxy",
            "url": "http://prometheus:9090",
            "isDefault": true,
            "jsonData": {
                "timeInterval": "15s",
                "queryTimeout": "60s",
                "httpMethod": "POST"
            }
        }')

    local message=$(echo "${response}" | jq -r '.message // .id // empty')

    if [[ -n "${message}" && "${message}" != "null" ]]; then
        log_success "Datasource created: ${message}"
    else
        log_warning "Datasource might already exist or creation failed"
    fi
}

load_dashboard() {
    local dashboard_file=$1
    local dashboard_name=$(basename "${dashboard_file}" .json)

    log_info "Loading dashboard: ${dashboard_name}"

    if [[ ! -f "${dashboard_file}" ]]; then
        log_error "Dashboard file not found: ${dashboard_file}"
        return 1
    fi

    # Wrap dashboard in the structure Grafana expects
    local dashboard_json=$(jq -c --slurpfile dashboard "${dashboard_file}" '{
        overwrite: true,
        dashboard: $dashboard[0],
        message: "Imported from CLI",
        folderId: 0
    }' < "${dashboard_file}")

    local response=$(curl -s "${GRAFANA_URL}/api/dashboards/db" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
        -d "${dashboard_json}")

    local status=$(echo "${response}" | jq -r '.status // empty')
    local id=$(echo "${response}" | jq -r '.id // empty')

    if [[ "${status}" == "success" && -n "${id}" ]]; then
        log_success "Dashboard loaded: ${dashboard_name} (ID: ${id})"
    else
        log_warning "Failed to load dashboard: ${dashboard_name}"
        echo "${response}" | jq -r '.message // .error // "Unknown error"'
    fi
}

load_all_dashboards() {
    log_info "Loading all Grafana dashboards..."

    local dashboards=(
        "/tmp/monitoring/grafana_dashboards/provider_metrics.json"
        "/tmp/monitoring/grafana_dashboards/request_metrics.json"
        "/tmp/monitoring/grafana_dashboards/system_health.json"
        "/tmp/monitoring/grafana_dashboards/tool_execution.json"
        "/tmp/monitoring/grafana_dashboards/workflow_metrics.json"
    )

    local loaded=0
    local failed=0

    for dashboard in "${dashboards[@]}"; do
        if load_dashboard "${dashboard}"; then
            loaded=$((loaded + 1))
        else
            failed=$((failed + 1))
        fi
    done

    log_success "Dashboards loaded: ${loaded} succeeded, ${failed} failed"
}

list_dashboards() {
    log_info "Listing installed dashboards..."

    local response=$(curl -s "${GRAFANA_URL}/api/search?type=dash-db" \
        -H "Authorization: Bearer ${GRAFANA_TOKEN}")

    echo "${response}" | jq -r '.[] | "\(.title) (ID: \(.uid))"'
}

main() {
    log_info "Starting Grafana dashboard configuration..."
    echo ""

    # Parse arguments
    SKIP_DASHBOARDS=false
    LIST_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-dashboards)
                SKIP_DASHBOARDS=true
                shift
                ;;
            --list)
                LIST_ONLY=true
                shift
                ;;
            --namespace)
                GRAFANA_NAMESPACE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-dashboards  Skip loading dashboards"
                echo "  --list             List installed dashboards only"
                echo "  --namespace NAMESPACE  Grafana namespace (default: victor-monitoring)"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Main configuration flow
    get_grafana_pod
    setup_port_forward
    trap cleanup_port_forward EXIT

    get_grafana_token

    if [[ "${LIST_ONLY}" == true ]]; then
        list_dashboards
        exit 0
    fi

    create_datasource

    if [[ "${SKIP_DASHBOARDS}" == false ]]; then
        load_all_dashboards
    fi

    list_dashboards

    log_success "Grafana configuration completed successfully!"
}

# Run main function
main "$@"
