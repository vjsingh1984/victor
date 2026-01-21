#!/bin/bash
################################################################################
# Interactive Deployment Dashboard
#
# This script provides an interactive menu-driven interface for deployment
# management with progress visualization, health status, log tailing, and alerts.
#
# Usage:
#   ./deployment_dashboard.sh [environment]
#
# Features:
# - Menu-driven interface
# - Progress visualization
# - Health status display
# - Log tailing
# - Error alerts
################################################################################

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_ROOT="${PROJECT_ROOT}/deployment"
LOG_DIR="${PROJECT_ROOT}/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# State
ENVIRONMENT=""
CURRENT_MENU="main"
REFRESH_INTERVAL=5

################################################################################
# Helper Functions
################################################################################

print_header() {
    clear
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║           Victor Deployment Dashboard                    ║${NC}"
    echo -e "${BOLD}${BLUE}║           Environment: ${ENVIRONMENT}                        ║${NC}"
    echo -e "${BOLD}${BLUE}║           $(date '+%Y-%m-%d %H:%M:%S')                              ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_menu_item() {
    local number=$1
    local description=$2
    printf "  ${CYAN}%2d)${NC} %s\n" "${number}" "${description}"
}

print_status_item() {
    local label=$1
    local value=$2
    local status=$3

    case "${status}" in
        healthy|ok|running|success)
            printf "  ${GREEN}✓${NC} %-30s: ${GREEN}%s${NC}\n" "${label}" "${value}"
            ;;
        warning|pending)
            printf "  ${YELLOW}⚠${NC} %-30s: ${YELLOW}%s${NC}\n" "${label}" "${value}"
            ;;
        error|failed|down)
            printf "  ${RED}✗${NC} %-30s: ${RED}%s${NC}\n" "${label}" "${value}"
            ;;
        *)
            printf "  • %-30s: %s\n" "${label}" "${value}"
            ;;
    esac
}

wait_for_input() {
    echo ""
    read -p "Press Enter to continue or 'q' to return to menu: " choice
    if [[ "${choice}" == "q" ]]; then
        CURRENT_MENU="main"
    fi
}

################################################################################
# Dashboard Views
################################################################################

show_main_menu() {
    print_header
    echo -e "${BOLD}Main Menu${NC}"
    echo ""
    print_menu_item 1 "Deployment Status"
    print_menu_item 2 "Deploy New Version"
    print_menu_item 3 "Rollback Deployment"
    print_menu_item 4 "View Logs"
    print_menu_item 5 "Health Checks"
    print_menu_item 6 "Monitoring"
    print_menu_item 7 "Configuration"
    print_menu_item 8 "Run Diagnostics"
    print_menu_item 0 "Exit"
    echo ""
    read -p "Select option: " choice

    case "${choice}" in
        1) show_deployment_status ;;
        2) deploy_new_version ;;
        3) rollback_deployment ;;
        4) view_logs ;;
        5) show_health_checks ;;
        6) show_monitoring ;;
        7) show_configuration ;;
        8) run_diagnostics ;;
        0) exit 0 ;;
        *) echo "Invalid option"; sleep 1 ;;
    esac
}

show_deployment_status() {
    while true; do
        print_header
        echo -e "${BOLD}Deployment Status${NC}"
        echo ""

        export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

        # Get deployment info
        local replicas_ready
        local replicas_desired
        replicas_ready=$(kubectl get deployment victor -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        replicas_desired=$(kubectl get deployment victor -n victor -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        local deployment_status
        deployment_status=$(kubectl get deployment victor -n victor -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "False")

        local revision
        revision=$(kubectl get deployment victor -n victor -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}' 2>/dev/null || echo "unknown")

        # Print status
        print_status_item "Replicas" "${replicas_ready}/${replicas_desired}" $([[ "${replicas_ready}" -ge "${replicas_desired}" ]] && echo "healthy" || echo "warning")
        print_status_item "Deployment" $([[ "${deployment_status}" == "True" ]] && echo "Available" || echo "Unavailable") $([[ "${deployment_status}" == "True" ]] && echo "healthy" || echo "error")
        print_status_item "Revision" "${revision}" "ok"

        echo ""
        echo -e "${BOLD}Pods${NC}"

        # List pods
        local pods
        pods=$(kubectl get pods -n victor -l app=victor -o name 2>/dev/null || echo "")

        if [[ -n "${pods}" ]]; then
            while IFS= read -r pod; do
                local pod_name
                pod_name=$(echo "${pod}" | cut -d'/' -f2)

                local pod_status
                pod_status=$(kubectl get "${pod}" -n victor -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")

                local pod_ready
                pod_ready=$(kubectl get "${pod}" -n victor -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")

                print_status_item "${pod_name}" "${pod_status}" $([[ "${pod_status}" == "Running" ]] && echo "healthy" || echo "error")
            done <<< "${pods}"
        else
            echo "  No pods found"
        fi

        echo ""
        echo -e "${BOLD}Recent Events${NC}"

        # Show recent events
        kubectl get events -n victor --sort-by='.lastTimestamp' --field-selector type=Warning 2>/dev/null | tail -n5 | awk 'NR>1 {print "  " $0}' || echo "  No recent events"

        echo ""
        echo "Press 'q' to return to menu or 'r' to refresh..."
        read -t "${REFRESH_INTERVAL}" -n 1 choice

        if [[ "${choice}" == "q" ]]; then
            break
        fi
    done
}

deploy_new_version() {
    print_header
    echo -e "${BOLD}Deploy New Version${NC}"
    echo ""

    # Get current image
    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    local current_image
    current_image=$(kubectl get deployment victor -n victor -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "unknown")

    echo "Current image: ${current_image}"
    echo ""

    # Get new image tag
    read -p "Enter new image tag (or 'q' to cancel): " image_tag

    if [[ "${image_tag}" == "q" ]]; then
        return
    fi

    if [[ -z "${image_tag}" ]]; then
        echo "Invalid image tag"
        sleep 2
        return
    fi

    echo ""
    echo "Deploying with image tag: ${image_tag}"
    echo ""

    # Confirm
    read -p "Continue? (yes/no): " confirm

    if [[ "${confirm}" != "yes" ]]; then
        echo "Deployment cancelled"
        sleep 2
        return
    fi

    echo ""
    echo "Starting deployment..."
    echo ""

    # Run deployment script
    if [[ -f "${SCRIPT_DIR}/deploy_production.sh" ]]; then
        bash "${SCRIPT_DIR}/deploy_production.sh" --environment "${ENVIRONMENT}" --image-tag "${image_tag}" || {
            echo ""
            echo -e "${RED}Deployment failed${NC}"
            wait_for_input
            return
        }
    else
        echo "Deployment script not found"
        wait_for_input
        return
    fi

    echo ""
    echo -e "${GREEN}Deployment completed${NC}"
    wait_for_input
}

rollback_deployment() {
    print_header
    echo -e "${BOLD}Rollback Deployment${NC}"
    echo ""

    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Show current revision
    local current_revision
    current_revision=$(kubectl get deployment victor -n victor -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}' 2>/dev/null || echo "unknown")

    echo "Current revision: ${current_revision}"
    echo ""

    # Show available revisions
    echo "Available revisions:"
    kubectl rollout history deployment/victor -n victor 2>/dev/null | awk 'NR>1 {print "  " $0}' || echo "  No history available"
    echo ""

    # Get target revision
    read -p "Enter revision to rollback to (or 'q' to cancel): " revision

    if [[ "${revision}" == "q" ]]; then
        return
    fi

    if [[ -z "${revision}" ]]; then
        echo "Invalid revision"
        sleep 2
        return
    fi

    echo ""
    echo "Rolling back to revision: ${revision}"
    echo ""

    # Confirm
    read -p "Continue? (yes/no): " confirm

    if [[ "${confirm}" != "yes" ]]; then
        echo "Rollback cancelled"
        sleep 2
        return
    fi

    echo ""
    echo "Starting rollback..."
    echo ""

    # Run rollback script
    if [[ -f "${SCRIPT_DIR}/rollback_production.sh" ]]; then
        bash "${SCRIPT_DIR}/rollback_production.sh" "${ENVIRONMENT}" --revision "${revision}" || {
            echo ""
            echo -e "${RED}Rollback failed${NC}"
            wait_for_input
            return
        }
    else
        echo "Rollback script not found"
        wait_for_input
        return
    fi

    echo ""
    echo -e "${GREEN}Rollback completed${NC}"
    wait_for_input
}

view_logs() {
    print_header
    echo -e "${BOLD}View Logs${NC}"
    echo ""

    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Select pod
    echo "Available pods:"
    local pods
    pods=$(kubectl get pods -n victor -l app=victor -o name 2>/dev/null || echo "")

    if [[ -z "${pods}" ]]; then
        echo "No pods found"
        wait_for_input
        return
    fi

    local pod_list=()
    local index=1

    while IFS= read -r pod; do
        local pod_name
        pod_name=$(echo "${pod}" | cut -d'/' -f2)
        echo "  ${index}. ${pod_name}"
        pod_list+=("${pod_name}")
        index=$((index + 1))
    done <<< "${pods}"

    echo ""
    read -p "Select pod number (or 'q' to cancel): " choice

    if [[ "${choice}" == "q" ]]; then
        return
    fi

    if ! [[ "${choice}" =~ ^[0-9]+$ ]] || [[ ${choice} -lt 1 ]] || [[ ${choice} -gt ${#pod_list[@]} ]]; then
        echo "Invalid selection"
        sleep 2
        return
    fi

    local selected_pod="${pod_list[$((choice - 1))]}"

    echo ""
    echo "Tailing logs for: ${selected_pod}"
    echo "Press Ctrl+C to stop"
    echo ""

    # Tail logs
    kubectl logs -f -n victor "${selected_pod}" --all-containers=true 2>/dev/null || echo "Failed to fetch logs"
}

show_health_checks() {
    while true; do
        print_header
        echo -e "${BOLD}Health Checks${NC}"
        echo ""

        export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

        # Check cluster health
        local cluster_health
        cluster_health=$(kubectl get cs 2>/dev/null | awk 'NR>1 {print $3}' | grep -c "True" || echo "0")

        print_status_item "Cluster Health" "${cluster_health} healthy" $([[ "${cluster_health}" -gt 0 ]] && echo "healthy" || echo "error")

        # Check deployment
        local deployment_status
        deployment_status=$(kubectl get deployment victor -n victor -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "False")

        print_status_item "Deployment" $([[ "${deployment_status}" == "True" ]] && echo "Available" || echo "Unavailable") $([[ "${deployment_status}" == "True" ]] && echo "healthy" || echo "error")

        # Check pods
        local pods_running
        local pods_total
        pods_running=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -c "Running" || echo "0")
        pods_total=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | wc -w || echo "0")

        print_status_item "Pods" "${pods_running}/${pods_total} Running" $([[ "${pods_running}" -eq "${pods_total}" ]] && echo "healthy" || echo "warning")

        # Check endpoint
        local service_ip
        service_ip=$(kubectl get service victor -n victor -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")

        if [[ -n "${service_ip}" ]]; then
            local http_code
            http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://${service_ip}/health" 2>/dev/null || echo "000")

            print_status_item "Health Endpoint" "HTTP ${http_code}" $([[ "${http_code}" == "200" ]] && echo "healthy" || echo "error")
        else
            print_status_item "Health Endpoint" "Not accessible" "error"
        fi

        # Check dependencies
        echo ""
        echo -e "${BOLD}Dependencies${NC}"

        # PostgreSQL
        if kubectl get statefulset -n victor postgresql &>/dev/null; then
            local pg_ready
            pg_ready=$(kubectl get statefulset postgresql -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            print_status_item "PostgreSQL" "${pg_ready} replicas" $([[ "${pg_ready}" -gt 0 ]] && echo "healthy" || echo "error")
        else
            print_status_item "PostgreSQL" "Not found" "warning"
        fi

        # Redis
        if kubectl get statefulset -n victor redis &>/dev/null; then
            local redis_ready
            redis_ready=$(kubectl get statefulset redis -n victor -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            print_status_item "Redis" "${redis_ready} replicas" $([[ "${redis_ready}" -gt 0 ]] && echo "healthy" || echo "error")
        else
            print_status_item "Redis" "Not found" "warning"
        fi

        echo ""
        echo "Press 'q' to return to menu or 'r' to refresh..."
        read -t "${REFRESH_INTERVAL}" -n 1 choice

        if [[ "${choice}" == "q" ]]; then
            break
        fi
    done
}

show_monitoring() {
    print_header
    echo -e "${BOLD}Monitoring${NC}"
    echo ""

    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Resource usage
    echo -e "${BOLD}Resource Usage${NC}"
    echo ""

    kubectl top pods -n victor -l app=victor 2>/dev/null | awk 'NR>1 {printf "  %-40s %10s %10s\n", $1, $2, $3}' || echo "  Metrics not available"

    echo ""
    echo -e "${BOLD}Monitoring Links${NC}"
    echo ""

    # Get Grafana URL
    local grafana_url
    grafana_url=$(kubectl get ingress -n monitoring grafana -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")

    if [[ -n "${grafana_url}" ]]; then
        echo "  Grafana: http://${grafana_url}"
    fi

    # Get Prometheus URL
    local prometheus_url
    prometheus_url=$(kubectl get ingress -n monitoring prometheus -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")

    if [[ -n "${prometheus_url}" ]]; then
        echo "  Prometheus: http://${prometheus_url}"
    fi

    wait_for_input
}

show_configuration() {
    print_header
    echo -e "${BOLD}Configuration${NC}"
    echo ""

    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    # Environment variables
    echo -e "${BOLD}Environment Variables${NC}"
    echo ""

    local pod
    pod=$(kubectl get pods -n victor -l app=victor -o name | head -n1 2>/dev/null || echo "")

    if [[ -n "${pod}" ]]; then
        kubectl exec -n victor "${pod}" -- env | grep -E "VICTOR_|DATABASE_|REDIS_" | sort | head -n20 || echo "  Failed to get environment"
    else
        echo "  No pods available"
    fi

    echo ""
    echo -e "${BOLD}Configuration Files${NC}"
    echo ""

    local config_dir="${DEPLOYMENT_ROOT}/kubernetes/overlays/${ENVIRONMENT}"

    if [[ -d "${config_dir}" ]]; then
        echo "  Configuration directory: ${config_dir}"
        echo ""
        ls -la "${config_dir}" | awk 'NR>1 {print "  " $9}' | grep -v "^$" || true
    else
        echo "  Configuration directory not found"
    fi

    wait_for_input
}

run_diagnostics() {
    print_header
    echo -e "${BOLD}Diagnostics${NC}"
    echo ""

    export KUBECONFIG="${DEPLOYMENT_ROOT}/kubeconfig_${ENVIRONMENT}"

    echo "Running diagnostic checks..."
    echo ""

    # Cluster connectivity
    echo -n "  Cluster connectivity... "
    if kubectl cluster-info &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi

    # Deployment status
    echo -n "  Deployment status... "
    local deployment_status
    deployment_status=$(kubectl get deployment victor -n victor -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "False")
    if [[ "${deployment_status}" == "True" ]]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi

    # Pod status
    echo -n "  Pod status... "
    local pods_running
    local pods_total
    pods_running=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | grep -c "Running" || echo "0")
    pods_total=$(kubectl get pods -n victor -l app=victor -o jsonpath='{.items[*].status.phase}' 2>/dev/null | wc -w || echo "0")
    if [[ "${pods_running}" -eq "${pods_total}" ]] && [[ "${pods_total}" -gt 0 ]]; then
        echo -e "${GREEN}OK${NC} (${pods_running}/${pods_total})"
    else
        echo -e "${YELLOW}WARNING${NC} (${pods_running}/${pods_total})"
    fi

    # Recent errors
    echo ""
    echo "Recent errors:"
    kubectl logs -n victor -l app=victor --tail=50 --all-containers=true 2>/dev/null | grep -i "error\|exception" | tail -n5 || echo "  No errors found"

    wait_for_input
}

################################################################################
# Main Execution
################################################################################

main() {
    ENVIRONMENT="${1:-production}"

    # Check dependencies
    if ! command -v kubectl &>/dev/null; then
        echo "Error: kubectl not found"
        exit 1
    fi

    # Main menu loop
    while true; do
        show_main_menu
    done
}

# Run main function
main "$@"
