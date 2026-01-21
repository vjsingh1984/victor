#!/bin/bash
#############################################################################
# Health Check Script for Victor AI
#
# Features:
# - API health endpoint
# - Database connectivity
# - Provider availability
# - Critical service checks
#
# Usage: ./health_check.sh [--endpoint URL] [--timeout SECONDS]
#############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-http://localhost:8000/health}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"
VERBOSE=false
EXIT_ON_FAILURE=true

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_CHECKS++)) || true
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_CHECKS++)) || true
    if [ "$EXIT_ON_FAILURE" = true ]; then
        exit 1
    fi
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

#############################################################################
# Parse Arguments
#############################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --endpoint)
                HEALTH_ENDPOINT="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-exit-on-failure)
                EXIT_ON_FAILURE=false
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --endpoint URL              Health check endpoint (default: http://localhost:8000/health)"
                echo "  --timeout SECONDS           Request timeout (default: 10)"
                echo "  --verbose                   Show verbose output"
                echo "  --no-exit-on-failure       Continue on failure (default: exit on first failure)"
                echo "  --help                      Show this help message"
                echo ""
                echo "Exit codes:"
                echo "  0   All health checks passed"
                echo "  1   One or more health checks failed"
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
# HTTP Health Check
#############################################################################
check_http_endpoint() {
    ((TOTAL_CHECKS++)) || true
    log_info "Checking HTTP endpoint: ${HEALTH_ENDPOINT}"

    local response
    local http_code

    response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "${HEALTH_ENDPOINT}" 2>&1) || {
        log_error "HTTP endpoint is not reachable"
        return 1
    }

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    log_verbose "HTTP Status Code: $http_code"
    log_verbose "Response Body: $body"

    if [ "$http_code" = "200" ]; then
        log_success "HTTP endpoint is healthy (200 OK)"

        # Parse JSON response if available
        if command -v jq &> /dev/null; then
            local status
            status=$(echo "$body" | jq -r '.status // "unknown"')
            log_verbose "Health Status: $status"

            if [ "$status" = "healthy" ]; then
                log_success "Application status: healthy"
            else
                log_warning "Application status: $status"
            fi
        fi
    else
        log_error "HTTP endpoint returned status code: $http_code"
    fi
}

#############################################################################
# Service Availability Check
#############################################################################
check_service_availability() {
    ((TOTAL_CHECKS++)) || true
    log_info "Checking service availability..."

    # Check if port is listening
    local host
    local port

    # Extract host and port from endpoint
    host=$(echo "$HEALTH_ENDPOINT" | sed -n 's|.*//\([^:/]*\).*|\1|p')
    port=$(echo "$HEALTH_ENDPOINT" | sed -n 's|.*:\([0-9]*\).*|\1|p')

    log_verbose "Host: $host, Port: $port"

    if command -v nc &> /dev/null; then
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "Service is listening on ${host}:${port}"
        else
            log_error "Service is not listening on ${host}:${port}"
        fi
    elif command -v bash &> /dev/null; then
        # Fallback to bash
        timeout 1 bash -c "cat < /dev/null > /dev/tcp/${host}/${port}" 2>/dev/null && {
            log_success "Service is listening on ${host}:${port}"
        } || {
            log_error "Service is not listening on ${host}:${port}"
        }
    fi
}

#############################################################################
# Database Connectivity Check
#############################################################################
check_database() {
    ((TOTAL_CHECKS++)) || true
    log_info "Checking database connectivity..."

    # Try to access database health endpoint
    local db_endpoint
    db_endpoint=$(echo "$HEALTH_ENDPOINT" | sed 's|/health|/health/db|')

    local response
    response=$(curl -s --max-time "$TIMEOUT" "${db_endpoint}" 2>&1) || {
        log_warning "Database health endpoint not available"
        return
    }

    if command -v jq &> /dev/null; then
        local db_status
        db_status=$(echo "$response" | jq -r '.status // "unknown"')

        if [ "$db_status" = "healthy" ]; then
            log_success "Database connectivity: OK"
        else
            log_error "Database connectivity: $db_status"
        fi
    else
        # Fallback: check if response contains "healthy"
        if echo "$response" | grep -q "healthy"; then
            log_success "Database connectivity: OK"
        else
            log_warning "Database health status unclear"
        fi
    fi
}

#############################################################################
# Provider Health Check
#############################################################################
check_providers() {
    ((TOTAL_CHECKS++)) || true
    log_info "Checking provider availability..."

    local providers_endpoint
    providers_endpoint=$(echo "$HEALTH_ENDPOINT" | sed 's|/health|/health/providers|')

    local response
    response=$(curl -s --max-time "$TIMEOUT" "${providers_endpoint}" 2>&1) || {
        log_warning "Providers health endpoint not available"
        return
    }

    if command -v jq &> /dev/null; then
        local available_providers
        available_providers=$(echo "$response" | jq -r '.providers // [] | length')

        if [ "$available_providers" -gt 0 ]; then
            log_success "Available providers: $available_providers"

            log_verbose "Provider details:"
            echo "$response" | jq -r '.providers[]? // "  - Unknown"' | sed 's/^/  /' | head -n 5
        else
            log_warning "No providers available"
        fi
    else
        # Fallback
        if echo "$response" | grep -q "providers"; then
            log_success "Provider health check passed"
        else
            log_warning "Provider health status unclear"
        fi
    fi
}

#############################################################################
# Memory and CPU Check
#############################################################################
check_resources() {
    ((TOTAL_CHECKS++)) || true
    log_info "Checking system resources..."

    local resources_endpoint
    resources_endpoint=$(echo "$HEALTH_ENDPOINT" | sed 's|/health|/health/resources|')

    local response
    response=$(curl -s --max-time "$TIMEOUT" "${resources_endpoint}" 2>&1) || {
        log_warning "Resources health endpoint not available"
        return
    }

    if command -v jq &> /dev/null; then
        local memory_usage
        local cpu_usage

        memory_usage=$(echo "$response" | jq -r '.memory_percent // -1')
        cpu_usage=$(echo "$response" | jq -r '.cpu_percent // -1')

        log_verbose "Memory Usage: ${memory_usage}%"
        log_verbose "CPU Usage: ${cpu_usage}%"

        # Check thresholds
        if [ "$memory_usage" != "-1" ] && [ "$memory_usage" -lt 90 ]; then
            log_success "Memory usage: ${memory_usage}%"
        elif [ "$memory_usage" != "-1" ]; then
            log_warning "High memory usage: ${memory_usage}%"
        else
            log_warning "Memory usage not available"
        fi

        if [ "$cpu_usage" != "-1" ] && [ "$cpu_usage" -lt 90 ]; then
            log_success "CPU usage: ${cpu_usage}%"
        elif [ "$cpu_usage" != "-1" ]; then
            log_warning "High CPU usage: ${cpu_usage}%"
        else
            log_warning "CPU usage not available"
        fi
    fi
}

#############################################################################
# Critical Services Check
#############################################################################
check_critical_services() {
    ((TOTAL_CHECKS++)) || true
    log_info "Checking critical services..."

    local critical_endpoint
    critical_endpoint=$(echo "$HEALTH_ENDPOINT" | sed 's|/health|/health/critical|')

    local response
    response=$(curl -s --max-time "$TIMEOUT" "${critical_endpoint}" 2>&1) || {
        log_warning "Critical services endpoint not available"
        return
    }

    if command -v jq &> /dev/null; then
        local services
        services=$(echo "$response" | jq -r '.services // {}')

        if [ "$services" != "{}" ]; then
            log_info "Critical Services Status:"
            echo "$response" | jq -r '.services | to_entries[] | "  \(.key): \(.value.status)"' | while read -r line; do
                if echo "$line" | grep -q "healthy"; then
                    log_success "$line"
                else
                    log_error "$line"
                fi
            done
        else
            log_success "All critical services operational"
        fi
    fi
}

#############################################################################
# Print Summary
#############################################################################
print_summary() {
    echo ""
    echo "=========================================="
    echo "Health Check Summary"
    echo "=========================================="
    echo "Total Checks:   $TOTAL_CHECKS"
    echo "Passed:         ${GREEN}${PASSED_CHECKS}${NC}"
    echo "Failed:         ${RED}${FAILED_CHECKS}${NC}"
    echo "=========================================="
    echo ""

    if [ $FAILED_CHECKS -eq 0 ]; then
        log_success "All health checks passed!"
        exit 0
    else
        log_error "Some health checks failed!"
        exit 1
    fi
}

#############################################################################
# Main Health Check Flow
#############################################################################
main() {
    log_info "Starting health checks for Victor AI..."
    log_info "Endpoint: ${HEALTH_ENDPOINT}"
    log_info "Timeout: ${TIMEOUT}s"
    echo ""

    # Parse arguments
    parse_args "$@"

    # Run health checks
    check_service_availability
    check_http_endpoint
    check_database
    check_providers
    check_resources
    check_critical_services

    # Print summary
    print_summary
}

# Run main health check
main "$@"
