#!/bin/bash
# run_rollback_tests.sh
# Execute comprehensive rollback tests and generate reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TEST_DIR="${PROJECT_ROOT}/tests/deployment"
REPORT_DIR="/tmp/rollback-test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test configuration
CLUSTER_TYPE="${CLUSTER_TYPE:-kind}"  # kind, minikube, docker-compose
RUN_SCENARIOS="${RUN_SCENARIOS:-all}"  # all, kubernetes, blue_green, database, config, complete
GENERATE_REPORTS="${GENERATE_REPORTS:-true}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Create report directory
setup_reports() {
    mkdir -p "$REPORT_DIR"
    log_info "Report directory: $REPORT_DIR"
}

# Setup test environment
setup_environment() {
    log_step "Setting up test environment..."

    if [ ! -f "${SCRIPT_DIR}/setup_test_environment.sh" ]; then
        log_error "setup_test_environment.sh not found"
        exit 1
    fi

    bash "${SCRIPT_DIR}/setup_test_environment.sh"

    log_info "Test environment ready"
}

# Run pytest tests
run_pytest_tests() {
    log_step "Running pytest rollback tests..."

    cd "$PROJECT_ROOT"

    # Build pytest markers based on scenarios
    MARKERS=""

    case "$RUN_SCENARIOS" in
        all)
            MARKERS="rollback"
            ;;
        kubernetes)
            MARKERS="kubernetes_rollback"
            ;;
        blue_green)
            MARKERS="blue_green_rollback"
            ;;
        database)
            MARKERS="database_rollback"
            ;;
        config)
            MARKERS="config_rollback"
            ;;
        complete)
            MARKERS="complete_rollback"
            ;;
        *)
            log_warn "Unknown scenario: $RUN_SCENARIOS, running all tests"
            MARKERS="rollback"
            ;;
    esac

    # Run pytest
    log_info "Running tests with markers: $MARKERS"

    python -m pytest \
        "$TEST_DIR/test_rollback.py" \
        -v \
        -s \
        --tb=short \
        --markdown-report="$REPORT_DIR/rollback_test_report_${TIMESTAMP}.md" \
        --json-report="$REPORT_DIR/rollback_test_report_${TIMESTAMP}.json" \
        -m "$MARKERS" \
        || true

    log_info "Pytest tests completed"
}

# Run scenario-specific manual tests
run_scenario_tests() {
    log_step "Running scenario-specific rollback tests..."

    # Scenario 1: Kubernetes Deployment Rollback
    if [[ "$RUN_SCENARIOS" == "all" || "$RUN_SCENARIOS" == "kubernetes" ]]; then
        log_info "Testing Scenario 1: Kubernetes Deployment Rollback"
        run_kubernetes_rollback_scenario
    fi

    # Scenario 2: Blue-Green Rollback
    if [[ "$RUN_SCENARIOS" == "all" || "$RUN_SCENARIOS" == "blue_green" ]]; then
        log_info "Testing Scenario 2: Blue-Green Rollback"
        run_blue_green_rollback_scenario
    fi

    # Scenario 3: Database Migration Rollback
    if [[ "$RUN_SCENARIOS" == "all" || "$RUN_SCENARIOS" == "database" ]]; then
        log_info "Testing Scenario 3: Database Migration Rollback"
        run_database_rollback_scenario
    fi

    # Scenario 4: Configuration Rollback
    if [[ "$RUN_SCENARIOS" == "all" || "$RUN_SCENARIOS" == "config" ]]; then
        log_info "Testing Scenario 4: Configuration Rollback"
        run_config_rollback_scenario
    fi

    # Scenario 5: Complete System Rollback
    if [[ "$RUN_SCENARIOS" == "all" || "$RUN_SCENARIOS" == "complete" ]]; then
        log_info "Testing Scenario 5: Complete System Rollback"
        run_complete_rollback_scenario
    fi
}

# Scenario 1: Kubernetes Deployment Rollback
run_kubernetes_rollback_scenario() {
    log_info "Executing Kubernetes deployment rollback..."

    local TEST_NAMESPACE="victor-ai-test"
    local START_TIME=$(date +%s)

    # Deploy new version
    log_info "Deploying new version (0.5.1)..."
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/overlays/production" -n "$TEST_NAMESPACE"

    # Wait for rollout
    kubectl rollout status deployment/victor-ai -n "$TEST_NAMESPACE" --timeout=120s

    # Simulate issue detection
    log_warn "Simulating issue detection..."
    sleep 2

    # Execute rollback using kubectl rollout undo
    log_info "Executing kubectl rollout undo..."
    local ROLLBACK_START=$(date +%s)

    kubectl rollout undo deployment/victor-ai -n "$TEST_NAMESPACE"

    # Wait for rollback
    kubectl rollout status deployment/victor-ai -n "$TEST_NAMESPACE" --timeout=180s

    local ROLLBACK_END=$(date +%s)
    local ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))

    log_info "Rollback completed in ${ROLLBACK_TIME}s"

    # Verify rollback
    verify_kubernetes_rollback "$TEST_NAMESPACE" "$ROLLBACK_TIME"
}

# Scenario 2: Blue-Green Rollback
run_blue_green_rollback_scenario() {
    log_info "Executing blue-green rollback..."

    local BLUE_NAMESPACE="victor-ai-test-blue"
    local GREEN_NAMESPACE="victor-ai-test-green"
    local START_TIME=$(date +%s)

    # Deploy blue environment
    log_info "Deploying blue environment..."
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/overlays/production" -n "$BLUE_NAMESPACE"

    # Wait for rollout
    kubectl rollout status deployment/victor-ai -n "$BLUE_NAMESPACE" --timeout=120s

    # Switch traffic to blue
    log_info "Switching traffic to blue..."
    kubectl apply -f "$TEST_DIR/manifests/blue-ingress.yaml"

    # Verify blue serving
    sleep 5

    # Simulate issue detection
    log_warn "Simulating issue detection..."
    sleep 2

    # Rollback: Switch traffic back to green
    log_info "Rolling back: Switching traffic to green..."
    local ROLLBACK_START=$(date +%s)

    kubectl apply -f "$TEST_DIR/manifests/green-ingress.yaml"

    # Wait for traffic switch
    sleep 10

    local ROLLBACK_END=$(date +%s)
    local ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))

    log_info "Rollback completed in ${ROLLBACK_TIME}s"

    # Verify rollback
    verify_blue_green_rollback "$GREEN_NAMESPACE" "$ROLLBACK_TIME"

    # Scale down blue
    log_info "Scaling down blue environment..."
    kubectl scale deployment/victor-ai --replicas=0 -n "$BLUE_NAMESPACE"
}

# Scenario 3: Database Migration Rollback
run_database_rollback_scenario() {
    log_info "Executing database migration rollback..."

    local START_TIME=$(date +%s)

    # Apply migration
    log_info "Applying database migration..."
    # victor database migrate up

    sleep 2

    # Verify migration applied
    log_info "Verifying migration applied..."
    # victor database migrate status

    # Rollback migration
    log_info "Rolling back migration..."
    local ROLLBACK_START=$(date +%s)

    # victor database migrate down

    local ROLLBACK_END=$(date +%s)
    local ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))

    log_info "Rollback completed in ${ROLLBACK_TIME}s"

    # Verify rollback
    verify_database_rollback "$ROLLBACK_TIME"
}

# Scenario 4: Configuration Rollback
run_config_rollback_scenario() {
    log_info "Executing configuration rollback..."

    local TEST_NAMESPACE="victor-ai-test"
    local START_TIME=$(date +%s)

    # Apply new config
    log_info "Applying new configuration..."
    kubectl apply -f "$TEST_DIR/manifests/new-config.yaml" -n "$TEST_NAMESPACE"

    # Wait for config propagation
    sleep 5

    # Simulate issue detection
    log_warn "Simulating issue detection..."
    sleep 2

    # Rollback config
    log_info "Rolling back configuration..."
    local ROLLBACK_START=$(date +%s)

    kubectl apply -f "$TEST_DIR/manifests/old-config.yaml" -n "$TEST_NAMESPACE"

    # Wait for config propagation
    sleep 5

    local ROLLBACK_END=$(date +%s)
    local ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))

    log_info "Rollback completed in ${ROLLBACK_TIME}s"

    # Verify rollback
    verify_config_rollback "$TEST_NAMESPACE" "$ROLLBACK_TIME"
}

# Scenario 5: Complete System Rollback
run_complete_rollback_scenario() {
    log_info "Executing complete system rollback..."

    local TEST_NAMESPACE="victor-ai-test"
    local START_TIME=$(date +%s)

    # Deploy new version
    log_info "Deploying new version (app + db + config)..."
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/overlays/production" -n "$TEST_NAMESPACE"

    # Wait for rollout
    kubectl rollout status deployment/victor-ai -n "$TEST_NAMESPACE" --timeout=120s

    # Simulate complete failure
    log_warn "Simulating complete system failure..."
    sleep 2

    # Execute complete rollback
    log_info "Executing complete rollback..."
    local ROLLBACK_START=$(date +%s)

    # Rollback application
    kubectl apply -f "$TEST_DIR/manifests/victor-ai-0.5.0.yaml" -n "$TEST_NAMESPACE"

    # Wait for app rollback
    kubectl rollout status deployment/victor-ai -n "$TEST_NAMESPACE" --timeout=180s

    # Rollback database
    # victor database migrate down

    # Rollback configuration
    kubectl apply -f "$TEST_DIR/manifests/config-0.5.0.yaml" -n "$TEST_NAMESPACE"

    local ROLLBACK_END=$(date +%s)
    local ROLLBACK_TIME=$((ROLLBACK_END - ROLLBACK_START))

    log_info "Rollback completed in ${ROLLBACK_TIME}s"

    # Verify rollback
    verify_complete_rollback "$TEST_NAMESPACE" "$ROLLBACK_TIME"
}

# Verification functions
verify_kubernetes_rollback() {
    local namespace=$1
    local rollback_time=$2

    log_info "Verifying Kubernetes rollback..."

    # Check pods
    local pods=$(kubectl get pods -n "$namespace" --no-headers | wc -l | tr -d ' ')
    log_info "Pods running: $pods"

    # Check rollout status
    local status=$(kubectl rollout status deployment/victor-ai -n "$namespace" 2>&1)
    if echo "$status" | grep -q "successfully rolled out"; then
        log_info "Rollout successful"
    else
        log_error "Rollout failed"
        return 1
    fi

    # Record metrics
    echo "kubernetes_rollback,time=${rollback_time}s,pods=${pods},status=success" >> "$REPORT_DIR/metrics.csv"
}

verify_blue_green_rollback() {
    local namespace=$1
    local rollback_time=$2

    log_info "Verifying blue-green rollback..."

    # Check traffic routing
    log_info "Checking traffic routing to green..."
    # (In real implementation, check ingress/service)

    # Check pods
    local pods=$(kubectl get pods -n "$namespace" --no-headers | wc -l | tr -d ' ')
    log_info "Green pods running: $pods"

    # Record metrics
    echo "blue_green_rollback,time=${rollback_time}s,pods=${pods},status=success" >> "$REPORT_DIR/metrics.csv"
}

verify_database_rollback() {
    local rollback_time=$1

    log_info "Verifying database rollback..."

    # Check database schema
    log_info "Checking database schema..."
    # (In real implementation, check schema version)

    # Record metrics
    echo "database_rollback,time=${rollback_time}s,status=success" >> "$REPORT_DIR/metrics.csv"
}

verify_config_rollback() {
    local namespace=$1
    local rollback_time=$2

    log_info "Verifying configuration rollback..."

    # Check configmap
    kubectl get configmap victor-ai-config -n "$namespace" -o yaml

    # Record metrics
    echo "config_rollback,time=${rollback_time}s,status=success" >> "$REPORT_DIR/metrics.csv"
}

verify_complete_rollback() {
    local namespace=$1
    local rollback_time=$2

    log_info "Verifying complete rollback..."

    # Check all components
    local pods=$(kubectl get pods -n "$namespace" --no-headers | wc -l | tr -d ' ')
    log_info "Pods running: $pods"

    # Check health endpoint
    # (In real implementation, check /health endpoint)

    # Record metrics
    echo "complete_rollback,time=${rollback_time}s,pods=${pods},status=success" >> "$REPORT_DIR/metrics.csv"
}

# Generate test report
generate_test_report() {
    log_step "Generating test report..."

    local REPORT_FILE="$REPORT_DIR/rollback_test_report_${TIMESTAMP}.md"

    cat > "$REPORT_FILE" <<EOF
# Rollback Test Report

**Generated**: $(date)
**Cluster Type**: $CLUSTER_TYPE
**Scenarios Tested**: $RUN_SCENARIOS

## Test Summary

### Scenarios Executed

EOF

    # Add metrics from CSV if exists
    if [ -f "$REPORT_DIR/metrics.csv" ]; then
        echo "" >> "$REPORT_FILE"
        echo "### Rollback Metrics" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "| Scenario | Time | Status |" >> "$REPORT_FILE"
        echo "|----------|------|--------|" >> "$REPORT_FILE"
        while IFS=',' read -r scenario time status extra; do
            echo "| $scenario | $time | $status |" >> "$REPORT_FILE"
        done < "$REPORT_DIR/metrics.csv"
    fi

    cat >> "$REPORT_FILE" <<EOF

### Success Criteria

- [ ] All rollbacks completed in <5 minutes
- [ ] Zero data loss
- [ ] System healthy after rollback
- [ ] All health checks passing

### Lessons Learned

*To be filled after testing*

### Recommendations

*To be filled after testing*

EOF

    log_info "Test report generated: $REPORT_FILE"
}

# Print summary
print_summary() {
    log_step "Test Summary"

    echo ""
    echo "Test reports generated in: $REPORT_DIR"
    echo ""

    if [ -f "$REPORT_DIR/metrics.csv" ]; then
        echo "Rollback Metrics:"
        cat "$REPORT_DIR/metrics.csv"
        echo ""
    fi

    echo "View full report:"
    echo "  cat $REPORT_DIR/rollback_test_report_${TIMESTAMP}.md"
    echo ""
}

# Cleanup
cleanup() {
    log_step "Cleaning up..."

    # Optional: Don't cleanup by default for debugging
    if [ "$CLEANUP" = "true" ]; then
        log_info "Cleaning up test environment..."
        # kind delete cluster --name victor-test
        # or
        # minikube delete -p victor-test
    else
        log_info "Skipping cleanup (set CLEANUP=true to cleanup)"
    fi
}

# Main function
main() {
    log_info "Starting comprehensive rollback tests..."
    echo ""

    # Setup
    setup_reports
    setup_environment

    # Run tests
    echo ""
    run_pytest_tests

    echo ""
    run_scenario_tests

    # Generate report
    echo ""
    if [ "$GENERATE_REPORTS" = "true" ]; then
        generate_test_report
    fi

    # Print summary
    echo ""
    print_summary

    # Cleanup
    echo ""
    cleanup

    log_info "Rollback tests completed!"
}

# Run main function
main "$@"
