#!/bin/bash
# Victor AI 0.5.1 - Configuration Drift Detection Script
#
# This script compares configurations between environments to detect
# unexpected configuration drift. It identifies:
# - Unexpected differences in resource allocation
# - Missing environment variables
# - Configuration inconsistencies
# - Unauthorized changes
#
# Usage: ./detect_drift.sh [--env1 <env>] [--env2 <env>] [--output <file>]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
ENV1="staging"
ENV2="production"
OUTPUT_FILE=""
DRIFT_COUNT=0
DRIFT_REPORT=""

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[DRIFT]${NC} $1"
    ((DRIFT_COUNT++))
}

log_section() {
    echo ""
    echo -e "${CYAN}================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================${NC}"
    echo ""
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env1)
                ENV1="$2"
                shift 2
                ;;
            --env2)
                ENV2="$2"
                shift 2
                ;;
            --output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [--env1 <env>] [--env2 <env>] [--output <file>]"
                echo ""
                echo "Options:"
                echo "  --env1       First environment to compare (default: staging)"
                echo "  --env2       Second environment to compare (default: production)"
                echo "  --output     Save report to file"
                echo "  --help, -h   Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                                    # Compare staging vs production"
                echo "  $0 --env1 development --env2 staging  # Compare development vs staging"
                echo "  $0 --output drift_report.txt          # Save report to file"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate environment names
validate_environments() {
    valid_envs=("development" "testing" "staging" "production")

    for env in "$ENV1" "$ENV2"; do
        if [[ ! " ${valid_envs[@]} " =~ " ${env} " ]]; then
            log_error "Invalid environment: $env"
            echo "Valid environments: ${valid_envs[*]}"
            exit 1
        fi
    done
}

# Get overlay directory for environment
get_overlay_dir() {
    local env=$1
    echo "/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays/$env"
}

# Extract value from YAML
extract_yaml_value() {
    local file=$1
    local path=$2

    if command -v yq &> /dev/null; then
        yq eval "$path" "$file" 2>/dev/null || echo ""
    else
        # Fallback: grep-based extraction (less reliable)
        echo ""
    fi
}

# Compare resource allocations
compare_resources() {
    log_section "Comparing Resource Allocations"

    local overlay1=$(get_overlay_dir "$ENV1")
    local overlay2=$(get_overlay_dir "$ENV2")

    local patch1="$overlay1/deployment-patch.yaml"
    local patch2="$overlay2/deployment-patch.yaml"

    if [ ! -f "$patch1" ]; then
        log_warning "Deployment patch not found for $ENV1: $patch1"
        return
    fi

    if [ ! -f "$patch2" ]; then
        log_warning "Deployment patch not found for $ENV2: $patch2"
        return
    fi

    # Extract resources using Python/yq (or fallback to grep)
    python3 <<EOF
import yaml
import sys

try:
    with open('$patch1') as f:
        config1 = yaml.safe_load(f)
    with open('$patch2') as f:
        config2 = yaml.safe_load(f)

    # Extract resources
    resources1 = config1.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [{}])[0].get('resources', {})
    resources2 = config2.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [{}])[0].get('resources', {})

    print(f"CPU Request ($ENV1):  {resources1.get('requests', {}).get('cpu', 'N/A')}")
    print(f"CPU Request ($ENV2):  {resources2.get('requests', {}).get('cpu', 'N/A')}")
    print(f"CPU Limit ($ENV1):    {resources1.get('limits', {}).get('cpu', 'N/A')}")
    print(f"CPU Limit ($ENV2):    {resources2.get('limits', {}).get('cpu', 'N/A')}")
    print(f"Memory Request ($ENV1):  {resources1.get('requests', {}).get('memory', 'N/A')}")
    print(f"Memory Request ($ENV2):  {resources2.get('requests', {}).get('memory', 'N/A')}")
    print(f"Memory Limit ($ENV1):    {resources1.get('limits', {}).get('memory', 'N/A')}")
    print(f"Memory Limit ($ENV2):    {resources2.get('limits', {}).get('memory', 'N/A')}")

    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    echo ""
    log_info "Resource comparison complete"
}

# Compare environment variables
compare_environment_variables() {
    log_section "Comparing Environment Variables"

    local overlay1=$(get_overlay_dir "$ENV1")
    local overlay2=$(get_overlay_dir "$ENV2")

    local kustomization1="$overlay1/kustomization.yaml"
    local kustomization2="$overlay2/kustomization.yaml"

    if [ ! -f "$kustomization1" ]; then
        log_warning "Kustomization not found for $ENV1: $kustomization1"
        return
    fi

    if [ ! -f "$kustomization2" ]; then
        log_warning "Kustomization not found for $ENV2: $kustomization2"
        return
    fi

    # Extract and compare environment variables
    python3 <<EOF
import yaml

try:
    with open('$kustomization1') as f:
        config1 = yaml.safe_load(f)
    with open('$kustomization2') as f:
        config2 = yaml.safe_load(f)

    # Extract literals from configMapGenerator
    literals1 = config1.get('configMapGenerator', [{}])[0].get('literals', [])
    literals2 = config2.get('configMapGenerator', [{}])[0].get('literals', [])

    # Parse into dictionaries
    env1 = {}
    for lit in literals1:
        if '=' in lit:
            key, value = lit.split('=', 1)
            env1[key] = value

    env2 = {}
    for lit in literals2:
        if '=' in lit:
            key, value = lit.split('=', 1)
            env2[key] = value

    # Find all keys
    all_keys = sorted(set(env1.keys()) | set(env2.keys()))

    # Compare
    print("Environment Variable Comparison:")
    print("-" * 80)
    print(f"{'Variable':<40} {'$ENV1':<20} {'$ENV2':<20} {'Status':<10}")
    print("-" * 80)

    drift_found = False
    for key in all_keys:
        val1 = env1.get(key, 'NOT SET')
        val2 = env2.get(key, 'NOT SET')

        if val1 == val2:
            status = "✓ OK"
        else:
            status = "⚠ DRIFT"
            drift_found = True

        print(f"{key:<40} {val1:<20} {val2:<20} {status:<10}")

    print("-" * 80)

    if drift_found:
        print("\n⚠ Configuration drift detected in environment variables")
    else:
        print("\n✓ No drift detected in environment variables")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
EOF

    echo ""
}

# Compare replicas
compare_replicas() {
    log_section "Comparing Replica Counts"

    local overlay1=$(get_overlay_dir "$ENV1")
    local overlay2=$(get_overlay_dir "$ENV2")

    local kustomization1="$overlay1/kustomization.yaml"
    local kustomization2="$overlay2/kustomization.yaml"

    if [ ! -f "$kustomization1" ] || [ ! -f "$kustomization2" ]; then
        log_warning "Cannot compare replicas (kustomization files not found)"
        return
    fi

    python3 <<EOF
import yaml

try:
    with open('$kustomization1') as f:
        config1 = yaml.safe_load(f)
    with open('$kustomization2') as f:
        config2 = yaml.safe_load(f)

    replicas1 = config1.get('replicas', [{}])[0].get('count', 'N/A')
    replicas2 = config2.get('replicas', [{}])[0].get('count', 'N/A')

    print(f"Replicas ($ENV1):  {replicas1}")
    print(f"Replicas ($ENV2):  {replicas2}")

    if replicas1 != replicas2:
        if replicas1 < replicas2:
            print("✓ Expected: $ENV2 has more replicas than $ENV1")
        else:
            print("⚠ WARNING: $ENV1 has more replicas than $ENV2")
    else:
        print("⚠ WARNING: Both environments have the same replica count")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
EOF

    echo ""
}

# Compare image tags
compare_image_tags() {
    log_section "Comparing Image Tags"

    local overlay1=$(get_overlay_dir "$ENV1")
    local overlay2=$(get_overlay_dir "$ENV2")

    local kustomization1="$overlay1/kustomization.yaml"
    local kustomization2="$overlay2/kustomization.yaml"

    if [ ! -f "$kustomization1" ] || [ ! -f "$kustomization2" ]; then
        log_warning "Cannot compare image tags (kustomization files not found)"
        return
    fi

    python3 <<EOF
import yaml

try:
    with open('$kustomization1') as f:
        config1 = yaml.safe_load(f)
    with open('$kustomization2') as f:
        config2 = yaml.safe_load(f)

    images1 = config1.get('images', [])
    images2 = config2.get('images', [])

    tag1 = images1[0].get('newTag', 'latest') if images1 else 'latest'
    tag2 = images2[0].get('newTag', 'latest') if images2 else 'latest'

    print(f"Image Tag ($ENV1):  {tag1}")
    print(f"Image Tag ($ENV2):  {tag2}")

    if tag1 == tag2:
        print("⚠ WARNING: Both environments are using the same image tag")
        print("  This may indicate unexpected synchronization or drift")
    else:
        print("✓ Different image tags (expected)")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
EOF

    echo ""
}

# Check for missing configurations
check_missing_configs() {
    log_section "Checking for Missing Configurations"

    local envs=("development" "testing" "staging" "production")

    for env in "${envs[@]}"; do
        overlay_dir=$(get_overlay_dir "$env")

        if [ ! -d "$overlay_dir" ]; then
            log_error "Missing overlay directory for $env"
            continue
        fi

        # Check for required files
        required_files=("kustomization.yaml" "deployment-patch.yaml")

        for file in "${required_files[@]}"; do
            if [ ! -f "$overlay_dir/$file" ]; then
                log_error "Missing required file for $env: $file"
            else
                log_success "Required file exists for $env: $file"
            fi
        done

        # Check for HPA patch in production
        if [ "$env" == "production" ]; then
            if [ ! -f "$overlay_dir/hpa-patch.yaml" ]; then
                log_error "Missing HPA patch for production"
            else
                log_success "HPA patch exists for production"
            fi
        fi
    done

    echo ""
}

# Check backend configuration progression
check_backend_progression() {
    log_section "Checking Backend Configuration Progression"

    expected_backends=(
        "development:memory:memory:memory"
        "testing:sqlite:memory:memory"
        "staging:sqlite:memory:memory"
        "production:postgres:redis:kafka"
    )

    python3 <<EOF
import yaml
import sys

overlay_base = '/Users/vijaysingh/code/codingagent/deployment/kubernetes/overlays'
expected = {
    'development': {'checkpoint': 'memory', 'cache': 'memory', 'event-bus': 'memory'},
    'testing': {'checkpoint': 'sqlite', 'cache': 'memory', 'event-bus': 'memory'},
    'staging': {'checkpoint': 'sqlite', 'cache': 'memory', 'event-bus': 'memory'},
    'production': {'checkpoint': 'postgres', 'cache': 'redis', 'event-bus': 'kafka'},
}

print("Backend Configuration Progression:")
print("-" * 80)
print(f"{'Environment':<15} {'Checkpoint':<15} {'Cache':<15} {'Event Bus':<15} {'Status':<10}")
print("-" * 80)

for env, backends in expected.items():
    kustomization = f'{overlay_base}/{env}/kustomization.yaml'

    try:
        with open(kustomization) as f:
            config = yaml.safe_load(f)

        literals = config.get('configMapGenerator', [{}])[0].get('literals', [])

        actual = {}
        for lit in literals:
            if lit.startswith('checkpoint-backend='):
                actual['checkpoint'] = lit.split('=')[1]
            elif lit.startswith('cache-backend='):
                actual['cache'] = lit.split('=')[1]
            elif lit.startswith('event-bus-backend='):
                actual['event-bus'] = lit.split('=')[1]

        checkpoint_status = "✓" if actual.get('checkpoint') == backends['checkpoint'] else "⚠"
        cache_status = "✓" if actual.get('cache') == backends['cache'] else "⚠"
        eventbus_status = "✓" if actual.get('event-bus') == backends['event-bus'] else "⚠"

        overall_status = "✓ OK" if all([
            actual.get('checkpoint') == backends['checkpoint'],
            actual.get('cache') == backends['cache'],
            actual.get('event-bus') == backends['event-bus']
        ]) else "⚠ DRIFT"

        print(f"{env:<15} {actual.get('checkpoint', 'N/A'):<15} {actual.get('cache', 'N/A'):<15} {actual.get('event-bus', 'N/A'):<15} {overall_status:<10}")

    except Exception as e:
        print(f"{env:<15} Error: {e}")

print("-" * 80)
EOF

    echo ""
}

# Generate drift report
generate_report() {
    local report_file="${OUTPUT_FILE:-/tmp/victor_drift_report_${ENV1}_vs_${ENV2}_$(date +%Y%m%d_%H%M%S).txt}"

    {
        echo "================================================"
        echo "Victor AI 0.5.1 - Configuration Drift Report"
        echo "================================================"
        echo ""
        echo "Comparison: $ENV1 vs $ENV2"
        echo "Timestamp: $(date)"
        echo ""
        echo "Drift Detected: $DRIFT_COUNT issue(s)"
        echo ""
        echo "================================================"
        echo "Recommendations"
        echo "================================================"
        echo ""
        if [ $DRIFT_COUNT -eq 0 ]; then
            echo "✓ No unexpected configuration drift detected"
            echo ""
            echo "Configuration differences appear to be intentional"
            echo "and aligned with environment-specific requirements."
        else
            echo "⚠ Configuration drift detected between environments"
            echo ""
            echo "Recommended actions:"
            echo "  1. Review the drift items above"
            echo "  2. Determine if differences are intentional"
            echo "  3. Synchronize configurations if needed"
            echo "  4. Update environment templates"
            echo "  5. Re-run this check after changes"
        fi
        echo ""
        echo "================================================"
    } > "$report_file"

    DRIFT_REPORT="$report_file"
    log_info "Drift report saved to: $report_file"
}

# Print summary
print_summary() {
    echo ""
    echo "================================================"
    echo "Drift Detection Summary"
    echo "================================================"
    echo ""
    echo "Environments Compared: $ENV1 vs $ENV2"
    echo "Drift Issues Found:   $DRIFT_COUNT"
    echo "Report Saved:         $DRIFT_REPORT"
    echo ""

    if [ $DRIFT_COUNT -eq 0 ]; then
        log_success "No unexpected configuration drift detected"
        exit 0
    else
        log_error "Configuration drift detected"
        exit 1
    fi
}

# Main function
main() {
    echo "================================================"
    echo "Victor AI 0.5.1 - Configuration Drift Detection"
    echo "================================================"
    echo ""

    # Parse arguments
    parse_args "$@"

    # Validate environments
    validate_environments

    # Print comparison info
    log_info "Comparing configurations: $ENV1 → $ENV2"
    echo ""

    # Run comparisons
    compare_replicas
    compare_resources
    compare_environment_variables
    compare_image_tags
    check_missing_configs
    check_backend_progression

    # Generate report
    generate_report

    # Print summary
    print_summary
}

# Run main function
main "$@"
