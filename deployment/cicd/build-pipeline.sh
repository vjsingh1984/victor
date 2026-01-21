#!/bin/bash
# Victor AI Build Pipeline Script
# This script handles the complete build, test, and deployment process

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VERSION="${VERSION:-0.5.1}"
REGISTRY="${REGISTRY:-docker.io}"
IMAGE_NAME="${IMAGE_NAME:-victorai/victor}"
BUILD_ENV="${BUILD_ENV:-staging}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    local deps=("docker" "kubectl" "python3" "git")

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "Missing dependency: $dep"
            exit 1
        fi
    done

    log_info "All dependencies found"
}

setup_environment() {
    log_info "Setting up environment..."
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    log_info "Environment: ${BUILD_ENV}"
}

lint_code() {
    log_info "Running linters..."

    log_info "Checking code with ruff..."
    ruff check victor tests

    log_info "Checking code formatting with black..."
    black --check victor tests

    log_info "Type checking with mypy..."
    mypy victor

    log_info "Linting complete"
}

run_tests() {
    log_info "Running tests..."

    log_info "Running unit tests..."
    pytest tests/unit -v --cov=victor --cov-report=xml --cov-report=html

    log_info "Running integration tests..."
    pytest tests/integration -v --timeout=300

    log_info "All tests passed"
}

security_scan() {
    log_info "Running security scans..."

    # Trivy filesystem scan
    if command -v trivy &> /dev/null; then
        log_info "Scanning with Trivy..."
        trivy fs --severity HIGH,CRITICAL .
    else
        log_warn "Trivy not found, skipping filesystem scan"
    fi

    # Bandit security linter
    log_info "Scanning with Bandit..."
    pip install bandit[toml]
    bandit -r victor -f json -o bandit-report.json || true

    log_info "Security scanning complete"
}

build_image() {
    log_info "Building Docker image..."

    local build_args=(
        --build-arg VERSION="${VERSION}"
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        --build-arg VCS_REF="$(git rev-parse --short HEAD)"
        --file ./deployment/docker/Dockerfile
        --target slim
        --tag "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        --tag "${REGISTRY}/${IMAGE_NAME}:latest"
    )

    # Add platform for production builds
    if [ "${BUILD_ENV}" = "production" ]; then
        build_args+=(--platform linux/amd64,linux/arm64)
    fi

    docker buildx build \
        "${build_args[@]}" \
        --push \
        .

    log_info "Docker image built and pushed: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
}

scan_image() {
    log_info "Scanning Docker image..."

    if command -v trivy &> /dev/null; then
        trivy image "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    else
        log_warn "Trivy not found, skipping image scan"
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes (${BUILD_ENV})..."

    local namespace="victor-ai-${BUILD_ENV}"
    local overlay_dir="deployment/kubernetes/overlays/${BUILD_ENV}"

    if [ ! -d "${overlay_dir}" ]; then
        log_error "Kubernetes overlay not found: ${overlay_dir}"
        exit 1
    fi

    # Apply manifests
    kubectl apply -k "${overlay_dir}"

    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/victor-ai -n "${namespace}" --timeout=10m

    # Verify deployment
    log_info "Verifying deployment..."
    kubectl run smoke-test \
        --image=curlimages/curl:latest \
        --rm -i --restart=Never \
        -- curl -f "http://victor-ai.${namespace}.svc.cluster.local/health"

    log_info "Deployment successful"
}

run_health_checks() {
    log_info "Running health checks..."

    local namespace="victor-ai-${BUILD_ENV}"

    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "${namespace}" -l app=victor-ai

    # Check service endpoints
    log_info "Checking service endpoints..."
    kubectl get endpoints -n "${namespace}" -l app=victor-ai

    # Check logs
    log_info "Checking logs..."
    kubectl logs -n "${namespace}" -l app=victor-ai --tail=50

    log_info "Health checks complete"
}

rollback() {
    log_error "Initiating rollback..."
    local namespace="victor-ai-${BUILD_ENV}"

    kubectl rollout undo deployment/victor-ai -n "${namespace}"
    log_info "Rollback complete"
}

cleanup() {
    log_info "Cleaning up..."
    # Remove temporary files
    rm -f bandit-report.json
    log_info "Cleanup complete"
}

# Main pipeline
main() {
    log_info "Starting Victor AI build pipeline..."
    log_info "Version: ${VERSION}"
    log_info "Environment: ${BUILD_ENV}"

    check_dependencies
    setup_environment
    lint_code

    if [ "${SKIP_TESTS}" != "true" ]; then
        run_tests
    fi

    security_scan
    build_image

    if [ "${SKIP_SCAN}" != "true" ]; then
        scan_image
    fi

    if [ "${DEPLOY}" = "true" ]; then
        deploy_kubernetes
        run_health_checks
    fi

    cleanup

    log_info "Build pipeline completed successfully!"
}

# Trap errors
trap 'log_error "Pipeline failed!"; cleanup; exit 1' ERR

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --environment)
            BUILD_ENV="$2"
            shift 2
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-scan)
            SKIP_SCAN=true
            shift
            ;;
        --rollback)
            rollback
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --version VERSION        Set version (default: 0.5.1)"
            echo "  --environment ENV        Set environment (default: staging)"
            echo "  --deploy                 Deploy to Kubernetes after build"
            echo "  --skip-tests             Skip running tests"
            echo "  --skip-scan              Skip security scanning"
            echo "  --rollback               Rollback deployment"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main pipeline
main
