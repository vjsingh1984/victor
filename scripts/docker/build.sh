#!/bin/bash
#############################################################################
# Docker Build Script for Victor AI
#
# Features:
# - Multi-stage builds
# - Tagging strategy
# - Registry push
# - Image scanning
#
# Usage: ./build.sh [version] [--push] [--scan]
#############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REGISTRY="${DOCKER_REGISTRY:-docker.io/victorai}"
IMAGE_NAME="${IMAGE_NAME:-victor}"
BASE_TAG="${REGISTRY}/${IMAGE_NAME}"

# Git information
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
VERSION="${1:-latest}"

# Flags
PUSH_IMAGE=false
SCAN_IMAGE=false
BUILD_ARGS=()

# Logging functions
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

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Trap errors
trap 'error_exit "Build failed at line $LINENO"' ERR

#############################################################################
# Parse Arguments
#############################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --push)
                PUSH_IMAGE=true
                shift
                ;;
            --scan)
                SCAN_IMAGE=true
                shift
                ;;
            --registry)
                REGISTRY="$2"
                BASE_TAG="${REGISTRY}/${IMAGE_NAME}"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [version] [options]"
                echo ""
                echo "Arguments:"
                echo "  version       Image version tag (default: latest)"
                echo ""
                echo "Options:"
                echo "  --push        Push image to registry"
                echo "  --scan        Scan image for vulnerabilities"
                echo "  --registry    Docker registry (default: docker.io/victorai)"
                echo "  --help        Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 1.0.0 --push"
                echo "  $0 latest --push --scan"
                echo "  $0 \$(git describe --tags) --registry ghcr.io/victorai"
                exit 0
                ;;
            *)
                VERSION="$1"
                shift
                ;;
        esac
    done
}

#############################################################################
# Validate Docker Environment
#############################################################################
validate_docker() {
    log_info "Validating Docker environment..."

    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error_exit "Docker daemon is not running"
    fi

    log_success "Docker environment validated"
}

#############################################################################
# Build Docker Image
#############################################################################
build_image() {
    log_info "Building Docker image..."
    log_info "Version: ${VERSION}"
    log_info "Git SHA: ${GIT_SHA}"
    log_info "Git Branch: ${GIT_BRANCH}"

    cd "${PROJECT_ROOT}"

    # Build arguments
    BUILD_ARGS=(
        "--build-arg" "VERSION=${VERSION}"
        "--build-arg" "GIT_SHA=${GIT_SHA}"
        "--build-arg" "GIT_BRANCH=${GIT_BRANCH}"
        "--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    )

    # Additional tags
    TAGS=(
        "-t" "${BASE_TAG}:${VERSION}"
        "-t" "${BASE_TAG}:${GIT_SHA}"
        "-t" "${BASE_TAG}:${GIT_BRANCH}"
    )

    # Tag 'latest' if this is a production build
    if [[ "$VERSION" != *"-"* ]] && [ "$VERSION" != "latest" ]; then
        TAGS+=("-t" "${BASE_TAG}:latest")
    fi

    log_info "Building with tags: ${TAGS[*]}"

    # Build image
    docker build \
        "${BUILD_ARGS[@]}" \
        "${TAGS[@]}" \
        -f "${PROJECT_ROOT}/Dockerfile" \
        "${PROJECT_ROOT}"

    log_success "Docker image built successfully"
}

#############################################################################
# Push Docker Image
#############################################################################
push_image() {
    if [ "$PUSH_IMAGE" = false ]; then
        log_info "Skipping image push (use --push to enable)"
        return
    fi

    log_info "Pushing Docker image to registry..."

    # Push all tags
    docker push "${BASE_TAG}:${VERSION}"
    docker push "${BASE_TAG}:${GIT_SHA}"
    docker push "${BASE_TAG}:${GIT_BRANCH}"

    if [[ "$VERSION" != *"-"* ]] && [ "$VERSION" != "latest" ]; then
        docker push "${BASE_TAG}:latest"
    fi

    log_success "Docker image pushed successfully"
}

#############################################################################
# Scan Docker Image
#############################################################################
scan_image() {
    if [ "$SCAN_IMAGE" = false ]; then
        log_info "Skipping image scan (use --scan to enable)"
        return
    fi

    log_info "Scanning Docker image for vulnerabilities..."

    # Check if Trivy is available
    if command -v trivy &> /dev/null; then
        log_info "Using Trivy for vulnerability scanning..."

        trivy image \
            --severity HIGH,CRITICAL \
            --exit-code 1 \
            "${BASE_TAG}:${VERSION}" || error_exit "Vulnerabilities found!"

        log_success "No critical vulnerabilities found"
    elif command -v docker &> /dev/null; then
        log_info "Using docker scout for vulnerability scanning..."

        docker scout quickcheck \
            --severity high,critical \
            "${BASE_TAG}:${VERSION}" || log_warning "Vulnerabilities found (non-blocking)"

        log_success "Vulnerability scan completed"
    else
        log_warning "No vulnerability scanner available (trivy or docker scout)"
        log_info "Install trivy: https://aquasecurity.github.io/trivy/"
    fi
}

#############################################################################
# Display Image Information
#############################################################################
display_image_info() {
    log_info "Image Information:"
    echo ""
    echo "=========================================="
    docker images "${BASE_TAG}:${VERSION}"
    echo "=========================================="
    echo ""
    echo "Tags created:"
    echo "  ${BASE_TAG}:${VERSION}"
    echo "  ${BASE_TAG}:${GIT_SHA}"
    echo "  ${BASE_TAG}:${GIT_BRANCH}"

    if [[ "$VERSION" != *"-"* ]] && [ "$VERSION" != "latest" ]; then
        echo "  ${BASE_TAG}:latest"
    fi

    echo ""
    echo "Image size:"
    docker images "${BASE_TAG}:${VERSION}" --format "  {{.Size}}"

    echo ""
    echo "Run container:"
    echo "  docker run -d -p 8000:8000 --name victor ${BASE_TAG}:${VERSION}"
    echo ""
}

#############################################################################
# Main Build Flow
#############################################################################
main() {
    log_info "Starting Docker build for Victor AI..."
    log_info "Build started at $(date)"

    # Parse arguments
    parse_args "$@"

    # Validate Docker
    validate_docker

    # Build image
    build_image

    # Scan image
    scan_image

    # Push image
    push_image

    # Display info
    display_image_info

    log_success "Docker build completed successfully!"
    log_info "Build finished at $(date)"
}

# Run main build
main "$@"
