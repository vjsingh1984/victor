#!/bin/bash
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Victor Docker Image Publisher
# Pushes Victor Docker images to Docker Hub.
#
# Usage:
#   ./scripts/push_docker.sh [command] [options]
#
# Commands:
#   latest    - Push the latest tag (default)
#   version   - Push the version tag
#   all       - Push all tags
#   mcp       - Push MCP server image
#   core      - Push core image
#   help      - Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Docker image configuration
DOCKER_REGISTRY="vjsingh1984"
IMAGE_NAME="victor-ai"
VERSION=$(grep -m1 'version = ' "$PROJECT_ROOT/pyproject.toml" | cut -d'"' -f2 || echo "latest")

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Print header
print_header() {
    echo ""
    print_msg "$BLUE" "======================================"
    print_msg "$BLUE" "  Victor Docker Hub Publisher"
    print_msg "$BLUE" "======================================"
    echo ""
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  latest    Push the latest tag (default)"
    echo "  version   Push the version tag ($VERSION)"
    echo "  all       Push all tags"
    echo "  mcp       Push MCP server image"
    echo "  core      Push core image"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  --tag TAG       Custom tag to push"
    echo "  --dry-run       Show what would be pushed without actually pushing"
    echo ""
    echo "Image naming convention:"
    echo "  - Full image:  ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    echo "  - Version:     ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
    echo "  - Core image:  ${DOCKER_REGISTRY}/${IMAGE_NAME}:core"
    echo "  - MCP image:   ${DOCKER_REGISTRY}/${IMAGE_NAME}:mcp-server"
    echo ""
    echo "Examples:"
    echo "  $0                         # Push latest tag"
    echo "  $0 version                 # Push version tag"
    echo "  $0 all                     # Push all tags"
    echo "  $0 --tag v1.0.0            # Push custom tag"
    echo "  $0 all --dry-run           # Show what would be pushed"
    echo ""
}

# Check Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_msg "$RED" "Error: Docker is not installed"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_msg "$RED" "Error: Docker daemon is not running"
        exit 1
    fi
}

# Check if logged in to Docker Hub
check_login() {
    if ! docker info 2>/dev/null | grep -q "Username"; then
        print_msg "$YELLOW" "Not logged in to Docker Hub. Please log in:"
        docker login
    fi
    print_msg "$GREEN" "Logged in to Docker Hub"
}

# Check if image exists locally
check_image_exists() {
    local image="$1"
    if ! docker image inspect "$image" &> /dev/null; then
        print_msg "$RED" "Error: Image '$image' not found locally"
        print_msg "$YELLOW" "Build it first with: ./scripts/build_docker.sh"
        return 1
    fi
    return 0
}

# Push a single image
push_image() {
    local image_tag="$1"
    local dry_run="${2:-false}"

    if [ "$dry_run" = "true" ]; then
        print_msg "$YELLOW" "[DRY-RUN] Would push: ${image_tag}"
        return 0
    fi

    print_msg "$BLUE" "Pushing ${image_tag}..."

    if ! check_image_exists "$image_tag"; then
        return 1
    fi

    docker push "${image_tag}"
    print_msg "$GREEN" "Successfully pushed: ${image_tag}"
}

# Main
main() {
    local command="${1:-latest}"
    shift || true

    local custom_tag=""
    local dry_run="false"

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)
                custom_tag="$2"
                shift 2
                ;;
            --dry-run)
                dry_run="true"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    print_header
    check_docker

    if [ "$dry_run" != "true" ]; then
        check_login
    fi

    echo ""

    case $command in
        latest)
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest" "$dry_run"
            ;;
        version)
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}" "$dry_run"
            ;;
        mcp)
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:mcp-server" "$dry_run"
            ;;
        core)
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:core" "$dry_run"
            ;;
        all)
            print_msg "$BLUE" "Pushing all Victor images..."
            echo ""
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:latest" "$dry_run"
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}" "$dry_run"
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:core" "$dry_run"
            push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:mcp-server" "$dry_run"
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            if [ -n "$custom_tag" ]; then
                push_image "${DOCKER_REGISTRY}/${IMAGE_NAME}:${custom_tag}" "$dry_run"
            else
                print_msg "$RED" "Unknown command: $command"
                show_help
                exit 1
            fi
            ;;
    esac

    echo ""
    if [ "$dry_run" = "true" ]; then
        print_msg "$YELLOW" "Dry-run completed. No images were pushed."
    else
        print_msg "$GREEN" "Push completed successfully!"
    fi
    echo ""
    print_msg "$YELLOW" "Docker Hub URL:"
    echo "  https://hub.docker.com/r/${DOCKER_REGISTRY}/${IMAGE_NAME}"
    echo ""
}

main "$@"
