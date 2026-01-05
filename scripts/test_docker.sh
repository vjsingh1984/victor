#!/bin/bash
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Victor Docker Local Testing
# Tests Victor Docker images locally before pushing to Docker Hub.
#
# Usage:
#   ./scripts/test_docker.sh [command] [options]
#
# Commands:
#   full      - Test the full Victor image (default)
#   core      - Test the minimal core image
#   mcp       - Test the MCP server image
#   all       - Test all images
#   cleanup   - Remove test containers
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
CONTAINER_PREFIX="victor-ai-test"

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
    print_msg "$BLUE" "  Victor Docker Image Tester"
    print_msg "$BLUE" "======================================"
    echo ""
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  full      Test the full Victor image (default)"
    echo "  core      Test the minimal core image"
    echo "  mcp       Test the MCP server image"
    echo "  all       Test all images"
    echo "  cleanup   Remove test containers"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  --interactive   Run container interactively"
    echo "  --keep          Keep container after test"
    echo ""
    echo "Container naming convention:"
    echo "  - Full test:  ${CONTAINER_PREFIX}-full"
    echo "  - Core test:  ${CONTAINER_PREFIX}-core"
    echo "  - MCP test:   ${CONTAINER_PREFIX}-mcp"
    echo ""
    echo "Examples:"
    echo "  $0                         # Test full image"
    echo "  $0 mcp                     # Test MCP server image"
    echo "  $0 all                     # Test all images"
    echo "  $0 full --interactive      # Run full image interactively"
    echo "  $0 cleanup                 # Remove all test containers"
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

# Run a test and report result
run_test() {
    local test_name="$1"
    local test_cmd="$2"

    echo -n "  Test: $test_name... "
    if eval "$test_cmd" &> /dev/null; then
        print_msg "$GREEN" "PASS"
        return 0
    else
        print_msg "$RED" "FAIL"
        return 1
    fi
}

# Test full image
test_full() {
    local interactive="${1:-false}"
    local keep="${2:-false}"
    local image="${DOCKER_REGISTRY}/${IMAGE_NAME}:latest"
    local container="${CONTAINER_PREFIX}-full"

    print_msg "$BLUE" "Testing full Victor image..."
    print_msg "$YELLOW" "  Image: ${image}"
    print_msg "$YELLOW" "  Container: ${container}"
    echo ""

    if ! check_image_exists "$image"; then
        return 1
    fi

    # Stop existing container if any
    docker rm -f "$container" 2>/dev/null || true

    if [ "$interactive" = "true" ]; then
        print_msg "$YELLOW" "Starting interactive session..."
        docker run -it --rm --name "$container" \
            -v "$PROJECT_ROOT:/workspace:ro" \
            "$image" bash
        return 0
    fi

    # Run container
    print_msg "$YELLOW" "Starting container..."
    docker run -d --name "$container" \
        -v "$PROJECT_ROOT:/workspace:ro" \
        "$image" sleep 300

    local passed=0
    local failed=0

    # Test 1: Victor version
    if run_test "Victor version" "docker exec $container victor --version"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2: Python import
    if run_test "Python import" "docker exec $container python -c 'import victor; print(victor.__version__)'"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 3: Check tools are registered
    if run_test "Tools registered" "docker exec $container python -c 'from victor.tools.registry import ToolRegistry; r = ToolRegistry(); print(f\"Tools: {len(r._tools)}\")'"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 4: Embedding model is cached
    if run_test "Embedding model cached" "docker exec $container python -c 'from sentence_transformers import SentenceTransformer; m = SentenceTransformer(\"BAAI/bge-small-en-v1.5\"); print(\"OK\")'"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 5: Health check
    if run_test "Health check" "docker exec $container victor --version"; then
        ((passed++))
    else
        ((failed++))
    fi

    echo ""

    # Cleanup unless --keep
    if [ "$keep" != "true" ]; then
        docker rm -f "$container" > /dev/null 2>&1
        print_msg "$YELLOW" "Container removed"
    else
        print_msg "$YELLOW" "Container kept: $container"
    fi

    echo ""
    print_msg "$BLUE" "Results: $passed passed, $failed failed"

    if [ $failed -eq 0 ]; then
        print_msg "$GREEN" "All tests passed for full image!"
        return 0
    else
        print_msg "$RED" "Some tests failed for full image"
        return 1
    fi
}

# Test core image
test_core() {
    local interactive="${1:-false}"
    local keep="${2:-false}"
    local image="${DOCKER_REGISTRY}/${IMAGE_NAME}:core"
    local container="${CONTAINER_PREFIX}-core"

    print_msg "$BLUE" "Testing core Victor image..."
    print_msg "$YELLOW" "  Image: ${image}"
    print_msg "$YELLOW" "  Container: ${container}"
    echo ""

    if ! check_image_exists "$image"; then
        return 1
    fi

    # Stop existing container if any
    docker rm -f "$container" 2>/dev/null || true

    if [ "$interactive" = "true" ]; then
        print_msg "$YELLOW" "Starting interactive session..."
        docker run -it --rm --name "$container" "$image" bash
        return 0
    fi

    # Run container
    print_msg "$YELLOW" "Starting container..."
    docker run -d --name "$container" "$image" sleep 300

    local passed=0
    local failed=0

    # Test 1: Python import
    if run_test "Python import" "docker exec $container python -c 'import victor'"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2: Non-root user
    if run_test "Non-root user" "docker exec $container whoami | grep -q victor"; then
        ((passed++))
    else
        ((failed++))
    fi

    echo ""

    # Cleanup unless --keep
    if [ "$keep" != "true" ]; then
        docker rm -f "$container" > /dev/null 2>&1
        print_msg "$YELLOW" "Container removed"
    else
        print_msg "$YELLOW" "Container kept: $container"
    fi

    echo ""
    print_msg "$BLUE" "Results: $passed passed, $failed failed"

    if [ $failed -eq 0 ]; then
        print_msg "$GREEN" "All tests passed for core image!"
        return 0
    else
        print_msg "$RED" "Some tests failed for core image"
        return 1
    fi
}

# Test MCP server image
test_mcp() {
    local interactive="${1:-false}"
    local keep="${2:-false}"
    local image="${DOCKER_REGISTRY}/${IMAGE_NAME}:mcp-server"
    local container="${CONTAINER_PREFIX}-mcp"

    print_msg "$BLUE" "Testing MCP server image..."
    print_msg "$YELLOW" "  Image: ${image}"
    print_msg "$YELLOW" "  Container: ${container}"
    echo ""

    if ! check_image_exists "$image"; then
        return 1
    fi

    # Stop existing container if any
    docker rm -f "$container" 2>/dev/null || true

    if [ "$interactive" = "true" ]; then
        print_msg "$YELLOW" "Starting interactive session..."
        docker run -it --rm --name "$container" \
            --entrypoint bash \
            "$image"
        return 0
    fi

    # Run container in background (override entrypoint for testing)
    print_msg "$YELLOW" "Starting container..."
    docker run -d --name "$container" \
        --entrypoint sleep \
        "$image" 300

    local passed=0
    local failed=0

    # Test 1: Victor version
    if run_test "Victor version" "docker exec $container victor --version"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 2: MCP server module importable
    if run_test "MCP module import" "docker exec $container python -c 'from victor.integrations.mcp.server import MCPServer'"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 3: Non-root user
    if run_test "Non-root user" "docker exec $container whoami | grep -q victor"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 4: Workspace directory exists
    if run_test "Workspace exists" "docker exec $container test -d /workspace"; then
        ((passed++))
    else
        ((failed++))
    fi

    # Test 5: MCP environment variables
    if run_test "MCP env vars" "docker exec $container bash -c 'test -n \"\$VICTOR_HOME\"'"; then
        ((passed++))
    else
        ((failed++))
    fi

    echo ""

    # Cleanup unless --keep
    if [ "$keep" != "true" ]; then
        docker rm -f "$container" > /dev/null 2>&1
        print_msg "$YELLOW" "Container removed"
    else
        print_msg "$YELLOW" "Container kept: $container"
    fi

    echo ""
    print_msg "$BLUE" "Results: $passed passed, $failed failed"

    if [ $failed -eq 0 ]; then
        print_msg "$GREEN" "All tests passed for MCP server image!"
        return 0
    else
        print_msg "$RED" "Some tests failed for MCP server image"
        return 1
    fi
}

# Cleanup test containers
cleanup() {
    print_msg "$BLUE" "Cleaning up test containers..."

    local containers=$(docker ps -a --filter "name=${CONTAINER_PREFIX}" --format "{{.Names}}")

    if [ -z "$containers" ]; then
        print_msg "$YELLOW" "No test containers found"
        return 0
    fi

    echo "$containers" | while read -r container; do
        print_msg "$YELLOW" "Removing: $container"
        docker rm -f "$container" > /dev/null 2>&1
    done

    print_msg "$GREEN" "Cleanup completed!"
}

# Main
main() {
    local command="${1:-full}"
    shift || true

    local interactive="false"
    local keep="false"

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --interactive|-i)
                interactive="true"
                shift
                ;;
            --keep|-k)
                keep="true"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    print_header
    check_docker

    local exit_code=0

    case $command in
        full)
            test_full "$interactive" "$keep" || exit_code=1
            ;;
        core)
            test_core "$interactive" "$keep" || exit_code=1
            ;;
        mcp)
            test_mcp "$interactive" "$keep" || exit_code=1
            ;;
        all)
            print_msg "$BLUE" "Testing all Victor images..."
            echo ""
            test_full "$interactive" "$keep" || exit_code=1
            echo ""
            test_core "$interactive" "$keep" || exit_code=1
            echo ""
            test_mcp "$interactive" "$keep" || exit_code=1
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            print_msg "$RED" "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac

    echo ""
    if [ $exit_code -eq 0 ]; then
        print_msg "$GREEN" "All tests completed successfully!"
    else
        print_msg "$RED" "Some tests failed!"
    fi

    exit $exit_code
}

main "$@"
