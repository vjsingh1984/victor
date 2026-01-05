#!/bin/bash
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

# Victor MCP Server Publisher
# This script helps users publish/run Victor as an MCP server.
#
# Usage:
#   ./scripts/publish_mcp_server.sh [command] [options]
#
# Commands:
#   run       - Run Victor MCP server directly (default)
#   docker    - Build and run Docker container
#   build     - Build Docker image only
#   install   - Install Victor for MCP server usage
#   config    - Generate Claude Desktop configuration
#   test      - Test MCP server functionality
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

# Default values
DOCKER_IMAGE="victor-ai/mcp-server"
DOCKER_TAG="latest"
LOG_LEVEL="${VICTOR_MCP_LOG_LEVEL:-WARNING}"

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
    print_msg "$BLUE" "  Victor MCP Server"
    print_msg "$BLUE" "======================================"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  run       Run Victor MCP server directly (default)"
    echo "  docker    Build and run Docker container"
    echo "  build     Build Docker image only"
    echo "  install   Install Victor for MCP server usage"
    echo "  config    Generate Claude Desktop configuration"
    echo "  test      Test MCP server functionality"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  --tools TOOLS       Comma-separated list of tools to expose"
    echo "  --vertical VERT     Vertical to expose (coding,devops,rag,dataanalysis,research)"
    echo "  --log-level LEVEL   Log level (DEBUG,INFO,WARNING,ERROR)"
    echo "  --tag TAG           Docker image tag (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 run                           # Run MCP server with all tools"
    echo "  $0 run --vertical coding         # Run with coding tools only"
    echo "  $0 docker                        # Build and run in Docker"
    echo "  $0 config > claude_config.json   # Generate Claude Desktop config"
    echo "  $0 test                          # Test server functionality"
    echo ""
}

# Install Victor
install_victor() {
    print_msg "$BLUE" "Installing Victor for MCP server usage..."

    if ! command_exists pip; then
        print_msg "$RED" "Error: pip is not installed"
        exit 1
    fi

    # Check if in project directory
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        print_msg "$YELLOW" "Installing from local source..."
        pip install -e "$PROJECT_ROOT[dev]"
    else
        print_msg "$YELLOW" "Installing from PyPI..."
        pip install victor-ai
    fi

    # Verify installation
    if command_exists victor; then
        print_msg "$GREEN" "Victor installed successfully!"
        victor --version
    else
        print_msg "$RED" "Error: Victor installation failed"
        exit 1
    fi
}

# Run MCP server directly
run_server() {
    local tools=""
    local vertical=""
    local log_level="$LOG_LEVEL"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tools)
                tools="$2"
                shift 2
                ;;
            --vertical)
                vertical="$2"
                shift 2
                ;;
            --log-level)
                log_level="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    print_msg "$BLUE" "Starting Victor MCP Server..."

    if ! command_exists victor; then
        print_msg "$RED" "Error: Victor is not installed"
        print_msg "$YELLOW" "Run: $0 install"
        exit 1
    fi

    # Set environment variables
    if [ -n "$tools" ]; then
        export VICTOR_MCP_TOOLS="$tools"
        print_msg "$YELLOW" "Exposing tools: $tools"
    fi

    if [ -n "$vertical" ]; then
        export VICTOR_MCP_VERTICALS="$vertical"
        print_msg "$YELLOW" "Exposing vertical: $vertical"
    fi

    export VICTOR_MCP_LOG_LEVEL="$log_level"

    # Run server
    victor mcp --log-level "$log_level"
}

# Build Docker image
build_docker() {
    local tag="${1:-$DOCKER_TAG}"

    print_msg "$BLUE" "Building Victor MCP Server Docker image..."

    if ! command_exists docker; then
        print_msg "$RED" "Error: Docker is not installed"
        exit 1
    fi

    cd "$PROJECT_ROOT"

    docker build \
        -t "$DOCKER_IMAGE:$tag" \
        -f docker/mcp-server/Dockerfile \
        .

    print_msg "$GREEN" "Docker image built: $DOCKER_IMAGE:$tag"
}

# Run Docker container
run_docker() {
    local tools=""
    local vertical=""
    local tag="$DOCKER_TAG"
    local workspace="${PWD}"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tools)
                tools="$2"
                shift 2
                ;;
            --vertical)
                vertical="$2"
                shift 2
                ;;
            --tag)
                tag="$2"
                shift 2
                ;;
            --workspace)
                workspace="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    print_msg "$BLUE" "Running Victor MCP Server in Docker..."

    if ! command_exists docker; then
        print_msg "$RED" "Error: Docker is not installed"
        exit 1
    fi

    # Check if image exists, build if not
    if ! docker image inspect "$DOCKER_IMAGE:$tag" >/dev/null 2>&1; then
        print_msg "$YELLOW" "Image not found, building..."
        build_docker "$tag"
    fi

    # Build docker run command
    local docker_args=(
        "run" "-it" "--rm"
        "-v" "$workspace:/workspace"
    )

    if [ -n "$tools" ]; then
        docker_args+=("-e" "VICTOR_MCP_TOOLS=$tools")
    fi

    if [ -n "$vertical" ]; then
        docker_args+=("-e" "VICTOR_MCP_VERTICALS=$vertical")
    fi

    docker_args+=("$DOCKER_IMAGE:$tag")

    print_msg "$GREEN" "Starting container..."
    docker "${docker_args[@]}"
}

# Generate Claude Desktop configuration
generate_config() {
    local victor_path

    if command_exists victor; then
        victor_path=$(which victor)
    else
        victor_path="victor"
    fi

    cat << EOF
{
  "mcpServers": {
    "victor": {
      "command": "$victor_path",
      "args": ["mcp", "--stdio"],
      "env": {}
    },
    "victor-coding": {
      "command": "$victor_path",
      "args": ["mcp", "--stdio"],
      "env": {
        "VICTOR_MCP_VERTICALS": "coding"
      }
    },
    "victor-safe": {
      "command": "$victor_path",
      "args": ["mcp", "--stdio"],
      "env": {
        "VICTOR_MCP_TOOLS": "read,ls,grep,find_files,code_search,git"
      }
    }
  }
}
EOF
}

# Test MCP server
test_server() {
    print_msg "$BLUE" "Testing Victor MCP Server..."

    if ! command_exists victor; then
        print_msg "$RED" "Error: Victor is not installed"
        exit 1
    fi

    # Create test script
    local test_script=$(mktemp)
    cat << 'EOF' > "$test_script"
import asyncio
import json
import subprocess
import sys

async def test_mcp_server():
    print("Starting MCP server test...")

    # Start server process
    process = subprocess.Popen(
        ["victor", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Test 1: Initialize
        print("\n1. Testing initialize...")
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": "1",
            "method": "initialize",
            "params": {"clientInfo": {"name": "Test Client", "version": "1.0.0"}}
        }) + "\n"

        process.stdin.write(init_msg)
        process.stdin.flush()

        response = json.loads(process.stdout.readline())
        if "result" in response and "serverInfo" in response["result"]:
            print(f"   OK: {response['result']['serverInfo']['name']}")
        else:
            print(f"   FAIL: {response}")
            return False

        # Test 2: List tools
        print("\n2. Testing list tools...")
        list_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": "2",
            "method": "tools/list"
        }) + "\n"

        process.stdin.write(list_msg)
        process.stdin.flush()

        response = json.loads(process.stdout.readline())
        if "result" in response and "tools" in response["result"]:
            tools_count = len(response["result"]["tools"])
            print(f"   OK: {tools_count} tools available")
        else:
            print(f"   FAIL: {response}")
            return False

        # Test 3: Ping
        print("\n3. Testing ping...")
        ping_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": "3",
            "method": "ping"
        }) + "\n"

        process.stdin.write(ping_msg)
        process.stdin.flush()

        response = json.loads(process.stdout.readline())
        if "result" in response and response["result"].get("pong"):
            print("   OK: pong received")
        else:
            print(f"   FAIL: {response}")
            return False

        print("\n" + "=" * 40)
        print("All tests passed!")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
EOF

    python "$test_script"
    local result=$?
    rm -f "$test_script"

    if [ $result -eq 0 ]; then
        print_msg "$GREEN" "\nMCP Server test completed successfully!"
    else
        print_msg "$RED" "\nMCP Server test failed!"
        exit 1
    fi
}

# Main
main() {
    local command="${1:-run}"
    shift || true

    case $command in
        run)
            run_server "$@"
            ;;
        docker)
            run_docker "$@"
            ;;
        build)
            build_docker "$@"
            ;;
        install)
            install_victor
            ;;
        config)
            generate_config
            ;;
        test)
            test_server
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_msg "$RED" "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
