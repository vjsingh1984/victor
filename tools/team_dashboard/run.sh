#!/bin/bash
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Startup script for Victor Team Dashboard

This script starts both the FastAPI backend server and the React frontend
development server for the team collaboration dashboard.

Usage:
    ./run.sh [--prod] [--port 8000]

Options:
    --prod       Run in production mode (build frontend, serve with uvicorn)
    --port PORT  Specify backend port (default: 8000)
    --help       Show this help message

Examples:
    ./run.sh                    # Start both servers in dev mode
    ./run.sh --prod             # Start in production mode
    ./run.sh --port 9000        # Use custom port
"""

set -e

# Default values
PROD_MODE=false
BACKEND_PORT=8000
FRONTEND_PORT=3000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            PROD_MODE=true
            shift
            ;;
        --port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --help)
            sed -n '/^"""/,/^"""/p' "$0" | sed '1d;$d' | sed 's/^"""//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Victor Team Dashboard${NC}"
echo "=========================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"

    # Kill background processes
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi

    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    echo -e "${GREEN}Done.${NC}"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# =============================================================================
# Start Backend Server
# =============================================================================

echo -e "${BLUE}[1/2]${NC} Starting FastAPI backend server..."

# Check if fastapi is installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Warning: FastAPI not found. Installing...${NC}"
    echo "Run: pip install victor-ai[api]"
    # Continue anyway - might be installed in different environment
fi

# Start backend server
cd "$SCRIPT_DIR/../../.."

if [ "$PROD_MODE" = true ]; then
    # Production mode: use uvicorn directly
    echo "Starting backend in production mode on port $BACKEND_PORT..."
    python3 -m uvicorn victor.workflows.team_dashboard_api:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        --log-level info &
    BACKEND_PID=$!
else
    # Development mode
    echo "Starting backend in development mode on port $BACKEND_PORT..."
    python3 -m uvicorn victor.workflows.team_dashboard_api:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        --reload \
        --log-level debug &
    BACKEND_PID=$!
fi

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend server failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend server started on port $BACKEND_PORT${NC}"

# =============================================================================
# Start Frontend Server
# =============================================================================

echo ""
echo -e "${BLUE}[2/2]${NC} Starting React frontend server..."

cd "$SCRIPT_DIR"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

if [ "$PROD_MODE" = true ]; then
    # Production mode: build and serve
    echo "Building frontend..."
    npm run build

    echo "Frontend built successfully"
    echo "Serve the 'dist' directory with your web server"
    echo "Backend API is available at: http://localhost:$BACKEND_PORT"

    # Keep only backend running
    FRONTEND_PID=
else
    # Development mode: start Vite dev server
    echo "Starting frontend in development mode on port $FRONTEND_PORT..."
    npm run dev &
    FRONTEND_PID=$!

    echo -e "${GREEN}✓ Frontend server started on port $FRONTEND_PORT${NC}"
fi

# =============================================================================
# Display Access Information
# =============================================================================

echo ""
echo "=========================================="
echo -e "${GREEN}Dashboard is running!${NC}"
echo "=========================================="
echo ""

if [ "$PROD_MODE" = false ]; then
    echo "Frontend:  http://localhost:$FRONTEND_PORT"
    echo "Backend:   http://localhost:$BACKEND_PORT"
    echo "API Docs:  http://localhost:$BACKEND_PORT/docs"
    echo ""
    echo "WebSocket: ws://localhost:$BACKEND_PORT/ws/team/<execution_id>"
else
    echo "Backend API:  http://localhost:$BACKEND_PORT"
    echo "API Docs:     http://localhost:$BACKEND_PORT/docs"
    echo ""
    echo "Frontend:     Serve the 'tools/team_dashboard/dist' directory"
fi

echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for any process to exit
if [ -n "$FRONTEND_PID" ]; then
    wait $FRONTEND_PID $BACKEND_PID
else
    wait $BACKEND_PID
fi
