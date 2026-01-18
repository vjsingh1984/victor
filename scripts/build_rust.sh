#!/bin/bash
# Build script for victor_native Rust extensions
#
# Usage: ./scripts/build_rust.sh [options]
#
# Options:
#   --release    Build with release optimizations (default)
#   --debug      Build with debug symbols
#   --test       Run tests after building
#
# Examples:
#   ./scripts/build_rust.sh
#   ./scripts/build_rust.sh --release
#   ./scripts/build_rust.sh --debug --test

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$PROJECT_ROOT/rust"

# Default options
PROFILE="release"
RUN_TESTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            PROFILE="release"
            shift
            ;;
        --debug)
            PROFILE="debug"
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🚀 Victor Native Rust Extensions Build Script"
echo "============================================================"

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}❌ Rust toolchain not found${NC}"
    echo "Please install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

RUST_VERSION=$(rustc --version)
echo -e "${GREEN}✅ Found Rust: ${RUST_VERSION}${NC}"

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo -e "${YELLOW}⚠️  Maturin not found. Installing...${NC}"
    python3 -m pip install "maturin>=1.4"
fi

MATURIN_VERSION=$(maturin --version)
echo -e "${GREEN}✅ Found maturin: ${MATURIN_VERSION}${NC}"

# Build the Rust extensions
echo ""
echo "🔨 Building Rust extensions with ${PROFILE} profile..."
echo "   Working directory: ${RUST_DIR}"

cd "${RUST_DIR}"

# Build command
if [ "$PROFILE" == "release" ]; then
    maturin develop --release
elif [ "$PROFILE" == "debug" ]; then
    maturin develop --release-with-debug
else
    echo "Unknown profile: ${PROFILE}"
    exit 1
fi

echo -e "${GREEN}✅ Build completed${NC}"

# Verify the build
echo ""
echo "🔍 Verifying build..."

cd "${PROJECT_ROOT}"

if python3 -c "import victor_native; print(f'victor_native version: {victor_native.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✅ Build verification successful!${NC}"
else
    echo -e "${RED}❌ Build verification failed${NC}"
    exit 1
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "🧪 Running Rust unit tests..."
    cd "${RUST_DIR}"
    if cargo test --lib; then
        echo -e "${GREEN}✅ All tests passed${NC}"
    else
        echo -e "${RED}❌ Tests failed${NC}"
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo "   Profile: ${PROFILE}"
echo "   Module: victor_native"
echo "============================================================"
