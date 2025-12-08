#!/bin/bash
# Build extended context Ollama models (from tool-enabled bases)
# Models share weights with base - no storage duplication!
# Usage: ./build_extended_context_models.sh [localhost|remote|both]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCALHOST="http://localhost:11434"
REMOTE="http://192.168.1.20:11434"

build_model() {
    local host=$1
    local modelfile=$2
    local name=$3

    echo "Building $name on $host..."
    OLLAMA_HOST=$host ollama create "$name" -f "$modelfile"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully built $name (shares weights with base)"
    else
        echo "✗ Failed to build $name"
        return 1
    fi
}

verify_model() {
    local host=$1
    local name=$2

    echo "Verifying $name..."
    OLLAMA_HOST=$host ollama show "$name" 2>&1 | grep -E "context length|parameters|Capabilities" | head -5
}

# Parse argument
TARGET=${1:-both}

echo "============================================"
echo "Ollama Extended Context Model Builder"
echo "Building from -tools bases (shared weights)"
echo "============================================"
echo ""

case $TARGET in
    localhost)
        echo "Building for localhost (M1 Max 48GB)..."
        echo ""

        # Tool models with extended context
        build_model "$LOCALHOST" "$SCRIPT_DIR/deepseek-coder-tools-33b-128K.Modelfile" "deepseek-coder-tools:33b-128K"
        build_model "$LOCALHOST" "$SCRIPT_DIR/mixtral-tools-8x7b-65K.Modelfile" "mixtral-tools:8x7b-65K"
        build_model "$LOCALHOST" "$SCRIPT_DIR/gemma3-tools-27b-128K.Modelfile" "gemma3-tools:27b-128K"

        echo ""
        echo "=== Already optimal (no changes needed) ==="
        echo "qwen3-coder-tools:30b  -> 262K context (native)"
        echo "deepseek-r1-tools:32b  -> 131K context (native)"
        echo "devstral-tools:latest  -> 131K context (native)"

        echo ""
        echo "Verifying..."
        verify_model "$LOCALHOST" "deepseek-coder-tools:33b-128K"
        ;;

    remote)
        echo "Building for 192.168.1.20 (RTX 4000 Ada + 128GB)..."
        echo ""

        # Extended context models for larger RAM system
        build_model "$REMOTE" "$SCRIPT_DIR/deepseek-r1-tools-32b-262K.Modelfile" "deepseek-r1-tools:32b-262K"
        build_model "$REMOTE" "$SCRIPT_DIR/deepseek-r1-tools-70b-262K.Modelfile" "deepseek-r1-tools:70b-262K"

        echo ""
        echo "=== Already optimal (no changes needed) ==="
        echo "qwen3-coder-tools:30b  -> 262K context (native)"
        echo "deepseek-r1-tools:70b  -> 131K context (native)"

        echo ""
        echo "Verifying..."
        verify_model "$REMOTE" "deepseek-r1-tools:32b-262K"
        ;;

    both)
        echo "Building for both hosts..."
        echo ""

        echo "=== LOCALHOST (M1 Max 48GB) ==="
        build_model "$LOCALHOST" "$SCRIPT_DIR/deepseek-coder-tools-33b-128K.Modelfile" "deepseek-coder-tools:33b-128K"
        build_model "$LOCALHOST" "$SCRIPT_DIR/mixtral-tools-8x7b-65K.Modelfile" "mixtral-tools:8x7b-65K"
        build_model "$LOCALHOST" "$SCRIPT_DIR/gemma3-tools-27b-128K.Modelfile" "gemma3-tools:27b-128K"

        echo ""
        echo "=== REMOTE (RTX 4000 Ada + 128GB) ==="
        build_model "$REMOTE" "$SCRIPT_DIR/deepseek-r1-tools-32b-262K.Modelfile" "deepseek-r1-tools:32b-262K"
        build_model "$REMOTE" "$SCRIPT_DIR/deepseek-r1-tools-70b-262K.Modelfile" "deepseek-r1-tools:70b-262K"
        ;;

    *)
        echo "Usage: $0 [localhost|remote|both]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""
echo "Key tool+context models available:"
echo ""
echo "LOCALHOST (use for general coding):"
echo "  qwen3-coder-tools:30b      - 262K ctx, 30B params, tools"
echo "  deepseek-r1-tools:32b      - 131K ctx, 32B params, tools+thinking"
echo "  deepseek-coder-tools:33b-128K - 128K ctx, 33B params, tools"
echo "  devstral-tools:latest      - 131K ctx, 24B params, tools"
echo ""
echo "REMOTE (use for heavy tasks):"
echo "  deepseek-r1-tools:70b      - 131K ctx, 70B params, tools+thinking"
echo "  deepseek-r1-tools:32b-262K - 262K ctx, 32B params, tools+thinking"
echo "  qwen3-coder-tools:30b      - 262K ctx, 30B params, tools"
echo ""
echo "Note: Extended models share weights with base - no extra storage!"
