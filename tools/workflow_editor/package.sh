#!/bin/bash
# Package workflow editor as desktop app (optional)

set -e

echo "📦 Creating desktop app package..."

# Check if electron is installed
if ! command -v electron &> /dev/null; then
    echo "⚠️  Electron is not installed"
    echo "   To package as desktop app, install Electron:"
    echo "   npm install -g electron electron-builder"
    echo ""
    echo "   Then add electron-builder configuration to frontend/package.json"
    exit 1
fi

echo "⚠️  Desktop packaging is optional and requires additional setup"
echo "   See README.md for desktop app instructions"
