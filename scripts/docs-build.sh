#!/bin/bash
# Script to build documentation for deployment

set -e

echo "📚 Building Victor Documentation"
echo "================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if we're in the project root
if [ ! -f "mkdocs.yml" ]; then
    echo "❌ Error: mkdocs.yml not found. Please run this script from the project root."
    exit 1
fi

# Install dependencies
echo "🔧 Installing documentation dependencies..."
pip install -e ".[docs]"

echo ""
echo "🔨 Building documentation..."
mkdocs build --clean

echo ""
echo "✅ Documentation built successfully!"
echo "   Output: ./site/"
echo ""
echo "📖 To view locally, you can run:"
echo "   python3 -m http.server 8000 --directory site"
echo ""
