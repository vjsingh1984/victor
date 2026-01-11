#!/bin/bash
# Script to serve documentation locally with proper dependencies

set -e

echo "📚 Victor Documentation Local Server"
echo "===================================="

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

# Install dependencies if needed
echo "🔧 Checking documentation dependencies..."
python3 -c "import mkdocs" 2>/dev/null || {
    echo "📦 Installing MkDocs and dependencies..."
    pip install -e ".[docs]"
}

# Check if mkdocs-material is installed
python3 -c "import mkdocs_material" 2>/dev/null || {
    echo "📦 Installing mkdocs-material..."
    pip install mkdocs-material
}

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🚀 Starting documentation server..."
echo "   Documentation will be available at: http://127.0.0.1:8000"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the development server
mkdocs serve "$@"
