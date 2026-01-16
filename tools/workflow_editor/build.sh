#!/bin/bash
# Build frontend for production

set -e

echo "🔨 Building frontend for production..."

cd frontend
npm run build
cd ..

echo "✅ Build complete!"
echo "   Output: frontend/dist/"
echo ""
echo "To serve the built files:"
echo "   cd frontend && npm run preview"
