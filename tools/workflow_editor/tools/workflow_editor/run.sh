#!/bin/bash
# Victor Workflow Editor Startup Script

echo "🎨 Starting Victor Workflow Editor..."
echo ""

# Check if Python dependencies are available
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Installing backend dependencies..."
    pip install fastapi uvicorn pydantic -q
fi

# Start backend in background
echo "📡 Starting backend API on http://localhost:8000..."
cd "$(dirname "$0")"
python3 backend/api.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 2

# Open browser
echo ""
echo "🌐 Opening browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:3000" || open "frontend/index.html"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "http://localhost:3000" || xdg-open "frontend/index.html"
else
    start "http://localhost:3000" || start "frontend/index.html"
fi

echo ""
echo "✅ Workflow Editor is running!"
echo "   Frontend: file://$(pwd)/frontend/index.html"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the backend server"
echo ""

# Wait for backend process
wait $BACKEND_PID
