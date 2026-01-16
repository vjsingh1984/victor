#!/bin/bash
# Install dependencies for Victor Workflow Editor

set -e

echo "🔧 Installing Victor Workflow Editor dependencies..."

# Check if we're in the right directory
if [ ! -f "install.sh" ]; then
    echo "❌ Error: Please run this script from the workflow_editor directory"
    exit 1
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << EOF
fastapi==0.111.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pydantic==2.7.1
EOF
fi
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

echo "✅ Installation complete!"
echo ""
echo "To start the editor:"
echo "  ./run.sh"
echo ""
echo "Or start separately:"
echo "  Backend:  cd backend && python api.py"
echo "  Frontend: cd frontend && npm run dev"
