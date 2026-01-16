# Installation Guide - Victor Workflow Editor

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- pip (Python package manager)

## Quick Install

```bash
cd tools/workflow_editor
./install.sh
```

This will:
1. Install backend dependencies (FastAPI, uvicorn, etc.)
2. Install frontend dependencies (React, React Flow, etc.)
3. Prepare the development environment

## Manual Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
```

Backend requirements:
- fastapi>=0.111.0
- uvicorn[standard]>=0.30.1
- python-multipart>=0.0.9
- pydantic>=2.7.1

### Frontend

```bash
cd frontend
npm install
```

Frontend dependencies:
- react@^18.3.1
- react-dom@^18.3.1
- reactflow@^11.11.4
- zustand@^4.5.2
- axios@^1.7.2
- yaml@^2.5.0
- @monaco-editor/react@^4.6.0

## Development Setup

### Start Both Services

```bash
./run.sh
```

This starts:
- Backend API on http://localhost:8000
- Frontend UI on http://localhost:3000

### Start Services Separately

**Backend:**
```bash
cd backend
python api.py
# Server runs on http://localhost:8000
```

**Frontend:**
```bash
cd frontend
npm run dev
# Server runs on http://localhost:3000
```

## Troubleshooting

### Port Already in Use

If port 8000 or 3000 is already in use:

**Change backend port:**
```bash
cd backend
# Edit api.py and change the port in uvicorn.run()
python api.py
```

**Change frontend port:**
```bash
cd frontend
npm run dev -- --port 3001
```

### Python Dependencies Issues

If you encounter Python dependency issues:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Node Modules Issues

If frontend has issues:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Production Build

```bash
./build.sh
```

This creates an optimized build in `frontend/dist/`.

To preview the production build:

```bash
cd frontend
npm run preview
```

## Next Steps

After installation:
1. Run `./run.sh` to start the editor
2. Open http://localhost:3000 in your browser
3. See [user_manual.md](user_manual.md) for usage instructions
