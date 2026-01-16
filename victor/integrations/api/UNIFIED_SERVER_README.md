# Victor Unified API Server

The **Victor Unified API Server** consolidates all Victor backend services into a single FastAPI application, simplifying deployment and reducing resource usage.

## Overview

The unified server combines three previously separate servers:

1. **Main API Server** (`/api/v1/*`)
   - Chat and completion endpoints
   - Code search (semantic and regex)
   - WebSocket streaming
   - Git integration
   - LSP services
   - Terminal commands
   - And 100+ more endpoints

2. **HITL API Server** (`/api/v1/hitl/*`)
   - Human-in-the-Loop approval management
   - Request/response handling
   - Approval history
   - WebSocket notifications

3. **Workflow Editor API** (`/workflow-editor/*`)
   - Workflow validation
   - YAML import/export
   - Node type definitions
   - Team configurations

## Quick Start

### Installation

```bash
# Install Victor with API server support
pip install victor-ai[api]

# Or install with all optional dependencies
pip install victor-ai[all]
```

### Running the Server

```bash
# Run with default settings (port 8000)
python -m victor.integrations.api.run_server

# Run on custom port
python -m victor.integrations.api.run_server --port 8080

# Run without HITL endpoints
python -m victor.integrations.api.run_server --no-hitl

# Run with specific workspace
python -m victor.integrations.api.run_server --workspace /path/to/project

# Run with in-memory HITL storage (default: SQLite)
python -m victor.integrations.api.run_server --hitl-in-memory
```

### Programmatic Usage

```python
from victor.integrations.api.unified_server import create_unified_server
import uvicorn

# Create the unified server
app = create_unified_server(
    host="0.0.0.0",
    port=8000,
    workspace_root="/path/to/project",
    enable_hitl=True,
    hitl_persistent=True,
)

# Run the server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## URL Structure

### API Endpoints

```
/api/v1/*                    - Main API endpoints (chat, completions, search, etc.)
/api/v1/hitl/*               - HITL approval endpoints
/api/v1/workflows/*          - Workflow editor endpoints (proxied)
/workflow-editor/*           - Workflow editor original routes
```

### Web UIs

```
/                            - Redirects to /ui
/ui                          - Landing page with links to all UIs
/ui/hitl                     - HITL approval interface
/ui/workflow-editor          - Visual workflow editor
/docs                        - Interactive API documentation (Swagger UI)
/redoc                       - Alternative API documentation (ReDoc)
```

### System

```
/health                      - Health check for all services
/shutdown                    - Graceful server shutdown
```

## Web UIs

### Landing Page (`/ui`)

The landing page provides a central portal to access all Victor web interfaces:

- **Workflow Editor**: Visual workflow creation and management
- **Approvals**: Human-in-the-Loop approval interface
- **API Documentation**: Interactive Swagger/OpenAPI docs

### HITL Approval UI (`/ui/hitl`)

Web-based interface for approving workflow decisions:

- View pending approval requests
- Approve or reject requests
- Add feedback and comments
- Real-time updates
- Approval history

### Workflow Editor (`/ui/workflow-editor`)

Visual workflow editor for StateGraph workflows:

- Drag-and-drop workflow design
- Real-time validation
- YAML import/export
- Team node configuration
- Multiple node types (agent, compute, team, condition, parallel, etc.)

**Note**: The workflow editor frontend requires building:

```bash
# For development
cd tools/workflow_editor/frontend
npm install
npm run dev  # Runs at http://localhost:5173

# For production
cd tools/workflow_editor/frontend
npm run build  # Creates dist/ directory
```

## Configuration

### Command-Line Options

```bash
--host HOST                 Host to bind to (default: 0.0.0.0)
--port PORT                 Port to listen on (default: 8000)
--workspace PATH            Workspace root directory
--no-hitl                   Disable HITL endpoints
--hitl-in-memory            Use in-memory storage for HITL (default: SQLite)
--hitl-auth-token TOKEN     Auth token for HITL endpoints
--log-level LEVEL           Logging level (debug, info, warning, error)
```

### Environment Variables

```bash
# HITL Configuration
HITL_AUTH_TOKEN=your-token-here    # Auth token for HITL endpoints

# API Configuration
VICTOR_API_PORT=8000               # Override default port
VICTOR_API_HOST=0.0.0.0            # Override default host
VICTOR_WORKSPACE=/path/to/project  # Set workspace directory
```

## Examples

### Basic Usage

```python
import httpx

async with httpx.AsyncClient() as client:
    # Chat endpoint
    response = await client.post(
        "http://localhost:8000/api/v1/chat",
        json={"messages": [{"role": "user", "content": "Hello!"}]}
    )
    print(response.json())

    # Semantic search
    response = await client.post(
        "http://localhost:8000/api/v1/search/semantic",
        json={"query": "authentication", "max_results": 10}
    )
    print(response.json())

    # List pending HITL requests
    response = await client.get("http://localhost:8000/api/v1/hitl/requests")
    print(response.json())
```

### WebSocket Streaming

```python
import asyncio
import websockets

async def stream_chat():
    uri = "ws://localhost:8000/api/v1/ws"
    async with websockets.connect(uri) as websocket:
        # Send chat message
        await websocket.send_json({
            "type": "chat",
            "messages": [{"role": "user", "content": "Explain async/await"}]
        })

        # Stream response
        while True:
            data = await websocket.recv_json()
            if data.get("type") == "done":
                break
            print(data.get("content", ""), end="")

asyncio.run(stream_chat())
```

### Workflow Editor API

```python
import httpx

async with httpx.AsyncClient() as client:
    # Get node types
    response = await client.get("http://localhost:8000/api/v1/workflows/nodes/types")
    node_types = response.json()
    print(f"Available node types: {list(node_types.keys())}")

    # Validate workflow
    workflow = {
        "nodes": [
            {"id": "1", "type": "agent", "name": "Researcher", "config": {"role": "researcher"}}
        ],
        "edges": []
    }
    response = await client.post(
        "http://localhost:8000/api/v1/workflows/validate",
        json=workflow
    )
    print(response.json())
```

## Health Check

The `/health` endpoint provides status for all integrated services:

```json
{
  "status": "healthy",
  "services": {
    "main_api": "healthy",
    "hitl": {
      "status": "healthy",
      "pending_requests": 3
    },
    "workflow_editor": "available"
  }
}
```

## CORS Configuration

By default, CORS is enabled for development with permissive settings:

```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

For production, configure specific origins:

```python
from victor.integrations.api.unified_server import create_unified_server

app = create_unified_server(enable_cors=False)

# Add custom CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Deployment

### Development

```bash
# Run with auto-reload
uvicorn victor.integrations.api.unified_server:create_unified_server --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Run with multiple workers
gunicorn victor.integrations.api.unified_server:create_unified_server \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install victor-ai[api]

COPY . .

EXPOSE 8000
CMD ["python", "-m", "victor.integrations.api.run_server", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  victor-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HITL_AUTH_TOKEN=${HITL_AUTH_TOKEN}
      - VICTOR_WORKSPACE=/workspace
    volumes:
      - ./workspace:/workspace
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python -m victor.integrations.api.run_server --port 8080
```

### HITL SQLite Database Error

```bash
# Remove existing database
rm ~/.victor/hitl.db

# Or use in-memory storage
python -m victor.integrations.api.run_server --hitl-in-memory
```

### Workflow Editor Not Loading

The workflow editor frontend requires building:

```bash
cd tools/workflow_editor/frontend
npm install
npm run build
```

For development, run the Vite dev server separately:

```bash
cd tools/workflow_editor/frontend
npm run dev
```

Then access the editor at `http://localhost:5173`.

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Unified FastAPI Application                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Main API    │  │ HITL API     │  │ Workflow Editor │  │
│  │ /api/v1/*   │  │ /api/v1/hitl │  │ /workflow-editor│  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 Frontend UI Routes                      │ │
│  │  /ui -> Landing page                                   │ │
│  │  /ui/hitl -> Approval interface                        │ │
│  │  /ui/workflow-editor -> Visual editor                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Contributing

To add a new service to the unified server:

1. Create a new function `_include_<service>_api(app: FastAPI)`
2. Import and mount the service's FastAPI app or router
3. Add configuration options to `create_unified_server()`
4. Update this README with the new service's endpoints

Example:

```python
def _include_my_service_api(app: FastAPI) -> None:
    """Include my custom service API routes."""
    from my_service.api import create_router

    router = create_router()
    app.include_router(router, prefix="/api/v1/my-service", tags=["MyService"])
    logger.info("MyService routes mounted at /api/v1/my-service")
```

Then call it in `create_unified_server()`:

```python
def create_unified_server(...) -> FastAPI:
    app = FastAPI(...)

    # Include main API
    _include_main_api(app, workspace_root)

    # Include HITL
    if enable_hitl:
        _include_hitl_api(app, hitl_persistent, hitl_auth_token)

    # Include workflow editor
    _include_workflow_editor_api(app)

    # Include your service
    _include_my_service_api(app)

    return app
```

## License

Apache License 2.0 - see LICENSE file for details.
