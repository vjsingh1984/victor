# Victor Unified Server - Quick Reference

## Quick Start

```bash
# Start the unified server
python -m victor.integrations.api.run_server

# Access the UIs
open http://localhost:8000/ui
```

## Common Commands

```bash
# Run on custom port
python -m victor.integrations.api.run_server --port 8080

# Disable HITL endpoints
python -m victor.integrations.api.run_server --no-hitl

# Use specific workspace
python -m victor.integrations.api.run_server --workspace /path/to/project

# Debug mode
python -m victor.integrations.api.run_server --log-level debug
```

## URLs

### Web UIs
- **Landing Page**: http://localhost:8000/ui
- **HITL Approvals**: http://localhost:8000/ui/hitl
- **Workflow Editor**: http://localhost:8000/ui/workflow-editor
- **API Documentation**: http://localhost:8000/docs

### API Endpoints
- **Main API**: http://localhost:8000/api/v1/*
- **HITL API**: http://localhost:8000/api/v1/hitl/*
- **Workflow Editor**: http://localhost:8000/api/v1/workflows/*

### System
- **Health Check**: http://localhost:8000/health
- **Server Info**: http://localhost:8000/

## API Examples

### Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Semantic Search

```bash
curl -X POST http://localhost:8000/api/v1/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication", "max_results": 10}'
```

### HITL - List Pending Requests

```bash
curl http://localhost:8000/api/v1/hitl/requests
```

### HITL - Submit Approval

```bash
curl -X POST http://localhost:8000/api/v1/hitl/respond/request-id \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "reason": "Looks good"}'
```

### Workflow - Get Node Types

```bash
curl http://localhost:8000/api/v1/workflows/nodes/types
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Python Examples

### Basic Chat

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/chat",
        json={"messages": [{"role": "user", "content": "Explain async/await"}]}
    )
    print(response.json())
```

### Streaming Chat

```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream(
        "POST",
        "http://localhost:8000/api/v1/chat/stream",
        json={"messages": [{"role": "user", "content": "Hello"}]}
    ) as response:
        async for chunk in response.aiter_text():
            if chunk:
                print(chunk, end="")
```

### WebSocket Connection

```python
import websockets
import json

async def chat_ws():
    async with websockets.connect("ws://localhost:8000/api/v1/ws") as ws:
        # Send message
        await ws.send_json({
            "type": "chat",
            "messages": [{"role": "user", "content": "Hello!"}]
        })

        # Receive response
        while True:
            data = await ws.recv_json()
            if data.get("type") == "done":
                break
            print(data.get("content", ""), end="")
```

## Testing

### Run Test Suite

```bash
python victor/integrations/api/test_unified_server.py
```

### Manual Testing Checklist

- [ ] Server starts without errors
- [ ] Landing page loads at http://localhost:8000/ui
- [ ] API docs load at http://localhost:8000/docs
- [ ] Health check returns 200 at http://localhost:8000/health
- [ ] Chat endpoint works
- [ ] HITL UI loads at http://localhost:8000/ui/hitl
- [ ] Workflow editor loads at http://localhost:8000/ui/workflow-editor
- [ ] WebSocket connection works

## Troubleshooting

### Port Already in Use

```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
python -m victor.integrations.api.run_server --port 8080
```

### HITL Database Error

```bash
# Remove database and restart
rm ~/.victor/hitl.db
python -m victor.integrations.api.run_server
```

### Workflow Editor Not Loading

```bash
# Build the frontend
cd tools/workflow_editor/frontend
npm install
npm run build
```

### Import Errors

```bash
# Reinstall Victor with API dependencies
pip install victor-ai[api] --force-reinstall
```

## Configuration

### Environment Variables

```bash
# HITL authentication
export HITL_AUTH_TOKEN=your-secret-token

# Server settings
export VICTOR_API_PORT=8000
export VICTOR_API_HOST=0.0.0.0
export VICTOR_WORKSPACE=/path/to/project
```

### Programmatic Configuration

```python
from victor.integrations.api.unified_server import create_unified_server

app = create_unified_server(
    host="0.0.0.0",
    port=8000,
    workspace_root="/path/to/project",
    enable_hitl=True,
    hitl_persistent=True,
    hitl_auth_token="secret",
)
```

## Production Deployment

### Gunicorn

```bash
gunicorn victor.integrations.api.unified_server:create_unified_server \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile - \
    --error-logfile -
```

### Docker

```bash
# Build
docker build -t victor-api .

# Run
docker run -p 8000:8000 victor-api
```

### Docker Compose

```yaml
services:
  victor:
    image: victor-api
    ports:
      - "8000:8000"
    environment:
      - HITL_AUTH_TOKEN=${HITL_TOKEN}
      - VICTOR_WORKSPACE=/workspace
    volumes:
      - ./workspace:/workspace
```

## More Information

- **Full Documentation**: `victor/integrations/api/UNIFIED_SERVER_README.md`
- **Implementation Summary**: `victor/integrations/api/UNIFICATION_SUMMARY.md`
- **Test Suite**: `victor/integrations/api/test_unified_server.py`

## Support

- **GitHub Issues**: https://github.com/vijaydsingh/victor/issues
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
