# API Server Migration Guide

## Overview

This document provides a comprehensive guide for migrating from the legacy FastAPI server (`web/server/main.py`) to the canonical FastAPI server (`victor/integrations/api/fastapi_server.py`).

**Status:** The canonical server (`victor/integrations/api/fastapi_server.py`) is the recommended implementation with 3506 lines of production-ready code. The legacy server (`web/server/main.py`) has 695 lines and is slated for deprecation.

---

## Executive Summary

| Aspect | Canonical Server | Legacy Server |
|--------|-----------------|---------------|
| **Location** | `victor/integrations/api/fastapi_server.py` | `web/server/main.py` |
| **Lines of Code** | 3,506 | 695 |
| **Architecture** | Class-based (`VictorFastAPIServer`) | Functional (module-level) |
| **Session Management** | Advanced (HMAC-signed tokens, TTL, cleanup) | Basic (dict-based) |
| **WebSocket Support** | Multi-channel (chat, events, workflows) | Single-channel (/ws) |
| **Authentication** | Bearer token + query param API key | Bearer token + query param API key |
| **CORS Configuration** | Regex-based origin matching | Simple list-based |
| **Endpoint Count** | 80+ REST + 3 WebSocket | 8 REST + 1 WebSocket + 6 render |
| **Feature Parity** | 100% (all Victor features) | ~15% (basic chat + render) |
| **Documentation** | Full OpenAPI/Swagger | Minimal |
| **Background Tasks** | Agent manager, cleanup, RL feedback | Idle session cleanup only |
| **HITL Support** | Yes (optional) | No |
| **Workflow Visualization** | Yes (Cytoscape.js HTML) | No |

---

## Detailed Feature Comparison

### 1. REST API Endpoints

#### Canonical Server Endpoints

**System & Health:**
- `GET /health` - Health check with active session count
- `GET /status` - Server status (provider, model, mode, workspace)
- `POST /shutdown` - Graceful shutdown with RL feedback recording

**Chat & Completions:**
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - Server-Sent Events (SSE) streaming
- `POST /completions` - Code completions with FIM support

**Search:**
- `POST /search/semantic` - Semantic code search
- `POST /search/code` - Regex/literal code search (ripgrep)

**Model & Mode Management:**
- `POST /model/switch` - Switch AI model
- `POST /mode/switch` - Switch agent mode
- `GET /models` - List available models
- `GET /providers` - List LLM providers
- `GET /capabilities` - Discover all Victor capabilities

**Tools:**
- `GET /tools` - List available tools
- `POST /tools/approve` - Approve/reject tool execution
- `GET /tools/pending` - List pending approvals

**Conversation Management:**
- `POST /conversation/reset` - Reset conversation history
- `GET /conversation/export` - Export conversation (JSON/markdown)

**History & Undo:**
- `POST /undo` - Undo last change
- `POST /redo` - Redo last undone change
- `GET /history` - Get change history

**Patch Operations:**
- `POST /patch/apply` - Apply a patch
- `POST /patch/create` - Create a patch

**LSP:**
- `POST /lsp/completions` - LSP completions
- `POST /lsp/hover` - LSP hover
- `POST /lsp/definition` - LSP definition
- `POST /lsp/references` - LSP references
- `POST /lsp/diagnostics` - LSP diagnostics

**Git Integration:**
- `GET /git/status` - Git status
- `POST /git/commit` - Create commit (with AI message generation)
- `GET /git/log` - Commit log
- `GET /git/diff` - Git diff

**Terminal:**
- `POST /terminal/suggest` - Suggest terminal command
- `POST /terminal/execute` - Execute terminal command

**Workspace Analysis:**
- `GET /workspace/overview` - Workspace structure
- `GET /workspace/metrics` - Code metrics
- `GET /workspace/security` - Security scan
- `GET /workspace/dependencies` - Dependency information

**MCP (Model Context Protocol):**
- `GET /mcp/servers` - List MCP servers
- `POST /mcp/connect` - Connect to MCP server
- `POST /mcp/disconnect` - Disconnect from MCP server

**RL (Reinforcement Learning):**
- `GET /rl/stats` - RL model selector statistics
- `GET /rl/recommend` - Get model recommendation
- `POST /rl/explore` - Set exploration rate
- `POST /rl/strategy` - Set selection strategy
- `POST /rl/reset` - Reset Q-values

**Background Agents:**
- `POST /agents/start` - Start background agent
- `GET /agents` - List agents
- `GET /agents/{agent_id}` - Get agent status
- `POST /agents/{agent_id}/cancel` - Cancel agent
- `POST /agents/clear` - Clear completed agents

**Plans:**
- `GET /plans` - List plans
- `POST /plans` - Create plan
- `GET /plans/{plan_id}` - Get plan details
- `POST /plans/{plan_id}/approve` - Approve plan
- `POST /plans/{plan_id}/execute` - Execute plan
- `DELETE /plans/{plan_id}` - Delete plan

**Teams:**
- `POST /teams` - Create team
- `GET /teams` - List teams
- `GET /teams/{team_id}` - Get team details
- `POST /teams/{team_id}/start` - Start team execution
- `POST /teams/{team_id}/cancel` - Cancel team
- `POST /teams/clear` - Clear completed teams
- `GET /teams/{team_id}/messages` - Get team messages

**Workflows:**
- `GET /workflows/templates` - List workflow templates
- `GET /workflows/templates/{template_id}` - Get template details
- `POST /workflows/execute` - Execute workflow
- `GET /workflows/executions` - List executions
- `GET /workflows/executions/{execution_id}` - Get execution details
- `POST /workflows/executions/{execution_id}/cancel` - Cancel execution
- `GET /workflows/{workflow_id}/graph` - Get workflow graph (Cytoscape.js)
- `GET /workflows/{workflow_id}/execution` - Get execution state
- `GET /workflows/visualize/{workflow_id}` - HTML visualization UI
- `GET /workflows/{workflow_id}/stream` - WebSocket event stream

**Legacy Compatibility:**
- `GET /credentials/get` - Placeholder credentials endpoint
- `POST /session/token` - Placeholder session token endpoint

#### Legacy Server Endpoints

**System & Health:**
- `GET /health` - Health check (basic)

**Chat & Session:**
- `POST /session/token` - Issue session token (HMAC-signed)
- `GET /history` - Empty history list (placeholder)
- `GET /credentials/get` - Placeholder credentials
- `GET /models` - Placeholder models
- `GET /providers` - Placeholder providers
- `GET /rl/stats` - Placeholder RL stats

**Diagram Rendering:**
- `POST /render/plantuml` - Render PlantUML to SVG
- `POST /render/mermaid` - Render Mermaid to SVG
- `POST /render/drawio` - Render Draw.io to SVG
- `POST /render/graphviz` - Render Graphviz DOT to SVG
- `POST /render/d2` - Render D2 diagram to SVG

---

### 2. WebSocket Support Comparison

#### Canonical Server

**Three WebSocket Endpoints:**

1. **`/ws` - Primary WebSocket**
   - Purpose: General bidirectional communication
   - Message types: `chat`, `ping`, `auth`, `subscribe`
   - Features:
     - Authentication via message (after connection)
     - JSON-based message protocol
     - Real-time streaming
     - Connection tracking

2. **`/ws/events` - EventBridge WebSocket**
   - Purpose: Real-time Victor event streaming
   - Features:
     - Subscribe to event categories
     - Tool execution events
     - File change events
     - Provider updates
     - Ping/pong heartbeat
   - Message types:
     - Incoming: `{"type": "subscribe", "categories": ["all"]}`
     - Outgoing: `{"type": "event", "event": {...}}`

3. **`/workflows/{workflow_id}/stream` - Workflow Event Stream**
   - Purpose: Real-time workflow execution updates
   - Features:
     - Node start/complete/error events
     - Progress updates
     - Completion notifications
   - Integrates with: `WorkflowEventBridge`

**WebSocket Features:**
- Client tracking (`_ws_clients`, `_event_clients`)
- Broadcast to all clients
- Per-client message handling
- Graceful disconnect handling

#### Legacy Server

**Single WebSocket Endpoint:**

1. **`/ws` - Chat WebSocket**
   - Purpose: Chat session only
   - Protocol: Plain text messages (not JSON)
   - Features:
     - API key validation (header or query param)
     - Session token validation
     - Session reuse with HMAC-signed tokens
     - Agent per session
     - Message size limits
     - Receive timeout (5 minutes)
     - Heartbeat (30s interval)
     - Idle session cleanup (1 hour)
   - Special commands:
     - `__reset_session__` - Reset conversation
   - Response format:
     - Plain text content chunks
     - `[session] <token>` - Session token
     - `[ping]` - Heartbeat
     - `[error] <message>` - Errors
     - Empty string - Stream completion

**WebSocket Features:**
- Session-aware agent reuse
- Background heartbeat loop
- Idle session cleanup task
- Graceful shutdown

---

### 3. Session Management

#### Canonical Server

**Minimal Session Support:**
- No built-in session persistence
- No session tokens
- No session limits
- Relies on client-side state management
- Optional HITL session storage (SQLite if enabled)

**Use Cases:**
- Stateless REST API
- Client-managed conversations
- Workflow executions (in-memory)

#### Legacy Server

**Advanced Session Management:**

**Session Storage:**
```python
SESSION_AGENTS: Dict[str, Dict[str, Any]] = {
    session_id: {
        "agent": AgentOrchestrator,
        "created_at": float,
        "last_activity": float,
        "connection_count": int,
        "session_token": str,
    }
}
SESSION_TOKENS: Dict[str, str]  # token -> session_id
```

**Session Token Format:**
- HMAC-signed tokens with SHA-256
- Format: `base64(session_id:issued_at:signature)`
- TTL: Configurable (default from settings)
- Validation: Signature check + expiration check

**Session Lifecycle:**
1. Token issuance: `POST /session/token`
2. WebSocket connection with `?session_token=<token>`
3. Agent reuse from `SESSION_AGENTS`
4. Activity tracking via `last_activity`
5. Background cleanup of idle sessions (1 hour)

**Session Limits:**
- Maximum sessions: `server_max_sessions` (from settings)
- Message size: `server_max_message_bytes`
- Render payload: `render_max_payload_bytes`
- Render concurrency: `render_max_concurrency`

**Cleanup Process:**
- Background task runs every 5 minutes
- Removes sessions idle > 1 hour
- Graceful agent shutdown
- Provider cleanup

---

### 4. Authentication & Authorization

#### Canonical Server

**Authentication Methods:**
- API key via `Authorization: Bearer <token>` header
- API key via `?api_key=<token>` query parameter
- WebSocket authentication via first message: `{"type": "auth", "api_key": "<token>"}`

**Authorization:**
- No role-based access control
- No user management
- Simple API key check

**CORS Configuration:**
```python
cors_origins = [
    "http://localhost:*",
    "http://127.0.0.1:*",
    "https://localhost:*",
    "https://127.0.0.1:*",
    "vscode-webview://*",
]
allow_origin_regex=r"^(http://localhost:\d+|http://127\.0\.0\.1:\d+|vscode-webview://[a-z0-9-]+)$"
```

#### Legacy Server

**Authentication Methods:**
- API key via `Authorization: Bearer <token>` header
- API key via `?api_key=<token>` query parameter
- FastAPI dependency: `_require_api_key()`

**Authorization:**
- Same as canonical (simple API key check)

**CORS Configuration:**
```python
# Environment variable: CORS_ORIGINS
# Default: localhost:5173, 127.0.0.1:5173
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
```

**Security:**
- API key validation for all endpoints
- API key validation for WebSocket (before accept)
- Message size limits
- Command validation (dangerous pattern detection)

---

### 5. CORS Configuration

#### Canonical Server

**Features:**
- Regex-based origin matching
- Supports dynamic port numbers
- VS Code webview support
- Wildcard subdomain support

**Configuration:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^(http://localhost:\d+|http://127\.0\.0\.1:\d+|vscode-webview://[a-z0-9-]+)$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
```

#### Legacy Server

**Features:**
- Simple list-based origin matching
- Environment variable configuration
- No regex support
- Fixed port numbers

**Configuration:**
```python
# From CORS_ORIGINS environment variable
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    allowed_origins = [origin.strip() for origin in cors_origins_env.split(",")]
else:
    allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Migration Impact:**
- Canonical server supports dynamic ports (better for development)
- Legacy server requires explicit port configuration
- Canonical server has better VS Code integration

---

### 6. Background Tasks & Lifecycle

#### Canonical Server

**Startup Tasks:**
```python
@asynccontextmanager
async def _lifespan(self, app: FastAPI):
    # Initialize EventBridge for real-time events
    self._event_bridge = EventBridge(event_bus)
    self._event_bridge.start()

    # Initialize WorkflowEventBridge for workflow visualization
    self._workflow_event_bridge = WorkflowEventBridge(event_bus)
    await self._workflow_event_bridge.start()

    yield

    # Cleanup
    if self._event_bridge:
        self._event_bridge.stop()
    if self._workflow_event_bridge:
        await self._workflow_event_bridge.stop()
    if self._orchestrator:
        await self._orchestrator.graceful_shutdown()
    # Close WebSocket connections
    for ws in self._ws_clients + self._event_clients:
        await ws.close()
```

**Background Features:**
- EventBridge: Real-time event streaming
- WorkflowEventBridge: Workflow visualization
- BackgroundAgentManager: Async agent execution
- RL feedback recording on shutdown

**Shutdown Process:**
1. Cancel background tasks
2. Close WebSocket connections
3. Stop event bridges
4. Shutdown orchestrator
5. Record RL feedback

#### Legacy Server

**Startup Tasks:**
```python
@app.on_event("startup")
async def startup_event():
    # Start idle session cleanup task
    task = asyncio.create_task(cleanup_idle_sessions())
    _background_tasks.append(task)
```

**Background Features:**
- `heartbeat_loop()`: Send pings every 30s
- `cleanup_idle_sessions()`: Remove idle sessions every 5 min
- Session cleanup agent shutdown

**Shutdown Process:**
```python
@app.on_event("shutdown")
async def shutdown_event():
    # Cancel background tasks
    for task in _background_tasks:
        task.cancel()
        await task

    # Clean up all active sessions
    for session_id, session_data in SESSION_AGENTS.items():
        agent = session_data.get("agent")
        if agent:
            if hasattr(agent, "shutdown"):
                await agent.shutdown()
            if hasattr(agent, "provider"):
                await agent.provider.close()
```

---

### 7. Special Features

#### Canonical Server

**1. Workflow Visualization**
- HTML page with Cytoscape.js integration
- Real-time workflow execution updates
- Graph export in Cytoscape.js format
- Execution state tracking
- Node-level progress

**2. HITL (Human-in-the-Loop)**
- Optional approval workflows
- SQLite or in-memory storage
- Web-based approval UI
- Approval history tracking

**3. Background Agents**
- Asynchronous agent execution
- Maximum 4 concurrent agents
- Real-time progress updates via WebSocket
- Agent lifecycle management (start, cancel, clear)

**4. Teams API**
- Multi-agent team creation
- Multiple formation types (sequential, parallel, hierarchical, pipeline)
- Team execution tracking
- Inter-agent message tracking

**5. RL Model Selector**
- Q-learning based provider selection
- Task-specific Q-tables
- Exploration rate control
- Strategy selection (epsilon-greedy, UCB, softmax)
- Provider rankings

**6. Advanced Code Completions**
- FIM (Fill-in-the-Middle) support
- Prefix + suffix context
- Low latency (bypasses orchestrator)
- Deterministic output (temperature=0)
- Custom stop sequences

**7. Git Integration**
- AI-generated commit messages
- Diff viewing (staged/unstaged)
- Commit log
- Status tracking

**8. Workspace Analysis**
- Code metrics (LOC, file types, largest files)
- Security scanning (secret detection)
- Dependency analysis (Python, Node, Rust, Go)
- Structure overview

**9. Terminal Command Execution**
- Command suggestion from intent
- Dangerous command detection
- Async execution with timeout
- Output size limiting

**10. LSP Integration**
- Completions
- Hover information
- Go-to-definition
- Find references
- Diagnostics

#### Legacy Server

**1. Diagram Rendering**
- PlantUML, Mermaid, Draw.io, Graphviz, D2 support
- CLI-based rendering (subprocess)
- Payload size limits
- Timeout protection
- Concurrency limits

**2. Session Management**
- HMAC-signed tokens
- Agent reuse across connections
- Activity tracking
- Idle cleanup

**3. Basic Chat**
- Streaming responses
- Message size limits
- Receive timeout
- Heartbeat monitoring

---

## Migration Guide

### Phase 1: Preparation (Week 1)

#### 1.1 Assess Current Usage

**Inventory Checks:**

```bash
# Check which endpoints your clients use
grep -r "POST /render" web/
grep -r "WebSocket.*ws" web/
grep -r "session_token" web/
grep -r "/chat" web/
```

**Client Types to Consider:**
- VS Code extension
- JetBrains plugin
- Web UI (React/Vue)
- CLI tools
- Custom integrations

#### 1.2 Feature Gap Analysis

**Identify Missing Features:**

| Legacy Feature | Canonical Equivalent | Migration Required |
|----------------|---------------------|-------------------|
| Diagram rendering endpoints | None | **YES** - Need to add |
| Session management | Stateless | **MAYBE** - If you need session persistence |
| WebSocket plain text protocol | JSON protocol | **YES** - Client protocol change |
| `/render/*` endpoints | None | **YES** - Need to add |

#### 1.3 Dependency Check

```bash
# Check if diagram CLI tools are installed
which plantuml
which mmdc
which drawio
which dot
which d2
```

### Phase 2: Add Missing Features (Week 2)

#### 2.1 Add Diagram Rendering to Canonical Server

**Create new file:** `victor/integrations/api/diagram_rendering.py`

```python
"""Diagram rendering endpoints for canonical server."""

import subprocess
import tempfile
import asyncio
from fastapi import Body, Depends, Response, HTTPException
from fastapi.concurrency import run_in_threadpool

# Copy render functions from legacy server
# _render_plantuml_svg, _render_mermaid_svg, etc.

def setup_diagram_endpoints(app, settings):
    """Setup diagram rendering endpoints."""

    @app.post("/render/plantuml")
    async def render_plantuml(
        payload: str = Body(..., media_type="text/plain"),
    ) -> Response:
        """Render PlantUML diagram to SVG."""
        # Implementation from legacy server
        ...

    # Add other render endpoints...
```

**Import in canonical server:**

```python
# In victor/integrations/api/fastapi_server.py

from victor.integrations.api.diagram_rendering import setup_diagram_endpoints

# In __init__ after _setup_routes():
setup_diagram_endpoints(self.app, settings)
```

#### 2.2 Optional: Add Session Management

If you need session persistence, add this to canonical server:

```python
# In VictorFastAPIServer.__init__
self._session_store: Dict[str, Dict[str, Any]] = {}
self._session_secret = settings.server_session_secret or secrets.token_hex(32)

# Add session endpoints
@app.post("/session/token")
async def issue_session_token(
    session_id: Optional[str] = None
) -> Dict[str, str]:
    """Issue signed session token."""
    # Implementation from legacy server
    ...

@app.websocket("/ws/session")
async def session_websocket(websocket: WebSocket, session_token: str):
    """WebSocket with session support."""
    # Implementation from legacy server
    ...
```

### Phase 3: Client Migration (Week 3)

#### 3.1 Update WebSocket Clients

**Legacy Protocol:**
```javascript
// Legacy: Plain text WebSocket
const ws = new WebSocket("ws://localhost:8765/ws");
ws.onmessage = (event) => {
    const text = event.data;
    if (text.startsWith("[session]")) {
        const token = text.split(" ")[1];
        // Handle session
    } else {
        // Plain text content
        console.log(text);
    }
};
ws.send("Hello, Victor!");
```

**Canonical Protocol:**
```javascript
// Canonical: JSON WebSocket
const ws = new WebSocket("ws://localhost:8765/ws");
ws.onopen = () => {
    // Authenticate first
    ws.send(JSON.stringify({
        type: "auth",
        api_key: "your-api-key"
    }));
};
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch(data.type) {
        case "auth_success":
            console.log("Authenticated");
            break;
        case "content":
            console.log(data.content);
            break;
        case "tool_call":
            console.log("Tool:", data.tool_call);
            break;
        case "done":
            console.log("Stream complete");
            break;
        case "error":
            console.error("Error:", data.message);
            break;
    }
};
ws.send(JSON.stringify({
    type: "chat",
    messages: [{role: "user", content: "Hello, Victor!"}]
}));
```

#### 3.2 Update REST Clients

**Legacy:**
```python
import requests

# Chat endpoint (doesn't exist in legacy)
response = requests.post("http://localhost:8765/chat", json={
    "messages": [{"role": "user", "content": "Hello"}]
})
```

**Canonical:**
```python
import requests

# Chat endpoint
response = requests.post(
    "http://localhost:8765/chat",
    json={"messages": [{"role": "user", "content": "Hello"}]},
    headers={"Authorization": "Bearer your-api-key"}
)
data = response.json()
print(data["content"])

# Streaming chat
import requests
response = requests.post(
    "http://localhost:8765/chat/stream",
    json={"messages": [{"role": "user", "content": "Hello"}]},
    headers={"Authorization": "Bearer your-api-key"},
    stream=True
)
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data == "[DONE]":
                break
            print(data.get("content", ""), end="")
```

#### 3.3 Update Diagram Rendering Clients

**No change needed** if you add the rendering endpoints to canonical server (Phase 2.1).

```python
# Same for both servers
import requests

response = requests.post(
    "http://localhost:8765/render/plantuml",
    data="@startuml\nAlice -> Bob: Hello\n@enduml",
    headers={"Content-Type": "text/plain"}
)
svg = response.content
with open("diagram.svg", "wb") as f:
    f.write(svg)
```

### Phase 4: Testing & Validation (Week 4)

#### 4.1 Unit Tests

```python
# tests/integrations/api/test_fastapi_server.py

import pytest
from victor.integrations.api.fastapi_server import VictorFastAPIServer
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check():
    server = VictorFastAPIServer()
    async with AsyncClient(app=server.app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_chat_endpoint():
    server = VictorFastAPIServer()
    async with AsyncClient(app=server.app, base_url="http://test") as client:
        response = await client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "test"}]}
        )
        assert response.status_code == 200
        assert "content" in response.json()

@pytest.mark.asyncio
async def test_diagram_rendering():
    server = VictorFastAPIServer()
    async with AsyncClient(app=server.app, base_url="http://test") as client:
        response = await client.post(
            "/render/plantuml",
            data="@startuml\nA->B: Test\n@enduml",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
```

#### 4.2 Integration Tests

```python
# tests/integrations/api/test_websocket.py

import pytest
import asyncio
from fastapi.testclient import TestClient
from victor.integrations.api.fastapi_server import VictorFastAPIServer

@pytest.mark.asyncio
async def test_websocket_chat():
    server = VictorFastAPIServer()
    with TestClient(app=server.app).websocket_connect("/ws") as websocket:
        # Authenticate
        websocket.send_json({"type": "auth", "api_key": "test-key"})
        data = websocket.receive_json()
        assert data["type"] == "auth_success"

        # Send chat message
        websocket.send_json({
            "type": "chat",
            "messages": [{"role": "user", "content": "Hello"}]
        })

        # Receive response
        done = False
        while not done:
            data = websocket.receive_json()
            if data["type"] == "content":
                print(data["content"])
            elif data["type"] == "done":
                done = True
```

#### 4.3 Performance Tests

```python
# tests/integrations/api/test_performance.py

import pytest
import time
import requests

def test_chat_latency():
    """Test chat endpoint latency."""
    server = VictorFastAPIServer()
    start = time.time()
    response = requests.post(
        "http://localhost:8765/chat",
        json={"messages": [{"role": "user", "content": "test"}]}
    )
    latency = (time.time() - start) * 1000
    assert latency < 5000  # Should be under 5 seconds
    print(f"Chat latency: {latency:.2f}ms")

def test_concurrent_requests():
    """Test concurrent request handling."""
    import concurrent.futures

    def make_request():
        response = requests.post(
            "http://localhost:8765/chat",
            json={"messages": [{"role": "user", "content": "test"}]}
        )
        return response.status_code == 200

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        success_rate = sum(results) / len(results) * 100
        assert success_rate > 95  # At least 95% success
```

### Phase 5: Deployment (Week 5)

#### 5.1 Configuration Migration

**Legacy Configuration:**
```bash
# .env or environment
CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
API_KEY=your-secret-key
SESSION_SECRET=your-session-secret
SESSION_TTL=3600
MAX_SESSIONS=100
```

**Canonical Configuration:**
```bash
# victor/config/settings.py or .env
VICTOR_SERVER_API_KEY=your-secret-key
VICTOR_SERVER_CORS_ENABLED=true
VICTOR_SERVER_HOST=127.0.0.1
VICTOR_SERVER_PORT=8765
VICTOR_SERVER_RATE_LIMIT_RPM=60  # Optional
```

#### 5.2 Deployment Steps

1. **Install dependencies:**
   ```bash
   pip install victor-ai[api]  # Ensures FastAPI is installed
   ```

2. **Update systemd service:**
   ```ini
   [Unit]
   Description=Victor AI Coding Assistant (Canonical Server)
   After=network.target

   [Service]
   Type=simple
   User=victor
   WorkingDirectory=/opt/victor
   Environment="PATH=/opt/victor/venv/bin"
   ExecStart=/opt/victor/venv/bin/python -m victor.integrations.api.fastapi_server
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

3. **Run the server:**
   ```bash
   # Direct run
   python -m victor.integrations.api.fastapi_server

   # Or with uvicorn
   uvicorn victor.integrations.api.fastapi_server:create_fastapi_app --host 0.0.0.0 --port 8765
   ```

4. **Verify deployment:**
   ```bash
   curl http://localhost:8765/health
   curl http://localhost:8765/status
   ```

#### 5.3 Monitoring

**Health Checks:**
```bash
# Basic health
curl http://localhost:8765/health

# Detailed status
curl http://localhost:8765/status

# Active sessions (if session support added)
curl http://localhost:8765/stats
```

**Metrics to Monitor:**
- Request latency
- WebSocket connection count
- Active agent count
- Error rates
- Memory usage

---

## Breaking Changes & Workarounds

### 1. WebSocket Protocol Change

**Breaking Change:** WebSocket protocol changed from plain text to JSON.

**Impact:** All WebSocket clients need to be updated.

**Workaround:**
```python
# Add compatibility layer to canonical server
@app.websocket("/ws/legacy")
async def legacy_websocket(websocket: WebSocket):
    """Legacy WebSocket endpoint for plain text protocol."""
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            # Handle plain text, respond with plain text
            # Convert to JSON internally
            ...
    except WebSocketDisconnect:
        pass
```

### 2. Session Token Format

**Breaking Change:** Session tokens use different format/signing.

**Impact:** Existing session tokens invalid after migration.

**Workaround:**
- Implement token migration endpoint
- Or require all clients to re-authenticate
- Or keep legacy session token validation running in parallel

### 3. Diagram Rendering Endpoints

**Breaking Change:** Diagram rendering endpoints not in canonical server.

**Impact:** Clients using `/render/*` endpoints will get 404.

**Workaround:**
- Copy rendering endpoints to canonical server (recommended)
- Or run legacy server on separate port for rendering only

### 4. Response Format Changes

**Breaking Change:** Some endpoints return different response formats.

**Impact:** Clients expecting specific response structures may break.

**Examples:**

| Endpoint | Legacy Response | Canonical Response |
|----------|----------------|-------------------|
| `GET /models` | `{"models": []}` | `{"models": [{provider, model_id, ...}]}` |
| `POST /chat` | N/A | `{"role": "assistant", "content": "...", "tool_calls": [...]}` |
| `GET /providers` | `{"providers": []}` | `{"providers": [{name, configured, supports_tools, ...}]}` |

**Workaround:**
- Update clients to handle new response formats
- Or add response transformation middleware

---

## Recommendations

### For New Projects

**Use the canonical server** (`victor/integrations/api/fastapi_server.py`) because:

1. **Feature Complete:** 80+ endpoints covering all Victor capabilities
2. **Well-Documented:** Auto-generated OpenAPI/Swagger docs
3. **Modern Architecture:** Class-based, extensible design
4. **Production-Ready:** Proper error handling, logging, lifecycle management
5. **Future-Proof:** Active maintenance, new features added first

### For Existing Projects

**Migration Strategy:**

**Option 1: Big Bang Migration (Recommended for small projects)**
- Duration: 4-5 weeks
- Downtime: Minimal (use feature flags)
- Risk: Medium
- Steps:
  1. Add missing features to canonical server (Week 2)
  2. Deploy canonical server alongside legacy (different port)
  3. Update clients one-by-one (Week 3)
  4. Cut over to canonical server (Week 4)
  5. Decommission legacy server (Week 5)

**Option 2: Gradual Migration (Recommended for large projects)**
- Duration: 8-12 weeks
- Downtime: None
- Risk: Low
- Steps:
  1. Set up API gateway/load balancer
  2. Route specific endpoints to canonical server
  3. Add session compatibility layer
  4. Migrate clients incrementally
  5. Monitor and adjust
  6. Full cutover once all clients migrated

**Option 3: Hybrid Approach (If diagram rendering is critical)**
- Keep legacy server running for diagram rendering only
- Run canonical server for all other features
- Use nginx/API gateway to route:
  - `/render/*` → legacy server (port 8766)
  - `/*` → canonical server (port 8765)

**nginx configuration:**
```nginx
upstream canonical {
    server localhost:8765;
}

upstream legacy {
    server localhost:8766;
}

server {
    listen 80;

    # Route diagram rendering to legacy
    location /render/ {
        proxy_pass http://legacy;
    }

    # Route everything else to canonical
    location / {
        proxy_pass http://canonical;
    }
}
```

### Feature Parity Roadmap

**Phase 1: Core Features (Week 1-2)**
- ✅ Chat endpoints
- ✅ WebSocket communication
- ✅ Health checks
- ⚠️ Diagram rendering (need to add)

**Phase 2: Advanced Features (Week 3-4)**
- ✅ Background agents
- ✅ Workflow execution
- ✅ Team coordination
- ⚠️ Session management (optional, stateless preferred)

**Phase 3: Integration Features (Week 5-6)**
- ✅ LSP integration
- ✅ Git operations
- ✅ Terminal commands
- ✅ Workspace analysis

**Phase 4: Monitoring & Ops (Week 7-8)**
- ✅ RL feedback
- ✅ Metrics collection
- ✅ Graceful shutdown
- ⚠️ Session cleanup (if needed)

---

## Decision Matrix

### When to Use Canonical Server

| Scenario | Recommendation |
|----------|---------------|
| New project | ✅ Use canonical |
| Need workflow visualization | ✅ Use canonical |
| Need background agents | ✅ Use canonical |
| Need teams/coordination | ✅ Use canonical |
| Stateless REST API | ✅ Use canonical |
| Full IDE integration | ✅ Use canonical |

### When to Use Legacy Server

| Scenario | Recommendation |
|----------|---------------|
| Heavy diagram rendering usage | ⚠️ Use legacy (or add to canonical) |
| Need session persistence | ⚠️ Use legacy (or add to canonical) |
| Plain-text WebSocket protocol | ⚠️ Use legacy (or add compatibility) |
| Minimal requirements | ⚠️ Either (canonical preferred) |

---

## Deprecation Timeline

**Current Status:** Legacy server maintained but frozen

**Phase 1: Deprecation Notice (v0.6.0 - Current)**
- Legacy server marked as deprecated
- Documentation updated
- Warning in server logs

**Phase 2: Feature Freeze (v0.7.0 - 2 months)**
- No new features in legacy server
- Security fixes only
- Canonical server recommended

**Phase 3: End of Life (v0.8.0 - 6 months)**
- Legacy server removed from repository
- All projects must migrate
- Documentation archived

**Phase 4: Removal (v0.9.0 - 9 months)**
- Legacy code deleted
- Migration tools removed
- All references removed

---

## Troubleshooting

### Issue: WebSocket Connection Fails

**Symptoms:**
- `WebSocket connection failed`
- `401 Unauthorized` on WebSocket upgrade

**Solutions:**
1. Check API key is set correctly
2. Verify CORS configuration
3. For canonical server: send auth message first
4. For legacy server: check token in query params

### Issue: Session Not Persisted

**Symptoms:**
- Each request creates new session
- Agent state not preserved

**Solutions:**
1. Canonical server: Use client-side state management
2. Or add session management to canonical server
3. Or use legacy server for session support

### Issue: Diagram Rendering Returns 404

**Symptoms:**
- `POST /render/plantuml` returns 404
- Diagrams not rendering

**Solutions:**
1. Add diagram rendering endpoints to canonical server
2. Or run legacy server on separate port
3. Or use external diagram service

### Issue: Response Format Mismatch

**Symptoms:**
- Client crashes parsing response
- Missing fields in response

**Solutions:**
1. Check OpenAPI docs: `http://localhost:8765/docs`
2. Update client to match new response format
3. Or add response transformation middleware

---

## Additional Resources

### Documentation

- **Canonical Server API Docs:** http://localhost:8765/docs (Swagger UI)
- **Canonical Server ReDoc:** http://localhost:8765/redoc
- **Legacy Server Code:** `/Users/vijaysingh/code/codingagent/web/server/main.py`
- **Canonical Server Code:** `/Users/vijaysingh/code/codingagent/victor/integrations/api/fastapi_server.py`

### Example Clients

**Python Client:**
```python
# examples/api_client.py

import requests
import websocket
import json

class VictorClient:
    def __init__(self, base_url="http://localhost:8765", api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, message):
        response = requests.post(
            f"{self.base_url}/chat",
            json={"messages": [{"role": "user", "content": message}]},
            headers=self.headers
        )
        return response.json()

    def stream_chat(self, message):
        response = requests.post(
            f"{self.base_url}/chat/stream",
            json={"messages": [{"role": "user", "content": message}]},
            headers=self.headers,
            stream=True
        )
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if data == "[DONE]":
                        break
                    yield data

# Usage
client = VictorClient(api_key="your-api-key")
result = client.chat("Hello, Victor!")
print(result["content"])

for chunk in client.stream_chat("Hello, Victor!"):
    print(chunk.get("content", ""), end="")
```

**JavaScript Client:**
```javascript
// examples/api_client.js

class VictorClient {
    constructor(baseUrl = 'http://localhost:8765', apiKey = null) {
        this.baseUrl = baseUrl;
        this.apiKey = apiKey;
    }

    async chat(message) {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: message }]
            })
        });
        return await response.json();
    }

    async *streamChat(message) {
        const response = await fetch(`${this.baseUrl}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey && { 'Authorization': `Bearer ${this.apiKey}` })
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: message }]
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data === '[DONE]') return;
                    yield data;
                }
            }
        }
    }
}

// Usage
const client = new VictorClient('your-api-key');
const result = await client.chat('Hello, Victor!');
console.log(result.content);

for await (const chunk of client.streamChat('Hello, Victor!')) {
    process.stdout.write(chunk.content || '');
}
```

### Testing

```bash
# Run API server tests
pytest tests/integrations/api/ -v

# Run with coverage
pytest tests/integrations/api/ --cov=victor.integrations.api --cov-report=html

# Run specific test
pytest tests/integrations/api/test_fastapi_server.py::test_health_check -v
```

---

## Summary

**Canonical Server Advantages:**
- 5x more features (80+ vs 15 endpoints)
- Modern architecture (class-based, extensible)
- Full OpenAPI documentation
- Production-ready (error handling, logging, lifecycle)
- Active development (new features first)

**Legacy Server Advantages:**
- Simpler (695 lines vs 3506 lines)
- Diagram rendering built-in
- Session management built-in
- Proven in production

**Migration Complexity:** Medium
- 4-5 weeks for big bang migration
- 8-12 weeks for gradual migration
- Main challenge: WebSocket protocol change
- Secondary challenge: Adding diagram rendering

**Recommendation:** Use canonical server for all new projects. Migrate existing projects using gradual approach with API gateway routing.

---

## Appendix: Code Location Reference

**Canonical Server:**
- Path: `/Users/vijaysingh/code/codingagent/victor/integrations/api/fastapi_server.py`
- Lines: 3,506
- Class: `VictorFastAPIServer`
- Entry Point: `create_fastapi_app()`

**Legacy Server:**
- Path: `/Users/vijaysingh/code/codingagent/web/server/main.py`
- Lines: 695
- Architecture: Functional (module-level handlers)
- Entry Point: `app` (FastAPI instance)

**Key Differences in Lines of Code:**
- Request/Response models: ~300 lines (canonical) vs 0 (legacy)
- Endpoint definitions: ~2,500 lines (canonical) vs ~400 lines (legacy)
- WebSocket handling: ~200 lines (canonical) vs ~150 lines (legacy)
- Lifecycle management: ~100 lines (canonical) vs ~50 lines (legacy)
- Background tasks: ~400 lines (canonical) vs ~100 lines (legacy)
