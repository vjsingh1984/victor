# Coordinator Recipes - Part 3

**Part 3 of 4:** Recipes 4-6 (Context, Middleware, Multi-Tenant)

---

## Navigation

- [Part 1: Recipes 1-2](part-1-recipes-1-2.md)
- [Part 2: Recipe 3](part-2-recipe-3.md)
- **[Part 3: Recipes 4-6](#)** (Current)
- [Part 4: Recipes 7-9](part-4-recipes-7-9.md)
- [**Complete Guide**](../coordinator_recipes.md)

---
## Recipe 5: Add Custom Middleware Integration

### Problem Statement

You want to add custom middleware logic (e.g., logging, authentication, rate limiting) to all orchestrator operations.

### Solution Overview

Create a middleware coordinator that wraps all orchestrator calls.

### Step-by-Step Instructions

#### Step 1: Create Middleware Coordinator

```python
# middleware_coordinator.py
from typing import Callable, Any
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

class MiddlewareCoordinator:
    """Apply middleware to orchestrator operations."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.middlewares = []

    def add_middleware(self, middleware: Callable):
        """Add middleware function."""
        self.middlewares.append(middleware)

    def apply_middleware(self, func: Callable) -> Callable:
        """Apply all middleware to a function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute middleware in order
            for middleware in self.middlewares:
                result = await middleware(self.orchestrator, func, *args, **kwargs)
                if result is not None:  # Middleware can return early
                    return result

            # Execute original function
            return await func(*args, **kwargs)

        return wrapper

# Middleware functions
async def logging_middleware(orchestrator, func, *args, **kwargs):
    """Log all orchestrator calls."""
    logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
    start = time.time()

    try:
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} completed in {duration:.2f}s")
        return result
    except Exception as e:
        duration = time.time() - start
        logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
        raise

async def rate_limit_middleware(orchestrator, func, *args, **kwargs):
    """Apply rate limiting to orchestrator calls."""
    # Implement rate limiting logic here
    # For example, using Redis or in-memory rate limiter
    return None  # Return None to continue to next middleware

async def authentication_middleware(orchestrator, func, *args, **kwargs):
    """Validate authentication tokens."""
    # Extract session_id from kwargs
    session_id = kwargs.get('session_id')

    if not session_id:
        raise AuthenticationError("No session_id provided")

    # Validate session
    is_valid = await orchestrator._session_coordinator.validate_session(session_id)

    if not is_valid:
        raise AuthenticationError(f"Invalid session: {session_id}")

    return None  # Continue to next middleware
```

#### Step 2: Apply Middleware

```python
# main.py
from victor.agent.orchestrator import AgentOrchestrator

# Create orchestrator
orchestrator = AgentOrchestrator(settings=settings, provider=provider, model=model)

# Create middleware coordinator
middleware_coordinator = MiddlewareCoordinator(orchestrator)

# Add middleware
middleware_coordinator.add_middleware(logging_middleware)
middleware_coordinator.add_middleware(authentication_middleware)
middleware_coordinator.add_middleware(rate_limit_middleware)

# Apply middleware to chat function
orchestrator.chat = middleware_coordinator.apply_middleware(orchestrator.chat)

# Use orchestrator (middleware is automatically applied)
response = await orchestrator.chat("Hello!", session_id="valid-session")
```

### Production Considerations

- **Middleware order**: Order matters (e.g., logging before auth for audit trail)
- **Performance**: Each middleware adds overhead
- **Error handling**: Ensure middleware handles errors gracefully
- **Testing**: Test middleware in isolation
- **Configuration**: Make middleware configurable via settings

---

## Recipe 6: Multi-Tenant Configuration

### Problem Statement

You want to serve multiple tenants with different configurations, prompts, and analytics.

### Solution Overview

Create tenant-specific coordinators that load configuration and behavior per tenant.

### Step-by-Step Instructions

#### Step 1: Create Tenant Config Provider

```python
# tenant_providers.py
from victor.protocols import IConfigProvider

class TenantConfigProvider(IConfigProvider):
    """Load configuration per tenant."""

    def __init__(self, db_connection):
        self.db = db_connection

    def priority(self) -> int:
        return 100  # High priority

    async def get_config(self, session_id: str) -> dict:
        """Load tenant config from database."""
        # Extract tenant_id from session_id
        tenant_id = session_id.split(':')[0]  # Format: "tenant:session"

        query = """
            SELECT config FROM tenant_configs
            WHERE tenant_id = $1 AND active = true
        """

        result = await self.db.fetchrow(query, tenant_id)

        if result:
            return result['config']
        return {}
```

#### Step 2: Create Tenant Prompt Contributor

```python
# tenant_contributors.py
from victor.agent.coordinators.prompt_coordinator import BasePromptContributor

class TenantPromptContributor(BasePromptContributor):
    """Add tenant-specific prompts."""

    def __init__(self, db_connection):
        self.db = db_connection

    def priority(self) -> int:
        return 80

    async def get_contribution(self, context: PromptContext) -> str:
        """Load tenant prompt template."""
        tenant_id = context.get('tenant_id')

        if not tenant_id:
            return ""

        query = """
            SELECT prompt_template FROM tenant_prompt_templates
            WHERE tenant_id = $1
        """

        result = await self.db.fetchrow(query, tenant_id)

        if result:
            return f"\n{result['prompt_template']}"
        return ""
```

#### Step 3: Create Multi-Tenant Orchestrator

```python
# multi_tenant_orchestrator.py
from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.coordinators import ConfigCoordinator, PromptCoordinator

class MultiTenantOrchestrator:
    """Manage orchestrators per tenant."""

    def __init__(self, db_connection, base_settings, base_provider):
        self.db = db_connection
        self.base_settings = base_settings
        self.base_provider = base_provider
        self.orchestrators = {}  # tenant_id -> orchestrator

    async def get_orchestrator(self, tenant_id: str) -> AgentOrchestrator:
        """Get or create orchestrator for tenant."""
        if tenant_id not in self.orchestrators:
            # Create tenant-specific coordinators
            config_coordinator = ConfigCoordinator(providers=[
                TenantConfigProvider(self.db),
            ])

            prompt_coordinator = PromptCoordinator(contributors=[
                TenantPromptContributor(self.db),
            ])

            # Create orchestrator
            orchestrator = AgentOrchestrator(
                settings=self.base_settings,
                provider=self.base_provider,
                model="claude-sonnet-4-5",
                _config_coordinator=config_coordinator,
                _prompt_coordinator=prompt_coordinator,
            )

            self.orchestrators[tenant_id] = orchestrator

        return self.orchestrators[tenant_id]

# Usage
multi_tenant = MultiTenantOrchestrator(db, settings, provider)

tenant_orchestrator = await multi_tenant.get_orchestrator("tenant-abc")
response = await tenant_orchestrator.chat("Hello!", session_id="tenant-abc:session-123")
```

### Production Considerations

- **Isolation**: Ensure tenant data is isolated at database level
- **Resource limits**: Implement per-tenant resource limits
- **Caching**: Cache tenant configurations
- **Fallbacks**: Provide default configuration for tenants without custom config
- **Monitoring**: Monitor per-tenant usage and performance

---

## Recipe 7: Real-Time Analytics Dashboard

### Problem Statement

You want to display real-time analytics (tokens, tool usage, costs) in a dashboard.

### Solution Overview

Create a WebSocket-based analytics exporter that pushes events to connected clients.

### Step-by-Step Instructions

#### Step 1: Create WebSocket Analytics Exporter

```python
# websocket_analytics.py
from typing import Set, List
from victor.protocols import IAnalyticsExporter, ExportResult, AnalyticsEvent
import websockets
import json

class WebSocketAnalyticsExporter(BaseAnalyticsExporter):
    """Export analytics to WebSocket clients."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def export(self, events: List[AnalyticsEvent]) -> ExportResult:
        """Broadcast events to all connected clients."""
        if not self.clients or not events:
            return ExportResult(success=True, exported_count=0)

        message = json.dumps([e.model_dump() for e in events])

        # Broadcast to all clients
        for client in list(self.clients):  # Copy to avoid modification during iteration
            try:
                await client.send(message)
            except Exception as e:
                print(f"Error sending to client: {e}")
                self.clients.discard(client)

        return ExportResult(success=True, exported_count=len(events))

    async def start_server(self):
        """Start WebSocket server."""
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port
        ):
            print(f"WebSocket analytics server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    async def _handle_client(self, websocket, path):
        """Handle new WebSocket client connection."""
        print(f"New client connected: {websocket.remote_address}")
        self.clients.add(websocket)

        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")
```

#### Step 2: Create Dashboard Client

```html
<!-- dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Victor Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Real-Time Analytics</h1>
    <canvas id="tokenChart"></canvas>
    <canvas id="toolChart"></canvas>

    <script>
        const ws = new WebSocket('ws://localhost:8765');

        const tokenData = { labels: [], datasets: [{ label: 'Tokens', data: [] }] };
        const toolData = { labels: [], datasets: [{ label: 'Tool Calls', data: [] }] };

        const tokenChart = new Chart(document.getElementById('tokenChart'), {
            type: 'line',
            data: tokenData,
        });

        const toolChart = new Chart(document.getElementById('toolChart'), {
            type: 'bar',
            data: toolData,
        });

        ws.onmessage = (event) => {
            const analyticsEvents = JSON.parse(event.data);

            analyticsEvents.forEach(e => {
                if (e.type === 'token_usage') {
                    // Update token chart
                    tokenData.labels.push(new Date(e.timestamp).toLocaleTimeString());
                    tokenData.datasets[0].data.push(e.data.total_tokens);
                    tokenChart.update();
                } else if (e.type === 'tool_call') {
                    // Update tool chart
                    const tool = e.data.tool;
                    const existingIndex = toolData.labels.indexOf(tool);

                    if (existingIndex >= 0) {
                        toolData.datasets[0].data[existingIndex]++;
                    } else {
                        toolData.labels.push(tool);
                        toolData.datasets[0].data.push(1);
                    }
                    toolChart.update();
                }
            });
        };
    </script>
</body>
</html>
```

#### Step 3: Integrate with Orchestrator

```python
# main.py
import asyncio
from victor.agent.coordinators import AnalyticsCoordinator
from victor.agent.orchestrator import AgentOrchestrator

# Create WebSocket exporter
ws_exporter = WebSocketAnalyticsExporter(host="0.0.0.0", port=8765)

# Create analytics coordinator
analytics_coordinator = AnalyticsCoordinator(exporters=[
    ws_exporter,
])

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=Settings(enable_analytics=True),
    provider=provider,
    model=model,
    _analytics_coordinator=analytics_coordinator,
)

# Start WebSocket server in background
async def run_server():
    await ws_exporter.start_server()

asyncio.create_task(run_server())

# Use orchestrator (analytics are broadcast to dashboard)
await orchestrator.chat("Hello!")
```

### Production Considerations

- **Authentication**: Add authentication to WebSocket connections
- **Rate limiting**: Limit client message frequency
- **Scalability**: Use Redis Pub/Sub for multiple dashboard instances
- **Data retention**: Aggregate data to reduce chart data points
- **Error handling**: Handle client disconnections gracefully

---

