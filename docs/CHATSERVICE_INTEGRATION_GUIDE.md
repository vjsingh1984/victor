# ChatService Integration Guide

**Target Audience**: Victor framework developers  
**Purpose**: Guide for integrating ChatService in custom code  
**Status**: Current (Phase 3 Architecture)

## Overview

ChatService is the canonical interface for all chat operations in Victor. This guide shows how to properly integrate ChatService in your code.

## Quick Start

### 1. Understanding ChatService

ChatService is a protocol that provides:
- `chat()` - Non-streaming chat completion
- `stream_chat()` - Streaming chat completion  
- Protocol-based dependency injection
- Unified error handling and logging

### 2. When to Use ChatService

**Use ChatService when**:
- Building background agents
- Creating API endpoints
- Implementing custom workflows
- Writing service layer code
- Creating framework extensions

**Don't use ChatService when**:
- Working with UI layer (use VictorClient instead)
- Writing simple scripts (use Agent.chat())
- Testing (use mocks)

---

## Integration Patterns

### Pattern 1: Constructor Injection

**Best for**: Service layer components

```python
from victor.agent.services.protocols import ChatServiceProtocol

class MyCustomService:
    """Custom service that uses ChatService."""
    
    def __init__(
        self,
        chat_service: ChatServiceProtocol,  # ✅ Protocol-based injection
        config: MyServiceConfig,
    ):
        self._chat_service = chat_service
        self._config = config
    
    async def process_task(self, task: str) -> str:
        """Process a task using ChatService."""
        response = await self._chat_service.chat(task)
        return response.content
```

### Pattern 2: Runtime Resolution

**Best for**: API layer, framework code

```python
from victor.runtime.chat_runtime import resolve_chat_service, resolve_chat_runtime

async def handle_api_request(orchestrator, message: str):
    """Handle API request using ChatService."""
    
    # Resolve chat service from orchestrator
    chat_service = resolve_chat_service(orchestrator)
    if chat_service is None:
        chat_service = resolve_chat_runtime(orchestrator)
    
    # Use chat service
    response = await chat_service.chat(message)
    return response
```

### Pattern 3: Background Agents

**Best for**: Background agent management

```python
from victor.agent.background_agent import BackgroundAgentManager
from victor.runtime.chat_runtime import resolve_chat_service

async def start_background_agent(orchestrator, task: str):
    """Start a background agent with ChatService."""
    
    # Resolve ChatService
    chat_service = resolve_chat_service(orchestrator)
    
    # Create manager with ChatService injection
    manager = BackgroundAgentManager(
        orchestrator=orchestrator,
        chat_service=chat_service,  # ✅ Inject ChatService
        max_concurrent=4,
    )
    
    # Start agent
    agent_id = await manager.start_agent(task=task, mode="build")
    return agent_id
```

### Pattern 4: API Endpoints

**Best for**: HTTP/WebSocket endpoints

```python
from fastapi import WebSocket
from victor.runtime.chat_runtime import resolve_chat_runtime

async def websocket_chat_endpoint(websocket: WebSocket, orchestrator):
    """WebSocket endpoint using ChatService."""
    
    # Resolve chat runtime
    chat_runtime = resolve_chat_runtime(orchestrator)
    
    # Stream responses
    async for chunk in chat_runtime.stream_chat(message):
        await websocket.send_json(chunk)
```

---

## Code Examples

### Example 1: Custom Workflow Node

```python
from victor.agent.services.protocols import ChatServiceProtocol

class CustomWorkflowNode:
    """Custom workflow node that uses ChatService."""
    
    def __init__(self, chat_service: ChatServiceProtocol):
        self._chat_service = chat_service
    
    async def execute(self, state: dict) -> dict:
        """Execute workflow node."""
        
        # Use ChatService for reasoning
        response = await self._chat_service.chat(
            f"Analyze this state: {state}"
        )
        
        # Update state with response
        state["analysis"] = response.content
        return state
```

### Example 2: Parallel Agent Execution

```python
from victor.agent.services.protocols import ChatServiceProtocol

class ParallelAgentExecutor:
    """Execute multiple agents in parallel."""
    
    def __init__(self, chat_service: ChatServiceProtocol):
        self._chat_service = chat_service
    
    async def execute_parallel(self, tasks: list[str]) -> list[str]:
        """Execute multiple tasks in parallel."""
        
        async def process_task(task: str) -> str:
            response = await self._chat_service.chat(task)
            return response.content
        
        # Run in parallel
        results = await asyncio.gather(*[
            process_task(task) for task in tasks
        ])
        
        return results
```

### Example 3: Streaming Response Handler

```python
from victor.agent.services.protocols import ChatServiceProtocol

class StreamingResponseHandler:
    """Handle streaming responses."""
    
    def __init__(self, chat_service: ChatServiceProtocol):
        self._chat_service = chat_service
    
    async def stream_with_processing(self, message: str):
        """Stream and process chunks."""
        
        async for chunk in self._chat_service.stream_chat(message):
            # Process each chunk
            if chunk.get("type") == "content":
                yield chunk["content"]
            elif chunk.get("type") == "tool_call":
                yield f"Tool: {chunk['tool_call']['name']}"
```

---

## Testing with ChatService

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""
    mock = AsyncMock()
    
    # Mock chat method
    mock.chat = MagicMock(return_value=AsyncMock(
        content="Test response",
        tool_calls=[]
    ))
    
    # Mock stream_chat method
    async def mock_stream():
        yield {"type": "content", "content": "Hello"}
        yield {"type": "content", "content": " World"}
    
    mock.stream_chat = MagicMock(return_value=mock_stream())
    return mock

def test_my_service(mock_chat_service):
    """Test MyService with mock ChatService."""
    service = MyService(
        chat_service=mock_chat_service,
        config=MyServiceConfig()
    )
    
    # Test chat method
    response = await service.process_task("test")
    assert response == "Test response"
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_chat_service_integration(orchestrator):
    """Test ChatService integration."""
    from victor.runtime.chat_runtime import resolve_chat_runtime
    
    # Resolve real ChatService
    chat_service = resolve_chat_runtime(orchestrator)
    
    # Test chat method
    response = await chat_service.chat("Say hello")
    assert response.content
    assert len(response.content) > 0
```

---

## Best Practices

### 1. Always Use Protocol Types

```python
# ✅ GOOD - Protocol type
from victor.agent.services.protocols import ChatServiceProtocol

def __init__(self, chat_service: ChatServiceProtocol):
    self._chat_service = chat_service

# ❌ BAD - Concrete type
from victor.agent.services.chat_service import ChatService

def __init__(self, chat_service: ChatService):
    self._chat_service = chat_service
```

### 2. Handle Optional ChatService

```python
# ✅ GOOD - Handle None gracefully
class MyService:
    def __init__(self, chat_service: Optional[ChatServiceProtocol] = None):
        self._chat_service = chat_service
    
    async def process(self, task: str):
        if self._chat_service:
            response = await self._chat_service.chat(task)
        else:
            # Fallback behavior
            response = "ChatService not available"
        return response
```

### 3. Use AsyncContext Managers for Streaming

```python
# ✅ GOOD - Proper async handling
async def stream_response(self, message: str):
    async for chunk in self._chat_service.stream_chat(message):
        try:
            processed = self._process_chunk(chunk)
            yield processed
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise
```

### 4. Error Handling

```python
# ✅ GOOD - Proper error handling
async def safe_chat(self, message: str) -> Optional[str]:
    """Chat with error handling."""
    try:
        response = await self._chat_service.chat(message)
        return response.content
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return None
```

---

## Common Patterns

### Pattern 1: Retry Logic

```python
class ResilientChatService:
    """Chat service with retry logic."""
    
    def __init__(self, chat_service: ChatServiceProtocol, max_retries: int = 3):
        self._chat_service = chat_service
        self._max_retries = max_retries
    
    async def chat_with_retry(self, message: str) -> str:
        """Chat with automatic retry."""
        for attempt in range(self._max_retries):
            try:
                response = await self._chat_service.chat(message)
                return response.content
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Pattern 2: Caching

```python
class CachedChatService:
    """Chat service with response caching."""
    
    def __init__(self, chat_service: ChatServiceProtocol, ttl: int = 300):
        self._chat_service = chat_service
        self._cache = {}
        self._ttl = ttl
    
    async def chat_cached(self, message: str) -> str:
        """Chat with caching."""
        cache_key = hashlib.md5(message.encode()).hexdigest()
        
        # Check cache
        if cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._ttl:
                return cached_result
        
        # Call ChatService
        response = await self._chat_service.chat(message)
        
        # Cache result
        self._cache[cache_key] = (response.content, time.time())
        return response.content
```

### Pattern 3: Middleware Chain

```python
class ChatServiceMiddleware:
    """Middleware for ChatService calls."""
    
    def __init__(self, chat_service: ChatServiceProtocol):
        self._chat_service = chat_service
        self._middleware = []
    
    def add_middleware(self, func):
        """Add middleware function."""
        self._middleware.append(func)
    
    async def chat(self, message: str) -> str:
        """Chat with middleware chain."""
        # Apply pre-processing middleware
        for mw in reversed(self._middleware):
            message = await mw.before_chat(message)
        
        # Call ChatService
        response = await self._chat_service.chat(message)
        
        # Apply post-processing middleware
        for mw in self._middleware:
            response = await mw.after_chat(response)
        
        return response
```

---

## Troubleshooting

### Issue: ChatService is None

**Problem**: `resolve_chat_service()` returns None

**Solution**:
```python
# Try fallback to chat_runtime
chat_service = resolve_chat_service(orchestrator)
if chat_service is None:
    chat_service = resolve_chat_runtime(orchestrator)
```

### Issue: AsyncMock Not Working

**Problem**: Tests fail with AsyncMock

**Solution**:
```python
# Create proper async mock
@pytest.fixture
def mock_chat_service():
    mock = AsyncMock()
    
    # Mock async generator
    async def mock_stream():
        yield {"type": "content", "content": "test"}
    
    mock.stream_chat = MagicMock(return_value=mock_stream())
    return mock
```

### Issue: Protocol Not Recognized

**Problem**: `isinstance(service, ChatServiceProtocol)` returns False

**Solution**:
```python
# Use @runtime_checkable decorator
from typing import Protocol, runtime_checkable

@runtime_checkable
class ChatServiceProtocol(Protocol):
    async def chat(self, message: str) -> CompletionResponse:
        ...
```

---

## Performance Tips

### 1. Avoid Unnecessary Calls

```python
# ✅ GOOD - Batch requests
responses = await self._chat_service.chat_batch([
    "Task 1", "Task 2", "Task 3"
])

# ❌ BAD - Individual calls
for task in tasks:
    response = await self._chat_service.chat(task)
```

### 2. Use Streaming for Long Responses

```python
# ✅ GOOD - Stream long responses
async for chunk in self._chat_service.stream_chat(long_prompt):
    yield chunk

# ❌ BAD - Wait for complete response
response = await self._chat_service.chat(long_prompt)
```

### 3. Cache When Appropriate

```python
# ✅ GOOD - Cache idempotent calls
@lru_cache(maxsize=128)
async def get_cached_response(self, message: str) -> str:
    return await self._chat_service.chat(message)
```

---

## References

- Service Protocol: `victor/agent/services/protocols/chat_service.py`
- Runtime Resolution: `victor/runtime/chat_runtime.py`
- Background Agents: `victor/agent/background_agent.py`
- Migration Guide: `docs/SERVICE_LAYER_API_MIGRATION_COMPLETE.md`

---

*Generated: 2026-05-01*  
*Status: Current (Phase 3 Architecture)*
