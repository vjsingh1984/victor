# API Migration Guide: Victor 0.5.x to 0.5.0

This guide provides detailed information about API changes in Victor 0.5.0, with before/after examples for all major changes.

## Table of Contents

1. [Overview](#overview)
2. [Core API Changes](#core-api-changes)
3. [Provider API Changes](#provider-api-changes)
4. [Tool API Changes](#tool-api-changes)
5. [Orchestrator API Changes](#orchestrator-api-changes)
6. [Workflow API Changes](#workflow-api-changes)
7. [Event API Changes](#event-api-changes)
8. [Protocol Changes](#protocol-changes)
9. [Type Signature Changes](#type-signature-changes)
10. [Code Examples](#code-examples)

---

## Overview

### API Change Categories

1. **Critical Changes** - Breaking changes that require code updates
2. **Moderate Changes** - Changes to function signatures or behavior
3. **Minor Changes** - Import path changes or deprecations
4. **New APIs** - New functionality available in 0.5.0

### API Versioning Policy

- **Major version (0.5.0)**: Breaking changes allowed
- **Minor version (1.x.0)**: New features, backward compatible
- **Patch version (1.0.x)**: Bug fixes only

---

## Core API Changes

### 1. Bootstrap API

#### Orchestrator Creation

**Before (0.5.x)**:
```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.openai_provider import OpenAIProvider
from victor.tools.registry import SharedToolRegistry
from victor.observability.event_bus import EventBus

# Manual dependency creation
provider = OpenAIProvider(api_key="sk-...")
tool_registry = SharedToolRegistry.get_instance()
event_bus = EventBus()

orchestrator = AgentOrchestrator(
    provider=provider,
    tool_registry=tool_registry,
    observability=event_bus,
    max_tokens=4096,
    temperature=0.7,
)
```

**After (0.5.0)**:
```python
from victor.core.bootstrap import bootstrap_orchestrator
from victor.config.settings import Settings

# Environment-based configuration
settings = Settings(
    provider="openai",
    model="gpt-4",
    max_tokens=4096,
    temperature=0.7,
)

orchestrator = bootstrap_orchestrator(settings)

# Or with custom provider
from victor.providers.openai_provider import OpenAIProvider

provider = OpenAIProvider()
orchestrator = bootstrap_orchestrator(settings, provider=provider)
```

**Changes**:
- Manual dependency creation replaced with `bootstrap_orchestrator()`
- DI container manages all dependencies
- API keys must come from environment variables or `Settings`
- Simplified API with fewer parameters

### 2. Settings API

**Before (0.5.x)**:
```python
from victor.config import Config

config = Config()
config.max_tokens = 4096
config.temperature = 0.7
config.tool_budget = 100
```

**After (0.5.0)**:
```python
from victor.config.settings import Settings

settings = Settings(
    max_tokens=4096,
    temperature=0.7,
    tool_budget=100,
)

# Or from environment variables
# VICTOR_MAX_TOKENS=4096
# VICTOR_TEMPERATURE=0.7
# VICTOR_TOOL_BUDGET=100

settings = Settings()
```

**Changes**:
- `Config` renamed to `Settings`
- Pydantic v2 validation
- Environment variable support via `pydantic-settings`
- Better type hints

### 3. DI Container API

**New in 0.5.0**:
```python
from victor.core.container import ServiceContainer, ServiceLifetime
from victor.protocols import ToolRegistryProtocol, ObservabilityProtocol

# Create container
container = ServiceContainer()

# Register service
container.register(
    ToolRegistryProtocol,
    lambda c: ToolRegistry(),
    ServiceLifetime.SINGLETON,
)

# Resolve service
registry = container.get(ToolRegistryProtocol)

# Scoped service lifetime
with container.create_scope() as scope:
    scoped_service = scope.get(ConversationStateMachineProtocol)
```

**Changes**:
- New DI container for dependency management
- Service lifetime management (singleton, scoped, transient)
- Thread-safe service resolution
- Automatic cleanup of scoped services

---

## Provider API Changes

### 1. Provider Initialization

**Before (0.5.x)**:
```python
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.anthropic_provider import AnthropicProvider

# API key passed to constructor
openai = OpenAIProvider(api_key="sk-...")
anthropic = AnthropicProvider(api_key="sk-ant-...")
```

**After (0.5.0)**:
```python
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.anthropic_provider import AnthropicProvider

# API keys from environment variables
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

openai = OpenAIProvider()
anthropic = AnthropicProvider()

# Or via Settings
from victor.config.settings import Settings

settings = Settings(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-..."
)

openai = OpenAIProvider(settings=settings)
anthropic = AnthropicProvider(settings=settings)
```

**Changes**:
- `api_key` parameter removed from constructor
- Keys must come from environment variables or `Settings`
- Centralized API key management
- Better security (no keys in code)

### 2. Provider Switching

**Before (0.5.x)**:
```python
# Manual provider replacement
orchestrator.provider = new_provider
```

**After (0.5.0)**:
```python
from victor.agent.coordinators import ProviderCoordinator

coordinator = container.get(ProviderCoordinator)

# Switch provider
await coordinator.switch_provider(
    provider_name="anthropic",
    model="claude-sonnet-4-5"
)

# Or create new orchestrator with different provider
new_orchestrator = bootstrap_orchestrator(
    settings,
    provider=AnthropicProvider()
)
```

**Changes**:
- Provider switching via `ProviderCoordinator`
- Clean provider lifecycle management
- Context preservation during switch
- Validation of provider compatibility

### 3. Provider Context Limits

**New in 0.5.0**:
```python
from victor.providers.base import BaseProvider

provider = OpenAIProvider()

# Get context limits
limits = provider.get_context_limits()
print(f"Max tokens: {limits.max_tokens}")
print(f"Max output: {limits.max_output_tokens}")

# Respect provider limits
from victor.agent.coordinators import ContextCoordinator

context_coordinator = container.get(ContextCoordinator)
truncated_content = await context_coordinator.truncate_for_context(
    content=large_content,
    provider=provider,
    tool_name="read_file"
)
```

**Changes**:
- Provider context limits now enforced
- `provider_context_limits.yaml` configuration
- Automatic content truncation
- Provider-specific limit handling

---

## Tool API Changes

### 1. Tool Registry Access

**Before (0.5.x)**:
```python
from victor.tools.registry import SharedToolRegistry

# Singleton access
registry = SharedToolRegistry.get_instance()

# Get all tools
tools = registry.get_tools()

# Get specific tool
tool = registry.get_tool("read_file")
```

**After (0.5.0)**:
```python
from victor.core.container import ServiceContainer
from victor.protocols import ToolRegistryProtocol

# DI container access
container = ServiceContainer()
registry = container.get(ToolRegistryProtocol)

# Get all tools
tools = await registry.get_tools()

# Get specific tool
tool = await registry.get_tool("read_file")

# List tools by category
tools = await registry.list_tools(category="coding")
```

**Changes**:
- Singleton replaced with DI container
- Methods are now async
- Better categorization support
- Tool metadata available

### 2. Tool Execution

**Before (0.5.x)**:
```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(...)

# Execute tool
result = orchestrator.execute_tool(
    tool_name="read_file",
    arguments={"path": "/path/to/file.txt"}
)
```

**After (0.5.0)**:
```python
from victor.agent.coordinators import ToolCoordinator
from victor.protocols import ToolCoordinatorProtocol

container = ServiceContainer()
coordinator = container.get(ToolCoordinatorProtocol)

# Execute tool
result = await coordinator.execute_tool(
    tool=tool_registry.get_tool("read_file"),
    arguments={"path": "/path/to/file.txt"}
)

# Execute multiple tools with budget
results = await coordinator.select_and_execute(
    query="Read and analyze Python files",
    context=AgentToolSelectionContext(max_tools=5)
)
```

**Changes**:
- Tool execution via `ToolCoordinator`
- Budget-aware execution
- Async execution
- Better error handling

### 3. Tool Selection

**Before (0.5.x)**:
```python
from victor.agent.tool_selector import ToolSelector

selector = ToolSelector(strategy="keyword")

tools = selector.select_tools(
    query="Read Python files",
    available_tools=all_tools
)
```

**After (0.5.0)**:
```python
from victor.protocols import IToolSelector
from victor.core.container import ServiceContainer

container = ServiceContainer()
selector = container.get(IToolSelector)

# Tool selection is now async
tools = await selector.select_tools(
    query="Read Python files",
    context=AgentToolSelectionContext(
        max_tools=10,
        conversation_stage=Stage.ANALYZING
    )
)

# With caching
from victor.config.settings import Settings

settings = Settings(
    tool_selection_strategy="hybrid",
    tool_cache_enabled=True,
    tool_cache_size=500
)
```

**Changes**:
- Protocol-based dependency injection
- Async selection
- Context-aware selection
- Caching support
- Multiple strategies (keyword, semantic, hybrid)

---

## Orchestrator API Changes

### 1. Chat API

**Before (0.5.x)**:
```python
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(...)

# Synchronous chat
result = orchestrator.chat(
    message="Hello, world!",
    temperature=0.7,
    max_tokens=1000
)
```

**After (0.5.0)**:
```python
from victor.core.bootstrap import bootstrap_orchestrator

orchestrator = bootstrap_orchestrator(settings)

# Async chat (required)
result = await orchestrator.chat(
    message="Hello, world!",
    temperature=0.7,
    max_tokens=1000
)

# Streaming chat
async for chunk in orchestrator.stream_chat(
    message="Hello, world!",
):
    print(chunk.content, end="")
```

**Changes**:
- `chat()` is now async (use `await`)
- Streaming API improved
- Better error handling
- Context management

### 2. Conversation State

**Before (0.5.x)**:
```python
from victor.agent.state_machine import ConversationStateMachine

sm = ConversationStateMachine()

# Transition state
sm.transition_to("READING")

# Get current state
state = sm.get_state()
print(state)  # "READING"

# Get history
history = sm.get_history()
```

**After (0.5.0)**:
```python
from victor.protocols import IConversationStateMachine
from victor.core.container import ServiceContainer
from victor.framework import Stage

container = ServiceContainer()
sm = container.get(IConversationStateMachine)

# Async transition
await sm.transition_to(Stage.READING)

# Get current state
state = await sm.get_state()
print(state)  # Stage.READING

# Get history
history = await sm.get_history()

# Check if can transition
can_transition = await sm.can_transition_to(Stage.EXECUTING)
```

**Changes**:
- All methods are async
- `Stage` enum instead of strings
- Protocol-based interface
- Better validation

### 3. Memory API

**New in 0.5.0**:
```python
from victor.agent.coordinators import MemoryCoordinator

memory = container.get(MemoryCoordinator)

# Store memory
await memory.store(
    content="User prefers TypeScript",
    memory_type="preference",
    metadata={"category": "language"}
)

# Retrieve memory
memories = await memory.retrieve(
    query="What language does the user prefer?",
    limit=5
)

# Search memories
results = await memory.search(
    filters={"memory_type": "preference"},
    limit=10
)

# Summarize memories
summary = await memory.summarize(
    time_window="7d"
)
```

**Changes**:
- New memory capability
- Semantic search
- Persistent storage
- Memory summarization

---

## Workflow API Changes

### 1. Workflow Compilation

**Before (0.5.x)**:
```python
from victor.workflows.executor import WorkflowExecutor

executor = WorkflowExecutor(orchestrator)

# Execute workflow
result = await executor.execute(
    workflow=workflow_definition,
    context=execution_context
)
```

**After (0.5.0)**:
```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

compiler = UnifiedWorkflowCompiler(orchestrator)

# Compile workflow
compiled = compiler.compile(workflow_definition)

# Execute workflow
result = await compiled.invoke(execution_context)

# With checkpointing
result = await compiled.invoke(
    execution_context,
    thread_id="conversation-123",
    checkpoint_ns="workflow-1"
)
```

**Changes**:
- `WorkflowExecutor` replaced with `UnifiedWorkflowCompiler`
- Two-level caching (definition + execution)
- Checkpointing support
- Better error recovery

### 2. Workflow Definition

**Before (0.5.x)**:
```yaml
# old_workflow.yaml
workflow:
  name: my_workflow
  nodes:
    - id: step1
      type: agent
      role: researcher
      goal: "Research topic"
```

**After (0.5.0)**:
```yaml
# new_workflow.yaml
workflows:
  my_workflow:
    nodes:
      - id: step1
        type: agent
        role: researcher
        goal: "Research topic"
        next: [step2]
        tool_budget: 10

      - id: step2
        type: agent
        role: writer
        goal: "Write summary"
        next: [END]

      - id: step3
        type: condition
        condition: "quality_check"
        branches:
          "pass": step4
          "fail": step1
```

**Changes**:
- `workflows` root key required
- `next` field required for all nodes
- `END` sentinel for terminal nodes
- Better node type validation
- Conditional edges support

### 3. YAML Workflow Provider

**New in 0.5.0**:
```python
from victor.framework.workflows import BaseYAMLWorkflowProvider

class MyWorkflowProvider(BaseYAMLWorkflowProvider):
    """YAML-based workflow provider."""

    def __init__(self):
        super().__init__(
            vertical_name="my_vertical",
            yaml_path="victor/config/workflows/my_workflows.yaml"
        )

# Use provider
provider = MyWorkflowProvider()
compiler = provider.compile_workflow("my_workflow")
result = await compiler.invoke(context)
```

**Changes**:
- YAML-first workflow definition
- Base class for workflow providers
- Automatic validation
- Hot-reloading support

---

## Event API Changes

### 1. Event Backend

**Before (0.5.x)**:
```python
from victor.observability.event_bus import EventBus

bus = EventBus()

# Publish event
bus.publish("tool.complete", data={"tool": "read_file"})

# Subscribe to events
bus.subscribe("tool.*", handler)

# Event handler
def handler(event):
    print(f"Event: {event.topic}, Data: {event.data}")
```

**After (0.5.0)**:
```python
from victor.core.events import create_event_backend, MessagingEvent
from victor.config.events import BackendConfig

# Create backend
backend = create_event_backend(BackendConfig.for_observability())
await backend.connect()

# Publish event
await backend.publish(
    MessagingEvent(
        topic="tool.complete",
        data={"tool": "read_file", "result": "..."},
        metadata={"timestamp": "..."}
    )
)

# Subscribe to events
handle = await backend.subscribe("tool.*", my_handler)

# Event handler (must be async)
async def my_handler(event: MessagingEvent):
    print(f"Event: {event.topic}, Data: {event.data}")

# Unsubscribe
await backend.unsubscribe(handle)
```

**Changes**:
- `EventBus` replaced with pluggable backends
- `MessagingEvent` wrapper for events
- Async publish/subscribe
- Multiple backend types (in-memory, Kafka, SQS, etc.)
- Better error handling

### 2. Event Backend Types

**New in 0.5.0**:
```python
from victor.core.events import BackendType, BackendConfig

# In-Memory (default)
config = BackendConfig(
    backend_type=BackendType.IN_MEMORY
)

# Kafka
config = BackendConfig(
    backend_type=BackendType.KAFKA,
    kafka_brokers=["localhost:9092"],
    kafka_topic="victor-events"
)

# AWS SQS
config = BackendConfig(
    backend_type=BackendType.SQS,
    sqs_queue_url="https://sqs.amazonaws.com/..."
)

# RabbitMQ
config = BackendConfig(
    backend_type=BackendType.RABBITMQ,
    rabbitmq_url="amqp://localhost:5672"
)

# Redis
config = BackendConfig(
    backend_type=BackendType.REDIS,
    redis_url="redis://localhost:6379"
)

backend = create_event_backend(config)
await backend.connect()
```

**Changes**:
- Multiple backend types
- Configuration-based backend creation
- Backend-specific settings
- Connection management

---

## Protocol Changes

### 1. Tool Coordinator Protocol

**New in 0.5.0**:
```python
from victor.protocols import ToolCoordinatorProtocol
from typing import Protocol

@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """Protocol for tool coordination operations."""

    async def select_and_execute(
        self,
        query: str,
        context: AgentToolSelectionContext,
    ) -> List[ToolCallResult]:
        """Select tools and execute within budget."""
        ...

    async def execute_tool(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
    ) -> ToolCallResult:
        """Execute a single tool."""
        ...

    async def get_tool_budget(self) -> ToolBudget:
        """Get current tool budget."""
        ...
```

**Usage**:
```python
from victor.agent.coordinators import ToolCoordinator

# ToolCoordinator implements ToolCoordinatorProtocol
coordinator: ToolCoordinatorProtocol = ToolCoordinator(...)
await coordinator.select_and_execute(query, context)
```

### 2. New Protocols

**New in 0.5.0**:

```python
# Provider coordination
from victor.protocols import ProviderCoordinatorProtocol

# State management
from victor.protocols import IConversationStateMachine

# Memory operations
from victor.protocols import IMemoryCoordinator

# Planning operations
from victor.protocols import IPlanningCoordinator

# Analytics
from victor.protocols import IAnalyticsCoordinator
```

**Changes**:
- 98 protocols defined
- Protocol-based dependencies throughout
- Easy mocking for tests
- Type-safe interfaces

---

## Type Signature Changes

### 1. Tool Execution Result

**Before (0.5.x)**:
```python
def execute_tool(tool_name: str, arguments: Dict) -> Any:
    """Execute tool and return result."""
    ...
```

**After (0.5.0)**:
```python
async def execute_tool(
    tool: BaseTool,
    arguments: Dict[str, Any],
) -> ToolCallResult:
    """Execute tool and return structured result."""
    ...
```

**Changes**:
- Async function
- `BaseTool` instead of tool name
- `ToolCallResult` instead of `Any`
- Type-safe arguments

### 2. Chat Result

**Before (0.5.x)**:
```python
def chat(
    self,
    message: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """Send chat message and get response."""
    ...
```

**After (0.5.0)**:
```python
async def chat(
    self,
    message: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
) -> ChatResponse:
    """Send chat message and get response."""
    ...
```

**Changes**:
- Async function
- `ChatResponse` object instead of `str`
- Optional parameters (use settings defaults)
- Streaming support

---

## Code Examples

### Example 1: Simple Chat Bot

**Before (0.5.x)**:
```python
from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.openai_provider import OpenAIProvider

orchestrator = AgentOrchestrator(
    provider=OpenAIProvider(api_key="sk-...")
)

while True:
    user_input = input("You: ")
    response = orchestrator.chat(user_input)
    print(f"Bot: {response}")
```

**After (0.5.0)**:
```python
import asyncio
from victor.core.bootstrap import bootstrap_orchestrator
from victor.config.settings import Settings

async def main():
    settings = Settings(provider="openai", model="gpt-4")
    orchestrator = bootstrap_orchestrator(settings)

    while True:
        user_input = input("You: ")
        response = await orchestrator.chat(user_input)
        print(f"Bot: {response.content}")

asyncio.run(main())
```

### Example 2: Custom Tool

**Before (0.5.x)**:
```python
from victor.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"

    def execute(self, **kwargs):
        return "result"

# Register tool
from victor.tools.registry import SharedToolRegistry
registry = SharedToolRegistry.get_instance()
registry.register(MyTool())
```

**After (0.5.0)**:
```python
from victor.tools.base import BaseTool, ToolMetadata
from victor.framework.capabilities import register_capability

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string"}
        }
    }
    cost_tier = CostTier.LOW

    async def execute(self, **kwargs):
        return "result"

# Register via capability
register_capability(MyTool())

# Or via DI container
from victor.core.container import ServiceContainer
from victor.protocols import ToolRegistryProtocol

container = ServiceContainer()
registry = container.get(ToolRegistryProtocol)
await registry.register_tool(MyTool())
```

### Example 3: Workflow Execution

**Before (0.5.x)**:
```python
from victor.workflows.executor import WorkflowExecutor
from victor.agent.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(...)
executor = WorkflowExecutor(orchestrator)

workflow = {
    "name": "my_workflow",
    "nodes": [...]
}

result = await executor.execute(workflow, context)
```

**After (0.5.0)**:
```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.core.bootstrap import bootstrap_orchestrator

orchestrator = bootstrap_orchestrator(settings)
compiler = UnifiedWorkflowCompiler(orchestrator)

workflow = {
    "workflows": {
        "my_workflow": {
            "nodes": [...]
        }
    }
}

compiled = compiler.compile(workflow)
result = await compiled.invoke(context)

# With streaming
async for event in compiled.stream(context):
    print(f"Event: {event}")
```

---

## Migration Checklist

Use this checklist to ensure you've migrated all API usage:

- [ ] Replace `AgentOrchestrator()` with `bootstrap_orchestrator()`
- [ ] Update provider initialization (remove `api_key` parameter)
- [ ] Replace `SharedToolRegistry.get_instance()` with DI container
- [ ] Make tool execution async
- [ ] Update chat calls to use `await`
- [ ] Replace `EventBus` with `create_event_backend()`
- [ ] Update workflow execution to use `UnifiedWorkflowCompiler`
- [ ] Replace string-based states with `Stage` enum
- [ ] Update type hints to use protocols
- [ ] Add error handling for new exceptions

---

## Additional Resources

- [Main Migration Guide](./MIGRATION_GUIDE.md)
- [Protocol Reference](../architecture/PROTOCOLS_REFERENCE.md)
- [API Reference](../API_REFERENCE.md)
- [Best Practices](../architecture/BEST_PRACTICES.md)

---

**Last Updated**: 2025-01-21
**Version**: 0.5.0
