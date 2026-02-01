# Protocols Reference

This document provides a comprehensive reference for all protocols defined in Victor's architecture. Protocols enable loose coupling, testability, and flexibility through dependency inversion.

## Table of Contents

1. [Core Protocols](#core-protocols)
2. [Agent Lifecycle Protocols](#agent-lifecycle-protocols)
3. [Tool Protocols](#tool-protocols)
4. [Conversation Protocols](#conversation-protocols)
5. [Streaming Protocols](#streaming-protocols)
6. [Coordinator Protocols](#coordinator-protocols)
7. [Infrastructure Protocols](#infrastructure-protocols)
8. [Event Protocols](#event-protocols)

---

## Core Protocols

### IAgentFactory

**Purpose**: Unified agent creation for all agent types

**Module**: `victor.agent.protocols`

**Methods**:
```python
async def create_agent(
    self,
    mode: str,  # "foreground", "background", or "team_member"
    config: Optional[Any] = None,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create any agent type using shared infrastructure."""
```

**Usage**:
```python
factory = container.get(IAgentFactory)
agent = await factory.create_agent(mode="foreground", task="Implement feature X")
```

**Implementations**: `AgentFactory`

---

### IAgent

**Purpose**: Canonical agent protocol for all agent types

**Module**: `victor.agent.protocols`

**Properties**:
- `id: str` - Unique agent identifier
- `orchestrator: Any` - Agent orchestrator instance

**Methods**:
```python
async def execute(self, task: str, context: Dict[str, Any]) -> str:
    """Execute a task and return result."""
```

**Usage**:
```python
def process_agent(agent: IAgent, task: str):
    result = await agent.execute(task, {})
    return result
```

**Implementations**: `Agent`, `BackgroundAgent`, `TeamMember`, `SubAgent`

---

## Agent Lifecycle Protocols

### ProviderManagerProtocol

**Purpose**: Manage LLM provider lifecycle and switching

**Module**: `victor.agent.protocols`

**Properties**:
- `provider: Any` - Current provider instance
- `model: str` - Current model identifier
- `provider_name: str` - Provider name (anthropic, openai, etc.)
- `tool_adapter: Any` - Tool calling adapter
- `capabilities: Any` - Tool calling capabilities

**Methods**:
```python
def initialize_tool_adapter(self) -> None:
    """Initialize tool adapter for current provider/model."""

async def switch_provider(
    self,
    provider_name: str,
    model: Optional[str] = None,
) -> bool:
    """Switch to different provider/model."""
```

**Usage**:
```python
provider_mgr = container.get(ProviderManagerProtocol)
await provider_mgr.switch_provider("openai", "gpt-4")
```

---

### TaskTrackerProtocol

**Purpose**: Track task state and progress

**Module**: `victor.agent.protocols`

**Methods**:
```python
def start_task(self, task_id: str, description: str) -> None:
    """Start tracking a task."""

def update_task(self, task_id: str, status: str, **metadata: Any) -> None:
    """Update task status."""

def complete_task(self, task_id: str, result: Any) -> None:
    """Mark task as complete."""

def fail_task(self, task_id: str, error: Exception) -> None:
    """Mark task as failed."""

def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
    """Get task information."""
```

**Usage**:
```python
tracker = container.get(TaskTrackerProtocol)
tracker.start_task("task_1", "Implement feature X")
tracker.complete_task("task_1", {"files_modified": 5})
```

---

## Tool Protocols

### ToolRegistryProtocol

**Purpose**: Manage tool registration and lookup

**Module**: `victor.agent.protocols`

**Methods**:
```python
def register(self, tool: Any) -> None:
    """Register a tool."""

def get(self, name: str) -> Optional[Any]:
    """Get tool by name."""

def list_tools(self) -> List[str]:
    """List all registered tool names."""

def get_tool_cost(self, name: str) -> CostTier:
    """Get cost tier for tool."""

def register_before_hook(self, hook: Callable[..., Any]) -> None:
    """Register hook to run before tool execution."""
```

**Usage**:
```python
registry = container.get(ToolRegistryProtocol)
registry.register(my_tool)
tool = registry.get("read_file")
```

---

### ToolPipelineProtocol

**Purpose**: Coordinate tool execution with budget and caching

**Module**: `victor.agent.protocols`

**Properties**:
- `calls_used: int` - Number of tool calls used
- `budget: int` - Maximum tool calls allowed

**Methods**:
```python
async def execute(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
) -> ToolCallResult:
    """Execute a tool with budget enforcement."""

def is_budget_exhausted(self) -> bool:
    """Check if budget is exhausted."""
```

**Usage**:
```python
pipeline = container.get(ToolPipelineProtocol)
result = await pipeline.execute("read_file", {"path": "/path/to/file"})
if pipeline.is_budget_exhausted():
    print("Budget exhausted!")
```

---

### ToolExecutorProtocol

**Purpose**: Execute individual tools with validation

**Module**: `victor.agent.protocols`

**Methods**:
```python
def execute(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    context: Optional[Any] = None,
) -> Any:
    """Execute tool synchronously."""

async def aexecute(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
    context: Optional[Any] = None,
) -> Any:
    """Execute tool asynchronously."""

def validate_arguments(
    self,
    tool_name: str,
    arguments: Dict[str, Any],
) -> bool:
    """Validate tool arguments before execution."""
```

**Usage**:
```python
executor = container.get(ToolExecutorProtocol)
if executor.validate_arguments("read_file", {"path": "/file"}):
    result = await executor.aexecute("read_file", {"path": "/file"})
```

---

### IToolSelector

**Purpose**: Select tools based on query and context

**Module**: `victor.protocols.tool_selector`

**Methods**:
```python
async def select_tools(
    self,
    query: str,
    available_tools: List[Any],
    max_tools: int = 10,
    threshold: float = 0.3,
) -> List[Any]:
    """Select relevant tools for query."""

def compute_similarity(self, query: str, tool_description: str) -> float:
    """Compute similarity score between query and tool."""
```

**Usage**:
```python
selector = container.get(IToolSelector)
tools = await selector.select_tools(
    query="Read and analyze Python files",
    available_tools=registry.get_all_tools(),
    max_tools=5,
)
```

---

## Conversation Protocols

### ConversationControllerProtocol

**Purpose**: Manage conversation messages and context

**Module**: `victor.agent.protocols`

**Methods**:
```python
def add_message(self, role: str, content: str) -> None:
    """Add message to conversation."""

def get_messages(self) -> List[Dict[str, Any]]:
    """Get all messages."""

def get_context_metrics(self) -> Any:
    """Get context utilization metrics."""

def compact_if_needed(self) -> bool:
    """Compact conversation if context is full."""

def set_system_prompt(self, prompt: str) -> None:
    """Set system prompt."""
```

**Usage**:
```python
controller = container.get(ConversationControllerProtocol)
controller.add_message("user", "Help me debug this code")
if controller.compact_if_needed():
    print("Conversation compacted")
```

---

### ConversationStateMachineProtocol

**Purpose**: Track conversation stage (INITIAL, PLANNING, EXECUTING, etc.)

**Module**: `victor.agent.protocols`

**Methods**:
```python
def get_stage(self) -> ConversationStage:
    """Get current conversation stage."""

def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
    """Record tool execution for stage inference."""

def record_message(self, content: str, is_user: bool = True) -> None:
    """Record message for stage inference."""
```

**Usage**:
```python
state_machine = container.get(ConversationStateMachineProtocol)
state_machine.record_tool_execution("read_file", {"path": "/file"})
stage = state_machine.get_stage()  # Returns ConversationStage.READING
```

---

## Streaming Protocols

### StreamingControllerProtocol

**Purpose**: Manage streaming sessions and chunks

**Module**: `victor.agent.protocols`

**Methods**:
```python
async def handle_chunk(self, chunk: Any) -> None:
    """Handle a streaming chunk."""

async def end_session(self, session_id: str) -> None:
    """End a streaming session."""
```

---

### StreamingToolAdapterProtocol

**Purpose**: Stream tool execution results in real-time

**Module**: `victor.agent.protocols`

**Properties**:
- `calls_used: int` - Number of tool calls used

**Methods**:
```python
async def execute_streaming(
    self,
    tool_calls: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[StreamingToolChunk]:
    """Execute tools with streaming output."""

async def execute_streaming_single(
    self,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_call_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[StreamingToolChunk]:
    """Execute single tool with streaming output."""
```

**Usage**:
```python
adapter = container.get(StreamingToolAdapterProtocol)
async for chunk in adapter.execute_streaming_single("read_file", {"path": "/file"}):
    if chunk.chunk_type == "result":
        print(f"Result: {chunk.content}")
```

---

## Coordinator Protocols

### ToolCoordinatorProtocol

**Purpose**: Coordinate tool selection, budgeting, and execution

**Module**: `victor.agent.protocols`

**Methods**:
```python
async def select_and_execute(
    self,
    query: str,
    context: AgentToolSelectionContext,
) -> List[ToolCallResult]:
    """Select tools and execute within budget."""

async def select_tools(
    self,
    query: str,
    context: AgentToolSelectionContext,
) -> List[BaseTool]:
    """Select tools for execution."""

async def execute_tools(
    self,
    tools: List[BaseTool],
) -> List[ToolCallResult]:
    """Execute tools with caching."""
```

**Usage**:
```python
coordinator = container.get(ToolCoordinatorProtocol)
context = AgentToolSelectionContext(max_tools=5)
results = await coordinator.select_and_execute("Read Python files", context)
```

---

### StateCoordinatorProtocol

**Purpose**: Coordinate conversation state and stage transitions

**Module**: `victor.agent.protocols`

**Methods**:
```python
async def transition_to(
    self,
    new_stage: ConversationStage,
    reason: Optional[str] = None,
) -> None:
    """Transition to new stage."""

async def get_current_stage(self) -> ConversationStage:
    """Get current stage."""

async def get_stage_history(self) -> List[Dict[str, Any]]:
    """Get stage transition history."""
```

**Usage**:
```python
coordinator = container.get(StateCoordinatorProtocol)
await coordinator.transition_to(ConversationStage.EXECUTING, reason="Starting tool execution")
```

---

### PromptCoordinatorProtocol

**Purpose**: Coordinate system prompt assembly

**Module**: `victor.agent.protocols`

**Methods**:
```python
async def build_prompt(
    self,
    mode: AgentMode,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build system prompt for mode."""

async def add_prompt_section(
    self,
    section: str,
    content: str,
) -> None:
    """Add a prompt section."""
```

**Usage**:
```python
coordinator = container.get(PromptCoordinatorProtocol)
prompt = await coordinator.build_prompt(AgentMode.BUILD)
```

---

## Infrastructure Protocols

### ObservabilityProtocol

**Purpose**: Publish observability events

**Module**: `victor.agent.protocols`

**Methods**:
```python
def on_tool_start(
    self,
    tool_name: str,
    arguments: dict,
    tool_id: str,
) -> None:
    """Called when tool starts."""

def on_tool_end(
    self,
    tool_name: str,
    result: Any,
    success: bool,
    tool_id: str,
    error: Optional[str] = None,
) -> None:
    """Called when tool ends."""

def wire_state_machine(self, state_machine: Any) -> None:
    """Wire state machine for stage tracking."""

def on_error(self, error: Exception, context: dict) -> None:
    """Called on error."""
```

---

### ToolCacheProtocol

**Purpose**: Cache tool execution results

**Module**: `victor.agent.protocols`

**Methods**:
```python
async def get(self, key: str) -> Optional[Any]:
    """Get cached value."""

async def set(
    self,
    key: str,
    value: Any,
    ttl: Optional[int] = None,
) -> None:
    """Set cached value with TTL."""

async def invalidate(self, key: str) -> None:
    """Invalidate cache entry."""

async def clear(self) -> None:
    """Clear all cache."""
```

**Usage**:
```python
cache = container.get(ToolCacheProtocol)
result = await cache.get("tool_result")
if result is None:
    result = await execute_tool()
    await cache.set("tool_result", result, ttl=300)
```

---

### IBudgetManager

**Purpose**: Manage tool execution budget

**Module**: `victor.agent.protocols`

**Methods**:
```python
def can_execute(self, num_calls: int = 1) -> bool:
    """Check if budget allows execution."""

def record_call(self, cost: int = 1) -> None:
    """Record tool call against budget."""

def get_remaining_budget(self) -> int:
    """Get remaining budget."""

def reset_budget(self) -> None:
    """Reset budget to initial value."""
```

**Usage**:
```python
budget = container.get(IBudgetManager)
if budget.can_execute(5):
    await execute_tools(tools)
    budget.record_call(5)
```

---

### RecoveryHandlerProtocol

**Purpose**: Handle provider failures with adaptive recovery

**Module**: `victor.agent.protocols`

**Properties**:
- `enabled: bool` - Whether recovery is enabled
- `consecutive_failures: int` - Number of consecutive failures

**Methods**:
```python
def detect_failure(
    self,
    error: Exception,
    context: Dict[str, Any],
) -> Optional[RecoveryStrategy]:
    """Detect failure and suggest recovery strategy."""

async def recover(
    self,
    error: Exception,
    context: Dict[str, Any],
) -> RecoveryOutcome:
    """Execute recovery strategy."""

def record_outcome(
    self,
    success: bool,
    quality_improvement: float = 0.0,
) -> None:
    """Record recovery outcome for learning."""
```

---

## Event Protocols

### IEventBackend

**Purpose**: Complete event backend implementation

**Module**: `victor.core.events.protocols`

**Properties**:
- `backend_type: BackendType` - Backend type (IN_MEMORY, KAFKA, etc.)
- `is_connected: bool` - Connection status

**Methods**:
```python
async def connect(self) -> None:
    """Connect to backend service."""

async def disconnect(self) -> None:
    """Disconnect from backend."""

async def publish(self, event: MessagingEvent) -> bool:
    """Publish event."""

async def subscribe(
    self,
    pattern: str,
    handler: EventHandler,
) -> SubscriptionHandle:
    """Subscribe to events matching pattern."""

async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
    """Unsubscribe from pattern."""

async def health_check(self) -> bool:
    """Check backend health."""
```

**Usage**:
```python
backend = create_event_backend(BackendConfig.for_observability())
await backend.connect()

await backend.subscribe("tool.*", my_handler)
await backend.publish(MessagingEvent(topic="tool.start", data={}))

await backend.disconnect()
```

---

### IEventPublisher

**Purpose**: Event publishing interface

**Module**: `victor.core.events.protocols`

**Methods**:
```python
async def publish(self, event: MessagingEvent) -> bool:
    """Publish single event."""

async def publish_batch(self, events: List[MessagingEvent]) -> int:
    """Publish multiple events."""
```

---

### IEventSubscriber

**Purpose**: Event subscription interface

**Module**: `victor.core.events.protocols`

**Methods**:
```python
async def subscribe(
    self,
    pattern: str,
    handler: EventHandler,
) -> SubscriptionHandle:
    """Subscribe to events matching pattern."""

async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
    """Unsubscribe from pattern."""
```

---

## Data Structures

### MessagingEvent

**Purpose**: Canonical event format

**Module**: `victor.core.events.protocols`

**Fields**:
```python
topic: str  # Dot-separated topic (e.g., "tool.start")
data: Dict[str, Any]  # Event payload
id: str  # Unique event ID
timestamp: float  # Unix timestamp
source: str  # Component that generated event
correlation_id: Optional[str]  # For correlating related events
partition_key: Optional[str]  # For partitioning
headers: Dict[str, str]  # Metadata headers
delivery_guarantee: DeliveryGuarantee  # AT_MOST_ONCE, AT_LEAST_ONCE, EXACTLY_ONCE
```

**Usage**:
```python
event = MessagingEvent(
    topic="tool.complete",
    data={"tool": "read_file", "result": "file content"},
    source="agent_1",
    correlation_id="task_abc123",
)
await event_bus.publish(event)
```

---

### AgentToolSelectionContext

**Purpose**: Context for tool selection

**Module**: `victor.agent.protocols`

**Fields**:
```python
stage: Optional[str]  # Conversation stage
task_type: str  # Task type (analysis, action, creation)
recent_tools: List[str]  # Recent tool executions
turn_number: int  # Conversation turn number
is_continuation: bool  # Is this a continuation
max_tools: int  # Maximum tools to select
planned_tools: Optional[List[str]]  # Pre-selected tools
metadata: Dict[str, Any]  # Additional metadata
```

---

### StreamingToolChunk

**Purpose**: Streaming tool execution chunk

**Module**: `victor.agent.protocols`

**Fields**:
```python
tool_name: str  # Tool being executed
tool_call_id: str  # Unique call ID
chunk_type: str  # "start", "progress", "result", "error", "cache_hit"
content: Any  # Chunk payload
is_final: bool  # Is this the final chunk
metadata: Dict[str, Any]  # Additional context
```

---

## Protocol Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Lifecycle** | Agent creation and management | IAgentFactory, IAgent |
| **Provider** | LLM provider management | ProviderManagerProtocol |
| **Tool** | Tool execution and selection | ToolPipelineProtocol, IToolSelector |
| **Conversation** | Message and state management | ConversationControllerProtocol, ConversationStateMachineProtocol |
| **Streaming** | Real-time output | StreamingControllerProtocol, StreamingToolAdapterProtocol |
| **Coordinator** | Complex operations | ToolCoordinatorProtocol, StateCoordinatorProtocol |
| **Infrastructure** | Caching, budget, observability | ToolCacheProtocol, IBudgetManager, ObservabilityProtocol |
| **Event** | Event-driven communication | IEventBackend, IEventPublisher, IEventSubscriber |

## Implementation Guidelines

### Creating Protocol Implementations

1. **Implement all protocol methods**: Must provide concrete implementations
2. **Follow protocol contracts**: Respect documented behavior
3. **Handle errors gracefully**: Implement proper error handling
4. **Document custom behavior**: Use docstrings for implementation-specific details

```python
# Example implementation
class MyToolSelector:
    """Custom tool selector implementation."""

    async def select_tools(
        self,
        query: str,
        available_tools: List[Any],
        max_tools: int = 10,
        threshold: float = 0.3,
    ) -> List[Any]:
        """Select tools using custom logic."""
        # Implementation
        return selected_tools

    def compute_similarity(self, query: str, tool_description: str) -> float:
        """Compute similarity using custom algorithm."""
        # Implementation
        return score
```

### Testing With Protocols

Use protocol-based mocks for testing:

```python
from unittest.mock import Mock

def test_my_component():
    # Create mock
    mock_tool_executor = Mock(spec=ToolExecutorProtocol)
    mock_tool_executor.aexecute.return_value = {"result": "success"}

    # Inject mock
    component = MyComponent(tool_executor=mock_tool_executor)

    # Test
    result = await component.do_work()

    # Verify
    mock_tool_executor.aexecute.assert_called_once()
```

## Additional Resources

- [REFACTORING_OVERVIEW.md](./REFACTORING_OVERVIEW.md) - Architecture overview
- [BEST_PRACTICES.md](./BEST_PRACTICES.md) - Usage patterns
- [MIGRATION_GUIDES.md](./MIGRATION_GUIDES.md) - Migration examples

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
