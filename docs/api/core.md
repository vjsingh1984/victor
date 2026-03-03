# Core APIs Reference

Core abstractions for events, state management, workflows, and multi-agent teams.

## Overview

The Core API provides:
- **Event system** for observability and monitoring
- **State management** with multiple scopes
- **Multi-agent teams** with different formations
- **Protocols** for extensibility
- **Error handling** with contextual errors

## Table of Contents

- [Events](#events)
- [State Management](#state-management)
- [Multi-Agent Teams](#multi-agent-teams)
- [Error Handling](#error-handling)

## Events

Victor provides an event-driven architecture for observability and monitoring.

### Quick Example

```python
from victor.core.events import get_observability_bus, LifecycleEvent

# Get event bus
bus = get_observability_bus()

# Emit lifecycle event
bus.emit_lifecycle_event(
    "agent_started",
    {"agent_id": "my-agent", "task": "Hello"}
)

# Subscribe to events
async for event in bus.subscribe():
    print(f"Event: {event.event_type}")
```

### Event Types

```python
# Lifecycle events
agent_started        # Agent started execution
agent_completed      # Agent completed successfully
agent_error          # Agent encountered an error

# Tool events
tool_call_started    # Tool execution started
tool_call_completed  # Tool execution completed
tool_call_failed     # Tool execution failed

# Workflow events
workflow_started     # Workflow execution started
workflow_completed   # Workflow execution completed
node_started         # Workflow node started
node_completed       # Workflow node completed

# Team events
team_started         # Team execution started
team_completed       # Team execution completed
member_started       # Team member started
member_completed     # Team member completed
```

### EventBus Methods

```python
class ObservabilityEventBus:
    """Event bus for observability events."""

    async def emit_lifecycle_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """Emit a lifecycle event."""

    async def emit_metric_event(
        self,
        metric_name: str,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a metric event."""

    def subscribe(
        self,
        event_filter: str | Callable | None = None,
    ) -> AsyncIterator[Event]:
        """Subscribe to events."""

    def get_metrics(
        self,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        """Get metrics for events."""
```

## State Management

Victor provides unified state management across multiple scopes.

### State Scopes

```python
from victor.state import StateScope, get_global_manager

# Available scopes
StateScope.GLOBAL        # Application-wide state
StateScope.WORKFLOW      # Workflow execution state
StateScope.CONVERSATION  # Conversation state
StateScope.TEAM         # Multi-agent team state
```

### Global State Manager

```python
from victor.state import get_global_manager, StateScope

# Get global state manager
state = get_global_manager()

# Set state with scope
await state.set("task_id", "task-123", scope=StateScope.WORKFLOW)

# Get state (searches all scopes)
task_id = await state.get("task_id")

# Get from specific scope
task_id = await state.get("task_id", scope=StateScope.WORKFLOW)

# Delete state
await state.delete("task_id", scope=StateScope.WORKFLOW)

# List all keys in scope
keys = await state.list_scope(StateScope.WORKFLOW)
```

### Workflow State Manager

```python
from victor.state import WorkflowStateManager

# Create workflow state manager
mgr = WorkflowStateManager()

# Set workflow state
await mgr.set("task_id", "task-123")
await mgr.set("status", "running")

# Get workflow state
task_id = await mgr.get("task_id")

# Delete workflow state
await mgr.delete("status")
```

## Multi-Agent Teams

Victor provides multi-agent team coordination with different formation patterns.

### Team Formation Types

```python
from victor.teams import TeamFormation

# Available formations
TeamFormation.SEQUENTIAL    # Execute members one after another
TeamFormation.PARALLEL      # Execute all members simultaneously
TeamFormation.HIERARCHICAL  # Manager delegates to workers
TeamFormation.PIPELINE      # Output of one feeds into next
TeamFormation.CONSENSUS     # All members must agree
```

### Creating Teams

```python
from victor.teams import (
    create_coordinator,
    TeamFormation,
    AgentMessage,
    MessageType,
)

# Create coordinator
coordinator = create_coordinator()

# Add members
coordinator.add_member("researcher", researcher_agent)
coordinator.add_member("coder", coder_agent)
coordinator.add_member("tester", tester_agent)

# Set formation
coordinator.set_formation(TeamFormation.SEQUENTIAL)

# Execute task
result = await coordinator.execute_task(
    "Implement a REST API endpoint",
    context={"requirements": "..."}
)

print(result.final_answer)
for member_id, member_result in result.member_results.items():
    print(f"{member_id}: {member_result.response}")
```

### Team Message Types

```python
from victor.teams import MessageType

# Available message types
MessageType.TASK         # Task assignment
MessageType.RESULT       # Result from task
MessageType.QUERY        # Information request
MessageType.FEEDBACK     # Feedback on work
MessageType.DELEGATION   # Task delegation
MessageType.DISCOVERY    # Discovery/finding
MessageType.REQUEST      # Generic request
MessageType.RESPONSE     # Response to request
MessageType.STATUS       # Status update
MessageType.ALERT       # Alert/warning
MessageType.HANDOFF      # Task handoff
```

### Agent Message

```python
from victor.teams import AgentMessage, MessageType, MessagePriority

# Create message
message = AgentMessage(
    sender_id="coordinator",
    content="Please analyze this code",
    message_type=MessageType.TASK,
    recipient_id="researcher",
    priority=MessagePriority.HIGH,
    data={"code": "..."}
)

# Broadcast message (no recipient)
broadcast = AgentMessage(
    sender_id="coordinator",
    content="Task completed",
    message_type=MessageType.STATUS,
    recipient_id=None,  # Broadcast
)

# Reply to message
reply = AgentMessage(
    sender_id="researcher",
    content="Analysis complete",
    message_type=MessageType.RESULT,
    recipient_id="coordinator",
    reply_to=message.id
)
```

### Team Result

```python
from victor.teams import TeamResult, MemberResult

# Team result contains:
result: TeamResult

result.final_answer      # Final answer from team
result.success          # Whether execution succeeded
result.member_results    # Dict of member_id -> MemberResult
result.metadata         # Additional metadata

# Member result contains:
member_result: MemberResult

member_result.response   # Response from member
member_result.success    # Whether member succeeded
member_result.error      # Error if failed
member_result.metadata   # Additional metadata
```

## Error Handling

Victor provides contextual error classes for better error handling.

### Error Classes

```python
from victor.core.errors import (
    VictorError,
    ProviderError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ToolExecutionError,
    ConfigurationError,
    ValidationError,
)
```

### Error Hierarchy

```
VictorError
├── ProviderError
│   ├── ProviderAuthError
│   ├── ProviderConnectionError
│   ├── ProviderTimeoutError
│   ├── ProviderRateLimitError
│   └── ProviderInvalidResponseError
├── ToolExecutionError
├── ConfigurationError
└── ValidationError
```

### Handling Errors

```python
from victor.core.errors import (
    ProviderConnectionError,
    ToolExecutionError,
    ConfigurationError,
)

try:
    result = await agent.run("Execute task")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Field: {e.field}")
    print(f"Suggestion: {e.suggestion}")
except ProviderConnectionError as e:
    print(f"Provider error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
except ToolExecutionError as e:
    print(f"Tool error: {e.message}")
    print(f"Tool: {e.tool_name}")
    print(f"Suggestion: {e.suggestion}")
```

### Contextual Error Attributes

| Error | Attribute | Type | Description |
|-------|-----------|------|-------------|
| `ConfigurationError` | `field` | `str \| None` | Configuration field that caused error |
| `ConfigurationError` | `suggestion` | `str \| None` | Suggested fix |
| `ToolExecutionError` | `tool_name` | `str \| None` | Tool that failed |
| `ToolExecutionError` | `suggestion` | `str \| None` | Suggested fix |
| `ProviderError` | `suggestion` | `str \| None` | Suggested fix |

## Protocols

Victor uses protocols for extensibility and type safety.

### ITeamCoordinator

Base protocol for team coordinators.

```python
class ITeamCoordinator(Protocol):
    """Base team coordinator protocol."""

    def add_member(
        self,
        member_id: str,
        member: ITeamMember,
    ) -> ITeamCoordinator:
        """Add a member to the team."""
        ...

    def set_formation(
        self,
        formation: TeamFormation,
    ) -> None:
        """Set team formation pattern."""
        ...

    async def execute_task(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> TeamResult:
        """Execute task with team."""
        ...
```

### ITeamMember

Protocol for team members.

```python
class ITeamMember(Protocol):
    """Team member protocol."""

    member_id: str
    role: str

    async def process(
        self,
        task: str,
        context: dict[str, Any],
    ) -> MemberResult:
        """Process task as team member."""
        ...
```

### IObservableCoordinator

Protocol for observability support.

```python
class IObservableCoordinator(Protocol):
    """Observability protocol for coordinators."""

    async def get_metrics(
        self,
    ) -> dict[str, Any]:
        """Get coordinator metrics."""
        ...

    async def get_event_history(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent events."""
        ...
```

## Best Practices

### 1. Use Events for Observability

```python
# Good - Emit events for monitoring
bus = get_observability_bus()
await bus.emit_lifecycle_event(
    "custom_event",
    {"data": "value"}
)

# Use events for debugging
async for event in bus.subscribe(event_filter="agent_*"):
    print(f"Event: {event}")
```

### 2. Use Appropriate State Scopes

```python
# Good - Use appropriate scopes
state = get_global_manager()

# Workflow state (isolated to workflow)
await state.set("workflow_id", "wf-123", scope=StateScope.WORKFLOW)

# Conversation state (isolated to conversation)
await state.set("user_id", "user-456", scope=StateScope.CONVERSATION)

# Global state (shared across all)
await state.set("config", {...}, scope=StateScope.GLOBAL)
```

### 3. Choose Right Team Formation

```python
# Sequential - Context chaining
coordinator.set_formation(TeamFormation.SEQUENTIAL)
# Researcher → Coder → Tester (context passed)

# Parallel - Independent work
coordinator.set_formation(TeamFormation.PARALLEL)
# All agents work independently, results combined

# Hierarchical - Manager/Worker
coordinator.set_formation(TeamFormation.HIERARCHICAL)
# Manager delegates to workers, synthesizes results

# Pipeline - Output chaining
coordinator.set_formation(TeamFormation.PIPELINE)
# Output of one becomes input of next

# Consensus - Agreement building
coordinator.set_formation(TeamFormation.CONSENSUS)
# All members discuss until agreement
```

### 4. Handle Contextual Errors

```python
# Good - Handle specific error types
try:
    result = await agent.run(task)
except ConfigurationError as e:
    # Handle configuration issues
    logger.error(f"Config error in {e.field}: {e.suggestion}")
except ProviderConnectionError as e:
    # Handle provider issues
    logger.warning(f"Provider unavailable: {e.suggestion}")
except ToolExecutionError as e:
    # Handle tool failures
    logger.warning(f"Tool {e.tool_name} failed: {e.suggestion}")
```

### 5. Use Message Priorities

```python
# Good - Prioritize important messages
from victor.teams import MessagePriority

# Urgent alert
urgent_message = AgentMessage(
    sender_id="monitor",
    content="Critical error detected",
    message_type=MessageType.ALERT,
    priority=MessagePriority.URGENT,
)

# Low priority info
info_message = AgentMessage(
    sender_id="logger",
    content="Task completed",
    message_type=MessageType.STATUS,
    priority=MessagePriority.LOW,
)
```

## See Also

- [Agent API](agent.md) - Agent usage with teams
- [StateGraph API](graph.md) - Workflow state management
- [Provider API](providers.md) - Provider error handling
