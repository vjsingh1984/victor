# Protocol Reference - Part 3

**Part 3 of 4:** Streaming, Observability, and Recovery Protocols

---

## Navigation

- [Part 1: Factory, Provider, Tool](part-1-factory-provider-tool.md)
- [Part 2: Coordinator & Conversation](part-2-coordinator-conversation.md)
- **[Part 3: Streaming, Observability, Recovery](#)** (Current)
- [Part 4: Utility, Access, Budget](part-4-utility-access-budget.md)
- [**Complete Reference**](../PROTOCOL_REFERENCE.md)

---
## Conversation Protocols

### ConversationControllerProtocol

Conversation management.

```python
@runtime_checkable
class ConversationControllerProtocol(Protocol):
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        ...

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in conversation."""
        ...

    def get_context_metrics(self) -> Any:
        """Get current context utilization metrics."""
        ...

    def compact_if_needed(self) -> bool:
        """Compact conversation if context is nearly full.

        Returns:
            True if compaction occurred
        """
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        ...
```

---

### ConversationStateMachineProtocol

Conversation stage tracking.

```python
@runtime_checkable
class ConversationStateMachineProtocol(Protocol):
    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage."""
        ...

    def get_current_stage(self) -> "ConversationStage":
        """Get current conversation stage (alias)."""
        ...

    def record_tool_execution(
        self,
        tool_name: str,
        args: Dict[str, Any]
    ) -> None:
        """Record tool execution for stage inference."""
        ...

    def record_message(
        self,
        content: str,
        is_user: bool = True
    ) -> None:
        """Record a message for stage inference."""
        ...
```

---

### MessageHistoryProtocol

Message storage.

```python
@runtime_checkable
class MessageHistoryProtocol(Protocol):
    def add_message(
        self,
        role: str,
        content: str,
        **kwargs: Any
    ) -> Any:
        """Add a message."""
        ...

    def get_messages_for_provider(self) -> List[Any]:
        """Get all messages for provider."""
        ...

    def clear(self) -> None:
        """Clear message history."""
        ...
```

---

## Streaming Protocols

### StreamingToolAdapterProtocol

Unified streaming interface for tool execution.

```python
@runtime_checkable
class StreamingToolAdapterProtocol(Protocol):
    async def execute_streaming(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["StreamingToolChunk"]:
        """Execute tools with streaming output.

        Yields StreamingToolChunk for each execution phase:
        1. "start" - Tool execution beginning
        2. "cache_hit" - Result served from cache (skips execution)
        3. "progress" - Intermediate progress updates
        4. "result" - Successful completion with result
        5. "error" - Execution failure

        Args:
            tool_calls: List of tool calls to execute
            context: Optional execution context

        Yields:
            StreamingToolChunk for each execution event
        """
        ...

    async def execute_streaming_single(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["StreamingToolChunk"]:
        """Execute a single tool with streaming output.

        Convenience method for single tool execution.

        Args:
            tool_name: Name of tool to execute
            tool_args: Tool arguments
            tool_call_id: Optional identifier for tracking
            context: Optional execution context

        Yields:
            StreamingToolChunk for each execution event
        """
        ...

    @property
    def calls_used(self) -> int:
        """Number of tool calls used (delegates to ToolPipeline)."""
        ...

    @property
    def calls_remaining(self) -> int:
        """Number of tool calls remaining in budget."""
        ...

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        ...
```

---

### StreamingControllerProtocol

Streaming session management.

```python
@runtime_checkable
class StreamingControllerProtocol(Protocol):
    def start_session(self, session_id: str) -> Any:
        """Start a new streaming session."""
        ...

    def end_session(self, session_id: str) -> None:
        """End a streaming session."""
        ...

    def get_active_session(self) -> Optional[Any]:
        """Get the currently active session."""
        ...
```

---

## Observability Protocols

### ObservabilityProtocol

Event emission for monitoring and tracing.

```python
@runtime_checkable
class ObservabilityProtocol(Protocol):
    def on_tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_id: str
    ) -> None:
        """Called when tool execution starts."""
        ...

    def on_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool,
        tool_id: str,
        error: Optional[str] = None,
    ) -> None:
        """Called when tool execution ends."""
        ...

    def wire_state_machine(self, state_machine: Any) -> None:
        """Wire state machine for automatic state change events."""
        ...

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Called when an error occurs."""
        ...
```

---

### MetricsCollectorProtocol

Metrics collection.

```python
@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    def on_tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        iteration: int
    ) -> None:
        """Record tool start metrics."""
        ...

    def on_tool_complete(self, result: Any) -> None:
        """Record tool completion metrics."""
        ...

    def on_streaming_session_complete(self, session: Any) -> None:
        """Record session completion metrics."""
        ...
```

---

## Recovery Protocols
