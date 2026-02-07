# Protocol Reference - Part 1

**Part 1 of 4:** Factory, Provider, and Tool Protocols

---

## Navigation

- **[Part 1: Factory, Provider, Tool](#)** (Current)
- [Part 2: Coordinator & Conversation](part-2-coordinator-conversation.md)
- [Part 3: Streaming, Observability, Recovery](part-3-streaming-observability-recovery.md)
- [Part 4: Utility, Access, Budget](part-4-utility-access-budget.md)
- [**Complete Reference**](../PROTOCOL_REFERENCE.md)

---
# Victor AI 0.5.0 Protocol Reference

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.


Complete reference for all protocol interfaces in Victor AI.

**Table of Contents**
- [Overview](#overview)
- [Factory Protocols](#factory-protocols)
- [Provider Protocols](#provider-protocols)
- [Tool Protocols](#tool-protocols)
- [Coordinator Protocols](#coordinator-protocols)
- [Conversation Protocols](#conversation-protocols)
- [Streaming Protocols](#streaming-protocols)
- [Observability Protocols](#observability-protocols)
- [Recovery Protocols](#recovery-protocols)
- [Utility Protocols](#utility-protocols)

---

## Overview

Protocols in Victor AI define interfaces for dependency injection, testing, and SOLID compliance. All protocols are runtime-checkable and can be used for type hints and mock validation.

### Importing Protocols

```python
from victor.agent.protocols import (
    ProviderManagerProtocol,
    ToolRegistryProtocol,
    ConversationControllerProtocol,
    ToolExecutorProtocol,
)
```

### Using Protocols

```python
def process_with_provider(provider: ProviderManagerProtocol) -> None:
    """Type-safe function using protocol."""
    model = provider.model
    print(f"Using model: {model}")

# Mock in tests
from unittest.mock import MagicMock

mock_provider = MagicMock(spec=ProviderManagerProtocol)
```

---

## Factory Protocols

### IAgentFactory

Unified agent creation protocol.

**Purpose:** Provides single interface for creating all agent types (foreground, background, team_member) following Single Responsibility Principle.

```python
@runtime_checkable
class IAgentFactory(Protocol):
    async def create_agent(
        self,
        mode: str,
        config: Optional[Any] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create any agent type using shared infrastructure.

        Args:
            mode: Agent creation mode - "foreground", "background", or "team_member"
            config: Optional unified agent configuration (UnifiedAgentConfig)
            task: Optional task description (for background agents)
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance (Agent, BackgroundAgent, or TeamMember/SubAgent)
        """
        ...
```

**Implementation Example:**

```python
from victor.agent.factory import OrchestratorAgentFactory

factory = OrchestratorAgentFactory(orchestrator)

# Foreground agent
agent = await factory.create_agent(mode="foreground")

# Background agent with task
agent = await factory.create_agent(
    mode="background",
    task="Implement feature X"
)

# Team member
agent = await factory.create_agent(
    mode="team_member",
    role="researcher"
)
```

---

### IAgent

Canonical agent protocol for all agent types.

**Purpose:** Ensures Liskov Substitution Principle compliance across all agent implementations.

```python
@runtime_checkable
class IAgent(Protocol):
    @property
    def id(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    def orchestrator(self) -> Any:
        """Agent orchestrator instance."""
        ...

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Execution result
        """
        ...
```

**Compliance:** All agent types (Agent, BackgroundAgent, TeamMember, SubAgent) must implement this protocol.

---

## Provider Protocols

### ProviderManagerProtocol

Provider lifecycle and switching management.

```python
@runtime_checkable
class ProviderManagerProtocol(Protocol):
    @property
    def provider(self) -> Any:
        """Get the current provider instance."""
        ...

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        ...

    @property
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'anthropic', 'openai')."""
        ...

    @property
    def tool_adapter(self) -> Any:
        """Get the tool calling adapter for current provider."""
        ...

    @property
    def capabilities(self) -> Any:
        """Get tool calling capabilities for current model."""
        ...

    def initialize_tool_adapter(self) -> None:
        """Initialize the tool adapter for current provider/model."""
        ...

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None
    ) -> bool:
        """Switch to a different provider/model.

        Args:
            provider_name: Name of provider to switch to
            model: Optional model to use

        Returns:
            True if switch successful
        """
        ...
```

**Usage:**

```python
def configure_provider(manager: ProviderManagerProtocol) -> None:
    """Configure provider with protocol."""
    manager.initialize_tool_adapter()

    caps = manager.capabilities
    print(f"Native tools: {caps.native_tool_calls}")
    print(f"Parallel tools: {caps.parallel_tool_calls}")

async def switch_to_local(manager: ProviderManagerProtocol) -> bool:
    """Switch to local provider."""
    return await manager.switch_provider(
        provider_name="ollama",
        model="qwen2.5:32b"
    )
```

---

## Tool Protocols

### ToolRegistryProtocol

Tool registration and lookup.

```python
@runtime_checkable
class ToolRegistryProtocol(Protocol):
    def register(self, tool: Any) -> None:
        """Register a tool with the registry."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        ...

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        ...

    def get_tool_cost(self, name: str) -> "CostTier":
        """Get the cost tier for a tool."""
        ...

    def register_before_hook(self, hook: Callable[..., Any]) -> None:
        """Register a hook to run before tool execution."""
        ...
```

---

### ToolExecutorProtocol

Tool execution with validation (DIP compliant).

```python
@runtime_checkable
class ToolExecutorProtocol(Protocol):
    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool synchronously.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments as dictionary
            context: Optional execution context (e.g., workspace, session info)

        Returns:
            Tool execution result
        """
        ...

    async def aexecute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute a tool asynchronously.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments as dictionary
            context: Optional execution context (e.g., workspace, session info)

        Returns:
            Tool execution result
        """
        ...

    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """Validate tool arguments before execution.

        Checks that the provided arguments match the tool's expected schema.
        Should be called before execute() or aexecute() to ensure valid input.

        Args:
            tool_name: Name of tool to validate against
            arguments: Arguments to validate

        Returns:
            True if arguments are valid for the tool, False otherwise
        """
        ...
```

**Usage Example:**

```python
def run_tool(
    executor: ToolExecutorProtocol,
    tool: str,
    args: dict
) -> Any:
    """Execute tool with validation."""
    if executor.validate_arguments(tool, args):
        return await executor.aexecute(tool, args)
    raise ValueError(f"Invalid arguments for {tool}")

# Mock in tests
mock_executor = MagicMock(spec=ToolExecutorProtocol)
mock_executor.validate_arguments.return_value = True
mock_executor.aexecute.return_value = {"result": "success"}
```

---

### ToolPipelineProtocol

Tool execution pipeline with budgeting and caching.

```python
@runtime_checkable
class ToolPipelineProtocol(Protocol):
    @property
    def calls_used(self) -> int:
        """Number of tool calls used in current session."""
        ...

    @property
    def budget(self) -> int:
        """Maximum tool calls allowed."""
        ...

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> "ToolCallResult":
        """Execute a tool call.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        ...

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        ...
```

---

## Coordinator Protocols
