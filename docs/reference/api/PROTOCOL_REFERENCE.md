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

### IToolSelectionCoordinator

Intelligent tool selection and routing.

**Purpose:** Extracts ~650 lines of tool selection logic from orchestrator following SRP.

```python
@runtime_checkable
class IToolSelectionCoordinator(Protocol):
    def get_recommended_search_tool(
        self,
        query: str,
        context: Optional["AgentToolSelectionContext"] = None,
    ) -> Optional[str]:
        """Get recommended search tool for a query.

        Analyzes the query to determine which search tool (semantic, grep,
        code_search, etc.) would be most appropriate.

        Args:
            query: Search query string
            context: Optional selection context (stage, task type, history)

        Returns:
            Recommended tool name or None if no recommendation
        """
        ...

    def route_search_query(
        self,
        query: str,
        available_tools: Set[str],
    ) -> str:
        """Route a search query to the appropriate tool.

        Determines the best search tool based on query characteristics
        and available tools.

        Args:
            query: Search query string
            available_tools: Set of available tool names

        Returns:
            Selected tool name
        """
        ...

    def detect_mentioned_tools(
        self,
        prompt: str,
        available_tools: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Detect tools mentioned in a prompt.

        Scans the prompt for explicit tool mentions (e.g., "use grep to
        find...", "run the web_search tool").

        Args:
            prompt: Prompt text to scan
            available_tools: Optional set of available tools (defaults to all)

        Returns:
            Set of detected tool names
        """
        ...

    def classify_task_keywords(
        self,
        task: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Classify task type using keyword analysis.

        Determines if a task is primarily analysis, action, or creation
        based on keyword presence.

        Args:
            task: Task description
            conversation_history: Optional conversation history for context

        Returns:
            Task type: "analysis", "action", or "creation"
        """
        ...

    def classify_task_with_context(
        self,
        task: str,
        context: Optional["AgentToolSelectionContext"] = None,
    ) -> str:
        """Classify task type with full context.

        Enhanced task classification using conversation stage, recent tools,
        and other context information.

        Args:
            task: Task description
            context: Selection context with stage, history, recent tools

        Returns:
            Task type: "analysis", "action", or "creation"
        """
        ...

    def should_use_tools(
        self,
        message: str,
        model_supports_tools: bool = True,
    ) -> bool:
        """Determine if tools should be used for a message.

        Analyzes the message to determine if tool use is appropriate.

        Args:
            message: User message
            model_supports_tools: Whether the model supports tool calling

        Returns:
            True if tools should be used
        """
        ...

    def extract_required_files(
        self,
        prompt: str,
    ) -> Set[str]:
        """Extract required files from a prompt.

        Parses the prompt to find file paths that are explicitly mentioned
        or implied as dependencies.

        Args:
            prompt: Prompt text to parse

        Returns:
            Set of required file paths
        """
        ...

    def extract_required_outputs(
        self,
        prompt: str,
    ) -> Set[str]:
        """Extract required outputs from a prompt.

        Parses the prompt to find output specifications (file paths,
        variable names, etc.) that the task should produce.

        Args:
            prompt: Prompt text to parse

        Returns:
            Set of required output identifiers
        """
        ...
```

---

### ToolCoordinatorProtocol

Tool coordination operations (WS-D refactoring).

```python
@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    async def select_tools(self, context: Any) -> List[Any]:
        """Select appropriate tools for the current context.

        Args:
            context: TaskContext with message, task_type, etc.

        Returns:
            List of selected tool definitions
        """
        ...

    def get_remaining_budget(self) -> int:
        """Get remaining tool call budget.

        Returns:
            Number of tool calls remaining
        """
        ...

    def consume_budget(self, amount: int = 1) -> None:
        """Consume tool call budget.

        Args:
            amount: Number of budget units to consume
        """
        ...

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Any] = None,
    ) -> Any:
        """Execute tool calls through the pipeline.

        Args:
            tool_calls: List of tool calls to execute
            context: Optional task context

        Returns:
            PipelineExecutionResult with execution details
        """
        ...

    def reset_budget(self, new_budget: Optional[int] = None) -> None:
        """Reset the tool budget.

        Args:
            new_budget: New budget to set, or use default
        """
        ...

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no budget remaining
        """
        ...
```

---

### StateCoordinatorProtocol

State coordination operations.

```python
@runtime_checkable
class StateCoordinatorProtocol(Protocol):
    def get_current_stage(self) -> Any:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage
        """
        ...

    def transition_to(
        self,
        stage: Any,
        reason: str = "",
        tool_name: Optional[str] = None,
    ) -> bool:
        """Transition to a new conversation stage.

        Args:
            stage: Target stage to transition to
            reason: Reason for the transition
            tool_name: Tool that triggered the transition

        Returns:
            True if transition was successful
        """
        ...

    def get_message_history(self) -> List[Any]:
        """Get the full message history.

        Returns:
            List of Message objects
        """
        ...

    def get_recent_messages(
        self,
        limit: int = 10,
        include_system: bool = False,
    ) -> List[Any]:
        """Get recent messages from history.

        Args:
            limit: Maximum messages to return
            include_system: Whether to include system messages

        Returns:
            List of recent Message objects
        """
        ...

    def is_in_exploration_phase(self) -> bool:
        """Check if currently in exploration phase.

        Returns:
            True if in exploration phase
        """
        ...

    def is_in_execution_phase(self) -> bool:
        """Check if currently in execution phase.

        Returns:
            True if in execution phase
        """
        ...
```

---

### PromptCoordinatorProtocol

Prompt coordination operations.

```python
@runtime_checkable
class PromptCoordinatorProtocol(Protocol):
    def build_system_prompt(
        self,
        context: Any,
        include_hints: bool = True,
    ) -> str:
        """Build the complete system prompt.

        Args:
            context: TaskContext for prompt building
            include_hints: Whether to include task hints

        Returns:
            Complete system prompt string
        """
        ...

    def add_task_hint(self, task_type: str, hint: str) -> None:
        """Add or update a task-type hint.

        Args:
            task_type: Task type (e.g., "edit", "debug")
            hint: Hint text for this task type
        """
        ...

    def get_task_hint(self, task_type: str) -> Optional[str]:
        """Get the hint for a task type.

        Args:
            task_type: Task type to get hint for

        Returns:
            Hint string or None
        """
        ...

    def add_section(
        self,
        name: str,
        content: str,
        priority: Optional[int] = None,
    ) -> None:
        """Add a runtime section to be included in prompts.

        Args:
            name: Section name (unique identifier)
            content: Section content
            priority: Optional priority
        """
        ...

    def set_grounding_mode(self, mode: str) -> None:
        """Set the grounding rules mode.

        Args:
            mode: "minimal" or "extended"
        """
        ...
```

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

### RecoveryHandlerProtocol

Model failure recovery.

```python
@runtime_checkable
class RecoveryHandlerProtocol(Protocol):
    def detect_failure(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]],
        elapsed_time: float,
        session_idle_timeout: float,
        quality_score: float,
        consecutive_failures: int,
        recent_responses: Optional[List[str]],
        context_utilization: Optional[float],
    ) -> Optional[Any]:
        """Detect failure type from response characteristics.

        Returns:
            FailureType if failure detected, None otherwise
        """
        ...

    async def recover(
        self,
        failure_type: Any,
        provider: str,
        model: str,
        content: str,
        tool_calls_made: int,
        tool_budget: int,
        iteration_count: int,
        max_iterations: int,
        elapsed_time: float,
        session_idle_timeout: float,
        current_temperature: float,
        consecutive_failures: int,
        mentioned_tools: Optional[List[str]],
        recent_responses: Optional[List[str]],
        quality_score: float,
        task_type: str,
        is_analysis_task: bool,
        is_action_task: bool,
        session_id: Optional[str],
    ) -> Any:
        """Attempt recovery using appropriate strategy.

        Returns:
            RecoveryOutcome with action to take
        """
        ...

    def record_outcome(
        self,
        success: bool,
        quality_improvement: float
    ) -> None:
        """Record recovery outcome for Q-learning."""
        ...

    def reset_session(self, session_id: str) -> None:
        """Reset recovery state for a new session."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about recovery system."""
        ...
```

---

## Utility Protocols

### ToolCacheProtocol

Tool result caching.

```python
@runtime_checkable
class ToolCacheProtocol(Protocol):
    def get(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached result for a tool call."""
        ...

    def set(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any
    ) -> None:
        """Cache a tool result."""
        ...

    def invalidate(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> None:
        """Invalidate a cached result."""
        ...
```

---

### TaskTrackerProtocol

Task tracking.

```python
@runtime_checkable
class TaskTrackerProtocol(Protocol):
    def start_task(self, task_id: str, description: str) -> None:
        """Start tracking a task."""
        ...

    def complete_task(self, task_id: str) -> None:
        """Mark task as complete."""
        ...

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks."""
        ...

    def is_loop_detected(self) -> bool:
        """Check if execution loop is detected."""
        ...
```

---

### ToolOutputFormatterProtocol

Tool output formatting.

```python
@runtime_checkable
class ToolOutputFormatterProtocol(Protocol):
    def format(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format tool output for LLM consumption.

        Args:
            tool_name: Name of the tool
            result: Raw tool result
            context: Optional formatting context

        Returns:
            Formatted output string
        """
        ...
```

---

### ArgumentNormalizerProtocol

Argument normalization.

```python
@runtime_checkable
class ArgumentNormalizerProtocol(Protocol):
    def normalize(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize tool arguments.

        Handles malformed arguments, type coercion, etc.
        """
        ...
```

---

## Tool Access Control Protocols

### IToolAccessController

Unified tool access control.

```python
@runtime_checkable
class IToolAccessController(Protocol):
    """Protocol for unified tool access control.

    The controller applies layers in precedence order:
    Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

    A tool is blocked if ANY layer denies it.
    """

    def check_access(
        self,
        tool_name: str,
        context: Optional[ToolAccessContext] = None
    ) -> ToolAccessDecision:
        """Check if a tool is allowed in the given context.

        Args:
            tool_name: Name of the tool to check
            context: Access context (mode, stage, intent, etc.)

        Returns:
            ToolAccessDecision with result and explanation
        """
        ...

    def filter_tools(
        self,
        tools: List[str],
        context: Optional[ToolAccessContext] = None
    ) -> Tuple[List[str], List[ToolAccessDecision]]:
        """Filter a list of tools to only allowed ones.

        Args:
            tools: List of tool names to filter
            context: Access context

        Returns:
            Tuple of (allowed_tools, denial_decisions)
        """
        ...

    def get_allowed_tools(
        self,
        context: Optional[ToolAccessContext] = None
    ) -> Set[str]:
        """Get all tools allowed in the given context.

        Args:
            context: Access context

        Returns:
            Set of allowed tool names
        """
        ...

    def explain_decision(
        self,
        tool_name: str,
        context: Optional[ToolAccessContext] = None
    ) -> str:
        """Get detailed explanation for a tool access decision.

        Args:
            tool_name: Name of the tool
            context: Access context

        Returns:
            Human-readable explanation
        """
        ...
```

---

## Budget Management Protocols

### IBudgetManager

Unified budget management.

```python
@runtime_checkable
class IBudgetManager(Protocol):
    """Protocol for unified budget management.

    Centralizes all budget tracking with consistent multiplier composition:
    effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier
    """

    def get_status(self, budget_type: BudgetType) -> BudgetStatus:
        """Get current status of a budget.

        Args:
            budget_type: Type of budget to check

        Returns:
            BudgetStatus with current usage and limits
        """
        ...

    def consume(self, budget_type: BudgetType, amount: int = 1) -> bool:
        """Consume budget for an operation.

        Args:
            budget_type: Type of budget to consume
            amount: Amount to consume (default 1)

        Returns:
            True if budget was available, False if exhausted
        """
        ...

    def is_exhausted(self, budget_type: BudgetType) -> bool:
        """Check if a budget is exhausted.

        Args:
            budget_type: Type of budget to check

        Returns:
            True if budget is fully consumed
        """
        ...

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set the model-specific multiplier.

        Args:
            multiplier: Model multiplier value
        """
        ...

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set the mode-specific multiplier.

        Args:
            multiplier: Mode multiplier value
        """
        ...

    def set_productivity_multiplier(self, multiplier: float) -> None:
        """Set the productivity multiplier.

        Args:
            multiplier: Productivity multiplier value
        """
        ...

    def reset(self, budget_type: Optional[BudgetType] = None) -> None:
        """Reset budget(s) to initial state.

        Args:
            budget_type: Specific budget to reset, or None for all
        """
        ...

    def get_prompt_budget_info(self) -> Dict[str, Any]:
        """Get budget information for system prompts.

        Returns:
            Dictionary with budget info for prompt building
        """
        ...

    def record_tool_call(
        self,
        tool_name: str,
        is_write_operation: bool = False
    ) -> bool:
        """Record a tool call and consume appropriate budget.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write/modify operation

        Returns:
            True if budget was available
        """
        ...
```

---

**See Also:**
- [API Reference](API_REFERENCE.md) - Main API documentation
- [Provider Reference](PROVIDER_REFERENCE.md) - Provider details
- [Configuration Reference](CONFIGURATION_REFERENCE.md) - Settings reference

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
