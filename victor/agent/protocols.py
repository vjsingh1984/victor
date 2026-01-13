# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protocols for orchestrator services.

Defines interfaces for all injectable services used by AgentOrchestrator.
These protocols enable:
- Type-safe dependency injection
- Easy testing via mock substitution
- Clear component contracts

Usage:
    from victor.agent.protocols import (
        ProviderManagerProtocol,
        ToolRegistryProtocol,
        ConversationControllerProtocol,
    )

    # Type hint with protocol
    def process_with_provider(provider: ProviderManagerProtocol) -> None:
        model = provider.model
        ...

    # Mock in tests
    mock_provider = MagicMock(spec=ProviderManagerProtocol)
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStage
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.tools.base import CostTier


# =============================================================================
# Factory Protocols
# =============================================================================


@runtime_checkable
class IAgentFactory(Protocol):
    """Protocol for unified agent creation.

    Defines interface for creating ANY agent type (foreground, background, team_member)
    using shared infrastructure. This protocol enables:

    - **Single Responsibility**: Factory only handles agent creation
    - **Open/Closed**: Extensible for new agent types via mode parameter
    - **Liskov Substitution**: Any IAgentFactory implementation is interchangeable
    - **Interface Segregation**: Focused protocol with single method
    - **Dependency Inversion**: Consumers depend on abstraction, not concrete factory

    All agent entrypoints (Agent.create, BackgroundAgentManager.start_agent, etc.)
    delegate to implementations of this protocol, ensuring consistent code maintenance
    and eliminating code proliferation.

    Usage:
        from victor.agent.protocols import IAgentFactory

        async def create_researcher(factory: IAgentFactory) -> IAgent:
            return await factory.create_agent(
                mode="foreground",
                task="research",
                config=my_config
            )
    """

    async def create_agent(
        self,
        mode: str,
        config: Optional[Any] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create any agent type using shared infrastructure.

        This is the ONLY method that should create agents. All other entrypoints
        (Agent.create, BackgroundAgentManager.start_agent, Vertical.create_agent)
        must delegate here to ensure consistent code maintenance.

        Args:
            mode: Agent creation mode - "foreground", "background", or "team_member"
            config: Optional unified agent configuration (UnifiedAgentConfig)
            task: Optional task description (for background agents)
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance (Agent, BackgroundAgent, or TeamMember/SubAgent)

        Examples:
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
        """
        ...


@runtime_checkable
class IAgent(Protocol):
    """Canonical agent protocol.

    ALL agent types (Agent, BackgroundAgent, TeamMember, SubAgent) must implement
    this protocol to ensure Liskov Substitution Principle compliance.

    This enables:
    - Polymorphic agent handling
    - Type-safe agent composition
    - Consistent agent interfaces across all contexts
    """

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


# =============================================================================
# Provider Protocols
# =============================================================================


@runtime_checkable
class ProviderManagerProtocol(Protocol):
    """Protocol for provider management.

    Manages LLM provider lifecycle, switching, and health monitoring.
    """

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

    async def switch_provider(self, provider_name: str, model: Optional[str] = None) -> bool:
        """Switch to a different provider/model.

        Args:
            provider_name: Name of provider to switch to
            model: Optional model to use

        Returns:
            True if switch successful
        """
        ...


# =============================================================================
# Tool Selection Data Classes
# =============================================================================


from dataclasses import dataclass, field


@dataclass
class AgentToolSelectionContext:
    """Agent-level context for tool selection decisions.

    Renamed from ToolSelectionContext to be semantically distinct:
    - AgentToolSelectionContext (here): Basic agent-level context
    - VerticalToolSelectionContext (victor.core.verticals.protocols.tool_provider): Vertical-specific
    - CrossVerticalToolSelectionContext (victor.tools.selection.protocol): Extended cross-vertical

    Provides conversation history, stage info, and task metadata
    to tool selectors for intelligent tool filtering.
    """

    # Conversation stage (e.g., PLANNING, EXECUTING, REVIEWING)
    stage: Optional[str] = None

    # Conversation stage object (typed version)
    conversation_stage: Optional[Any] = None

    # Task type from analysis (e.g., "analysis", "action", "create")
    task_type: str = "default"

    # Recent tool execution history
    recent_tools: List[str] = field(default_factory=list)

    # Current conversation turn number
    turn_number: int = 0

    # Whether this is a continuation of previous work
    is_continuation: bool = False

    # Maximum tools to select
    max_tools: int = 15

    # Planned tools from dependency planning (tool names pre-selected)
    planned_tools: Optional[List[str]] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Backward compatibility alias
ToolSelectionContext = AgentToolSelectionContext


@dataclass
class ToolSelectorFeatures:
    """Features supported by a tool selector implementation.

    Used to advertise capabilities and for feature detection.
    """

    # ML-based semantic matching (requires embeddings)
    supports_semantic_matching: bool = False

    # Context-aware selection (conversation stage, history)
    supports_context_awareness: bool = False

    # Cost-based tool optimization
    supports_cost_optimization: bool = False

    # Learning from usage patterns
    supports_usage_learning: bool = False

    # Workflow pattern detection
    supports_workflow_patterns: bool = False

    # Whether embeddings are required
    requires_embeddings: bool = False


# =============================================================================
# Tool Protocols
# =============================================================================


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registry.

    Manages tool registration, lookup, and cost tiers.
    """

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


# NOTE: ToolSelectorProtocol removed in Phase 9 migration
# Use IToolSelector from victor.protocols.tool_selector instead


@runtime_checkable
class ToolPipelineProtocol(Protocol):
    """Protocol for tool execution pipeline.

    Coordinates tool execution flow, budget enforcement, and caching.
    """

    @property
    def calls_used(self) -> int:
        """Number of tool calls used in current session."""
        ...

    @property
    def budget(self) -> int:
        """Maximum tool calls allowed."""
        ...

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> "ToolCallResult":
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


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Protocol for tool execution - DIP compliant.

    Handles individual tool execution with validation and context support.
    This protocol enables dependency inversion by allowing consumers to
    depend on the abstraction rather than concrete implementations.

    The protocol provides:
    - Synchronous and asynchronous execution methods
    - Argument validation before execution
    - Optional context passing for execution environment

    Usage:
        from victor.agent.protocols import ToolExecutorProtocol

        def run_tool(executor: ToolExecutorProtocol, tool: str, args: dict) -> Any:
            if executor.validate_arguments(tool, args):
                return await executor.aexecute(tool, args)
            raise ValueError(f"Invalid arguments for {tool}")

        # Mock in tests
        mock_executor = MagicMock(spec=ToolExecutorProtocol)
        mock_executor.validate_arguments.return_value = True
    """

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


# =============================================================================
# Coordinator Protocols
# =============================================================================


@runtime_checkable
class IToolSelectionCoordinator(Protocol):
    """Protocol for tool selection coordination.

    Manages intelligent tool selection, routing, and classification
    for optimizing tool usage based on conversation context and task type.

    This coordinator extracts ~650 lines of tool selection logic from
    the orchestrator, following SRP (Single Responsibility Principle).

    Key Responsibilities:
    - Tool selection and routing (semantic vs keyword matching)
    - Task classification (analysis, action, creation)
    - Tool mention detection from prompts
    - Required files/outputs extraction
    - Tool capability checking
    """

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


# =============================================================================
# Conversation Protocols
# =============================================================================


@runtime_checkable
class ConversationControllerProtocol(Protocol):
    """Protocol for conversation management.

    Manages message history, context tracking, and compaction.
    """

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


@runtime_checkable
class ConversationStateMachineProtocol(Protocol):
    """Protocol for conversation state machine.

    Tracks conversation stage (INITIAL, PLANNING, EXECUTING, etc.).
    """

    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage."""
        ...

    def get_current_stage(self) -> "ConversationStage":
        """Get current conversation stage (alias)."""
        ...

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution for stage inference."""
        ...

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record a message for stage inference."""
        ...


@runtime_checkable
class MessageHistoryProtocol(Protocol):
    """Protocol for message history.

    Manages raw message storage.
    """

    def add_message(self, role: str, content: str, **kwargs: Any) -> Any:
        """Add a message."""
        ...

    def get_messages_for_provider(self) -> List[Any]:
        """Get all messages for provider."""
        ...

    def clear(self) -> None:
        """Clear message history."""
        ...


# =============================================================================
# Streaming Protocols
# =============================================================================


@dataclass
class StreamingToolChunk:
    """Represents a chunk of streaming tool output.

    Used by StreamingToolAdapter to emit real-time updates during
    tool execution, enabling unified streaming behavior through ToolPipeline.

    Attributes:
        tool_name: Name of the tool being executed
        tool_call_id: Unique identifier for this tool call
        chunk_type: Type of chunk (start, progress, result, error, cache_hit)
        content: Chunk payload (varies by chunk_type)
        is_final: Whether this is the final chunk for this tool call
        metadata: Optional additional context
    """

    tool_name: str
    tool_call_id: str
    chunk_type: str  # "start", "progress", "result", "error", "cache_hit"
    content: Any
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StreamingToolAdapterProtocol(Protocol):
    """Protocol for streaming tool execution.

    Provides a unified streaming interface that wraps ToolPipeline,
    enabling real-time tool execution updates while preserving all
    ToolPipeline features (caching, middleware, callbacks, budget, etc.).

    This adapter solves the dual execution path problem where:
    - Batch path: Uses ToolPipeline with full feature support
    - Streaming path: Previously bypassed ToolPipeline using self.tools.execute()

    Now both paths route through this adapter -> ToolPipeline.
    """

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


@runtime_checkable
class StreamingControllerProtocol(Protocol):
    """Protocol for streaming session management.

    Manages streaming lifecycle and metrics collection.
    """

    def start_session(self, session_id: str) -> Any:
        """Start a new streaming session."""
        ...

    def end_session(self, session_id: str) -> None:
        """End a streaming session."""
        ...

    def get_active_session(self) -> Optional[Any]:
        """Get the currently active session."""
        ...


# =============================================================================
# Analysis Protocols
# =============================================================================


@runtime_checkable
class TaskAnalyzerProtocol(Protocol):
    """Protocol for task analysis.

    Analyzes user prompts for complexity, intent, and routing.
    """

    def analyze(self, prompt: str) -> Dict[str, Any]:
        """Analyze a user prompt.

        Returns:
            Analysis results (complexity, intent, etc.)
        """
        ...

    def classify_complexity(self, prompt: str) -> Any:
        """Classify task complexity."""
        ...

    def detect_intent(self, prompt: str) -> Any:
        """Detect user intent."""
        ...


@runtime_checkable
class ComplexityClassifierProtocol(Protocol):
    """Protocol for complexity classification."""

    def classify(self, prompt: str) -> Any:
        """Classify prompt complexity.

        Returns:
            TaskComplexity enum value
        """
        ...


@runtime_checkable
class ActionAuthorizerProtocol(Protocol):
    """Protocol for action authorization."""

    def authorize(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if an action is authorized.

        Returns:
            True if authorized
        """
        ...

    def detect_intent(self, prompt: str) -> Any:
        """Detect action intent from prompt."""
        ...


@runtime_checkable
class SearchRouterProtocol(Protocol):
    """Protocol for search routing."""

    def route(self, query: str) -> Any:
        """Route a search query to appropriate handler.

        Returns:
            SearchRoute with type and parameters
        """
        ...


# =============================================================================
# Observability Protocols
# =============================================================================


@runtime_checkable
class ObservabilityProtocol(Protocol):
    """Protocol for observability integration.

    Provides event emission for monitoring and tracing.
    """

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], tool_id: str) -> None:
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


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any], iteration: int) -> None:
        """Record tool start metrics."""
        ...

    def on_tool_complete(self, result: Any) -> None:
        """Record tool completion metrics."""
        ...

    def on_streaming_session_complete(self, session: Any) -> None:
        """Record session completion metrics."""
        ...


# =============================================================================
# Cache Protocols
# =============================================================================


@runtime_checkable
class ToolCacheProtocol(Protocol):
    """Protocol for tool result caching."""

    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for a tool call."""
        ...

    def set(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> None:
        """Cache a tool result."""
        ...

    def invalidate(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Invalidate a cached result."""
        ...


# =============================================================================
# Task Tracking Protocols
# =============================================================================


@runtime_checkable
class TaskTrackerProtocol(Protocol):
    """Protocol for task tracking."""

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


# =============================================================================
# Output Formatting Protocols
# =============================================================================


@runtime_checkable
class ToolOutputFormatterProtocol(Protocol):
    """Protocol for tool output formatting."""

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


@runtime_checkable
class ResponseSanitizerProtocol(Protocol):
    """Protocol for response sanitization."""

    def sanitize(self, response: str) -> str:
        """Sanitize model response."""
        ...


# =============================================================================
# Utility Protocols
# =============================================================================


@runtime_checkable
class ArgumentNormalizerProtocol(Protocol):
    """Protocol for argument normalization."""

    def normalize(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tool arguments.

        Handles malformed arguments, type coercion, etc.
        """
        ...


@runtime_checkable
class ProjectContextProtocol(Protocol):
    """Protocol for project context loading."""

    @property
    def content(self) -> Optional[str]:
        """Get loaded project context content."""
        ...

    def load(self) -> None:
        """Load project context from file."""
        ...

    def get_system_prompt_addition(self) -> str:
        """Get context as system prompt addition."""
        ...


# =============================================================================
# Component Lifecycle Protocols
# =============================================================================


@runtime_checkable
class CodeExecutionManagerProtocol(Protocol):
    """Protocol for code execution management."""

    def start(self) -> None:
        """Start the execution manager."""
        ...

    def stop(self) -> None:
        """Stop the execution manager."""
        ...


@runtime_checkable
class WorkflowRegistryProtocol(Protocol):
    """Protocol for workflow registry."""

    def register(self, workflow: Any) -> None:
        """Register a workflow."""
        ...

    def get(self, name: str) -> Optional[Any]:
        """Get a workflow by name."""
        ...


@runtime_checkable
class ToolRegistrarProtocol(Protocol):
    """Protocol for tool registration and plugin management."""

    def set_background_task_callback(self, callback: Callable[[Any], Any]) -> None:
        """Set callback for background task creation."""
        ...


@runtime_checkable
class UsageAnalyticsProtocol(Protocol):
    """Protocol for usage analytics."""

    def record_tool_selection(
        self,
        tool_name: str,
        score: float,
        selected: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a tool selection decision."""
        ...

    def end_session(self) -> None:
        """End the current analytics session."""
        ...

    @classmethod
    def get_instance(cls, config: Any) -> "UsageAnalyticsProtocol":
        """Get singleton instance."""
        ...


@runtime_checkable
class ToolSequenceTrackerProtocol(Protocol):
    """Protocol for tool sequence tracking."""

    def record_transition(self, from_tool: str, to_tool: str) -> None:
        """Record a tool-to-tool transition."""
        ...

    def get_next_tool_suggestions(
        self, current_tool: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get suggested next tools based on patterns."""
        ...


@runtime_checkable
class ContextCompactorProtocol(Protocol):
    """Protocol for context compaction."""

    def maybe_compact_proactively(self) -> bool:
        """Attempt proactive compaction if threshold reached.

        Returns:
            True if compaction occurred
        """
        ...

    def truncate_tool_result(
        self, tool_name: str, result: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Truncate tool result if needed."""
        ...


# =============================================================================
# Recovery Protocols
# =============================================================================


# =============================================================================
# Mode Controller Protocols
# =============================================================================


@runtime_checkable
class ModeControllerProtocol(Protocol):
    """Protocol for agent mode control.

    Controls agent modes (BUILD, PLAN, EXPLORE) that modify agent behavior
    for different operational contexts.
    """

    @property
    def current_mode(self) -> Any:
        """Get the current agent mode."""
        ...

    @property
    def config(self) -> Any:
        """Get the current mode configuration."""
        ...

    def switch_mode(self, new_mode: Any) -> bool:
        """Switch to a new mode.

        Args:
            new_mode: The mode to switch to

        Returns:
            True if switch was successful
        """
        ...

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed
        """
        ...

    def get_tool_priority(self, tool_name: str) -> float:
        """Get priority adjustment for a tool in current mode.

        Args:
            tool_name: Name of the tool

        Returns:
            Priority multiplier (1.0 = no adjustment)
        """
        ...

    def get_system_prompt_addition(self) -> str:
        """Get additional system prompt text for current mode."""
        ...


# =============================================================================
# Deduplication Protocols
# =============================================================================


@runtime_checkable
class ToolDeduplicationTrackerProtocol(Protocol):
    """Protocol for tool call deduplication tracking.

    Tracks recent tool calls to detect and prevent redundant operations.
    """

    def add_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Add a tool call to the tracker.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
        """
        ...

    def is_redundant(self, tool_name: str, args: Dict[str, Any], explain: bool = False) -> bool:
        """Check if a tool call is redundant given recent history.

        Args:
            tool_name: Name of the tool to call
            args: Tool arguments
            explain: If True, log explanation for why call is redundant

        Returns:
            True if the call is likely redundant, False otherwise
        """
        ...

    def clear(self) -> None:
        """Clear all tracked tool calls."""
        ...

    def get_recent_calls(self, limit: Optional[int] = None) -> List[Any]:
        """Get recent tool calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent tool calls (most recent first)
        """
        ...


# =============================================================================
# Embedding Store Protocols
# =============================================================================


@runtime_checkable
class ConversationEmbeddingStoreProtocol(Protocol):
    """Protocol for conversation embedding storage.

    Provides semantic search over conversation history using embeddings.
    """

    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized."""
        ...

    async def initialize(self) -> None:
        """Initialize the embedding store."""
        ...

    async def search_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[Any]:
        """Search for semantically similar messages.

        Args:
            query: Query text to search for
            session_id: Optional session to scope search
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            exclude_message_ids: Message IDs to exclude

        Returns:
            List of search results (message_id + similarity)
        """
        ...

    async def delete_session(self, session_id: str) -> int:
        """Delete all embeddings for a session.

        Args:
            session_id: Session ID to delete

        Returns:
            Number of embeddings deleted
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...


# =============================================================================
# Recovery Protocols
# =============================================================================


@runtime_checkable
class RecoveryHandlerProtocol(Protocol):
    """Protocol for model failure recovery.

    Provides a high-level interface for detecting and recovering from
    model failures, stuck states, and hallucinations. Integrates with:
    - Q-learning for adaptive strategy selection
    - UsageAnalytics for telemetry
    - ContextCompactor for proactive compaction
    """

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

    def record_outcome(self, success: bool, quality_improvement: float) -> None:
        """Record recovery outcome for Q-learning."""
        ...

    def reset_session(self, session_id: str) -> None:
        """Reset recovery state for a new session."""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about recovery system."""
        ...


@runtime_checkable
class StreamingRecoveryCoordinatorProtocol(Protocol):
    """Protocol for recovery coordination during streaming sessions.

    Centralizes all recovery and error handling logic for streaming chat,
    including:
    - Condition checking (time limits, iteration limits, budget, progress)
    - Action handling (empty responses, blocked tools, forced completion)
    - Recovery integration (with RecoveryHandler and OrchestratorRecoveryIntegration)
    - Filtering and truncation (blocked tools, budget limits)
    - Prompt and message generation (recovery prompts, fallback messages)
    - Metrics formatting (completion, budget exhausted)

    Note: Renamed from RecoveryCoordinatorProtocol to avoid confusion with
    victor.agent.recovery.coordinator.RecoveryCoordinator (SOLID recovery system).

    Extracted from CRITICAL-001 Phase 2A.
    """

    def check_time_limit(self, ctx: Any) -> Optional[Any]:
        """Check if session has exceeded time limit.

        Returns:
            StreamChunk if time limit reached, None otherwise
        """
        ...

    def check_iteration_limit(self, ctx: Any) -> Optional[Any]:
        """Check if session has exceeded iteration limit.

        Returns:
            StreamChunk if iteration limit reached, None otherwise
        """
        ...

    def check_natural_completion(
        self, ctx: Any, has_tool_calls: bool, content_length: int
    ) -> Optional[Any]:
        """Check for natural completion (no tool calls, sufficient content).

        Returns:
            StreamChunk if natural completion detected, None otherwise
        """
        ...

    def check_tool_budget(self, ctx: Any) -> bool:
        """Check if tool budget has been exhausted.

        Returns:
            True if budget exhausted, False otherwise
        """
        ...

    def check_progress(self, ctx: Any) -> bool:
        """Check if session is making progress (not looping).

        Returns:
            True if making progress, False if stuck/looping
        """
        ...

    def check_blocked_threshold(self, ctx: Any, all_blocked: bool) -> Optional[Tuple[Any, bool]]:
        """Check if too many tools have been blocked.

        Returns:
            Tuple of (chunk, should_clear_tools) if threshold exceeded, None otherwise
        """
        ...

    def check_force_action(self, ctx: Any) -> Tuple[bool, Optional[str]]:
        """Check if recovery handler recommends force action.

        Returns:
            Tuple of (should_force, action_type)
        """
        ...

    def handle_empty_response(self, ctx: Any) -> Tuple[Optional[Any], bool]:
        """Handle empty model response.

        Returns:
            Tuple of (StreamChunk if threshold exceeded, should_force_completion flag)
        """
        ...

    def handle_blocked_tool(
        self, ctx: Any, tool_name: str, tool_args: Dict[str, Any], block_reason: str
    ) -> Any:
        """Handle blocked tool call.

        Returns:
            StreamChunk with block notification
        """
        ...

    def handle_force_tool_execution(self, ctx: Any) -> Tuple[bool, Optional[List[Any]]]:
        """Handle forced tool execution.

        Returns:
            Tuple of (should_execute, chunks)
        """
        ...

    def handle_force_completion(self, ctx: Any) -> Optional[List[Any]]:
        """Handle forced completion.

        Returns:
            List of StreamChunks if forced completion, None otherwise
        """
        ...

    def handle_loop_warning(self, ctx: Any) -> Optional[List[Any]]:
        """Handle loop detection warning.

        Returns:
            List of warning chunks, None if no loop detected
        """
        ...

    async def handle_recovery_with_integration(
        self,
        ctx: Any,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]],
        message_adder: Any,
    ) -> Any:
        """Handle response using the recovery integration.

        Returns:
            RecoveryAction with action to take (continue, retry, abort, force_summary)
        """
        ...

    def apply_recovery_action(
        self, recovery_action: Any, ctx: Any, message_adder: Any
    ) -> Optional[Any]:
        """Apply a recovery action from the recovery integration.

        Returns:
            StreamChunk if action requires immediate yield, None otherwise
        """
        ...

    def filter_blocked_tool_calls(
        self, ctx: Any, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Any], int]:
        """Filter out blocked tool calls.

        Returns:
            Tuple of (filtered_tool_calls, blocked_chunks, blocked_count)
        """
        ...

    def truncate_tool_calls(
        self, ctx: Any, tool_calls: List[Dict[str, Any]], max_calls: int
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Truncate tool calls to budget limit.

        Returns:
            Tuple of (truncated_tool_calls, was_truncated)
        """
        ...

    def get_recovery_prompts(self, ctx: Any) -> List[str]:
        """Get recovery prompts for current context.

        Returns:
            List of recovery prompts
        """
        ...

    def get_recovery_fallback_message(self, ctx: Any) -> str:
        """Get fallback message when recovery fails.

        Returns:
            Fallback message
        """
        ...

    def should_use_tools_for_recovery(self, ctx: Any) -> bool:
        """Determine if tools should be used during recovery.

        Returns:
            True if tools should be used, False otherwise
        """
        ...

    def format_completion_metrics(self, ctx: Any) -> Dict[str, Any]:
        """Format completion metrics for display.

        Returns:
            Dictionary of formatted metrics
        """
        ...

    def format_budget_exhausted_metrics(self, ctx: Any) -> Dict[str, Any]:
        """Format budget exhausted metrics.

        Returns:
            Dictionary of formatted metrics
        """
        ...

    def generate_tool_result_chunks(self, results: List[Any], ctx: Any) -> List[Any]:
        """Generate stream chunks from tool results.

        Returns:
            List of StreamChunk objects
        """
        ...


@runtime_checkable
class ChunkGeneratorProtocol(Protocol):
    """Protocol for streaming chunk generation.

    Centralizes all streaming chunk generation operations for streaming chat,
    including:
    - Tool-related chunks (start, result)
    - Status chunks (thinking, budget errors, force response)
    - Content chunks (metrics, content, final markers)
    - Budget chunks (exhausted warnings)

    Extracted from CRITICAL-001 Phase 2B.
    """

    def generate_tool_start_chunk(
        self, tool_name: str, tool_args: Dict[str, Any], status_msg: str
    ) -> Any:
        """Generate chunk indicating tool execution start.

        Args:
            tool_name: Name of the tool being executed
            tool_args: Tool arguments
            status_msg: Status message to display

        Returns:
            StreamChunk with tool start metadata
        """
        ...

    def generate_tool_result_chunks(self, result: Dict[str, Any]) -> List[Any]:
        """Generate chunks for tool execution result.

        Args:
            result: Tool execution result dictionary

        Returns:
            List of StreamChunks representing the tool result
        """
        ...

    def generate_thinking_status_chunk(self) -> Any:
        """Generate chunk indicating thinking/processing status.

        Returns:
            StreamChunk with thinking status metadata
        """
        ...

    def generate_budget_error_chunk(self) -> Any:
        """Generate chunk for budget limit error.

        Returns:
            StreamChunk with budget limit error message
        """
        ...

    def generate_force_response_error_chunk(self) -> Any:
        """Generate chunk for forced response error.

        Returns:
            StreamChunk with force response error message
        """
        ...

    def generate_final_marker_chunk(self) -> Any:
        """Generate final marker chunk to signal stream completion.

        Returns:
            StreamChunk with is_final=True
        """
        ...

    def generate_metrics_chunk(
        self, metrics_line: str, is_final: bool = False, prefix: str = "\n\n"
    ) -> Any:
        """Generate chunk for metrics display.

        Args:
            metrics_line: Formatted metrics line
            is_final: Whether this is the final chunk
            prefix: Prefix before metrics line (default: double newline)

        Returns:
            StreamChunk with formatted metrics content
        """
        ...

    def generate_content_chunk(self, content: str, is_final: bool = False, suffix: str = "") -> Any:
        """Generate chunk for content display.

        Args:
            content: Sanitized content to display
            is_final: Whether this is the final chunk
            suffix: Optional suffix to append

        Returns:
            StreamChunk with content and optional suffix
        """
        ...

    def get_budget_exhausted_chunks(self, stream_ctx: Any) -> List[Any]:
        """Get chunks for budget exhaustion warning.

        Args:
            stream_ctx: Streaming context

        Returns:
            List of StreamChunks for budget exhausted warning
        """
        ...


@runtime_checkable
class ToolPlannerProtocol(Protocol):
    """Protocol for tool planning and intent-based filtering.

    Centralizes all tool planning operations, including:
    - Tool sequence planning using dependency graph
    - Goal inference from user messages
    - Intent-based tool filtering

    Extracted from CRITICAL-001 Phase 2C.
    """

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[Any]:
        """Plan a sequence of tools to satisfy goals.

        Args:
            goals: List of desired outputs
            available_inputs: Optional list of inputs already available

        Returns:
            List of ToolDefinition objects for the planned sequence
        """
        ...

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from user request.

        Args:
            user_message: The user's input message

        Returns:
            List of inferred goal outputs
        """
        ...

    def filter_tools_by_intent(
        self, tools: List[Any], current_intent: Optional[Any] = None
    ) -> List[Any]:
        """Filter tools based on detected user intent.

        Args:
            tools: List of tool definitions
            current_intent: The detected user intent (if None, no filtering)

        Returns:
            Filtered list of tools
        """
        ...


@runtime_checkable
class TaskCoordinatorProtocol(Protocol):
    """Protocol for task coordination and guidance.

    Centralizes all task coordination operations, including:
    - Task preparation with complexity detection
    - Intent-based prompt guards
    - Task-specific guidance and budget adjustments

    Extracted from CRITICAL-001 Phase 2D.
    """

    def prepare_task(
        self, user_message: str, unified_task_type: Any, conversation_controller: Any
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            conversation_controller: Conversation controller for message injection

        Returns:
            Tuple of (task_classification, complexity_tool_budget)
        """
        ...

    def apply_intent_guard(self, user_message: str, conversation_controller: Any) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Args:
            user_message: The user's input message
            conversation_controller: Conversation controller for message injection
        """
        ...

    def apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: Any,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
        conversation_controller: Any,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            is_analysis_task: Whether this is an analysis task
            is_action_task: Whether this is an action-oriented task
            needs_execution: Whether the task requires execution
            max_exploration_iterations: Maximum exploration iterations allowed
            conversation_controller: Conversation controller for message injection
        """
        ...

    @property
    def current_intent(self) -> Any:
        """Get the current detected intent."""
        ...

    @property
    def temperature(self) -> float:
        """Get the current temperature setting."""
        ...

    @property
    def tool_budget(self) -> int:
        """Get the current tool budget."""
        ...

    @property
    def observed_files(self) -> list:
        """Get the list of observed files."""
        ...


# =============================================================================
# Utility Service Protocols
# =============================================================================


@runtime_checkable
class DebugLoggerProtocol(Protocol):
    """Protocol for debug logging service.

    Provides clean, scannable debug output focused on meaningful events.
    """

    def reset(self) -> None:
        """Reset state for new conversation."""
        ...

    def log_iteration_start(self, iteration: int, **context: Any) -> None:
        """Log iteration start."""
        ...

    def log_iteration_end(
        self, iteration: int, has_tool_calls: bool = False, **context: Any
    ) -> None:
        """Log iteration end summary."""
        ...

    def log_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        iteration: int,
    ) -> None:
        """Log tool call."""
        ...

    def log_tool_result(
        self,
        tool_name: str,
        success: bool,
        output: Any,
        elapsed_ms: float,
    ) -> None:
        """Log tool result."""
        ...


@runtime_checkable
class TaskTypeHinterProtocol(Protocol):
    """Protocol for task type hint retrieval.

    Provides task-specific guidance for the LLM.
    """

    def get_hint(self, task_type: str) -> str:
        """Get prompt hint for a specific task type.

        Args:
            task_type: Type of task (edit, search, explain, etc.)

        Returns:
            Formatted hint string for system prompt
        """
        ...


@runtime_checkable
class ReminderManagerProtocol(Protocol):
    """Protocol for context reminder management.

    Manages intelligent injection of context reminders to reduce token waste.
    """

    def reset(self) -> None:
        """Reset state for a new conversation turn."""
        ...

    def update_state(
        self,
        observed_files: Optional[Set[str]] = None,
        executed_tool: Optional[str] = None,
        tool_calls: Optional[int] = None,
        tool_budget: Optional[int] = None,
        task_complexity: Optional[str] = None,
        task_hint: Optional[str] = None,
    ) -> None:
        """Update the current context state."""
        ...

    def add_observed_file(self, file_path: str) -> None:
        """Add a file to the observed files set."""
        ...

    def get_consolidated_reminder(self, force: bool = False) -> Optional[str]:
        """Get a consolidated reminder combining all active reminders."""
        ...


@runtime_checkable
class RLCoordinatorProtocol(Protocol):
    """Protocol for reinforcement learning coordinator.

    Manages all RL learners with unified SQLite storage.
    """

    def record_outcome(
        self,
        learner_name: str,
        outcome: Any,
        vertical: str = "coding",
    ) -> None:
        """Record an outcome for a specific learner."""
        ...

    def get_recommendation(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[Any]:
        """Get recommendation from a learner."""
        ...

    def export_metrics(self) -> Dict[str, Any]:
        """Export all learned values and metrics for monitoring."""
        ...

    def close(self) -> None:
        """Close database connection."""
        ...


@runtime_checkable
class SafetyCheckerProtocol(Protocol):
    """Protocol for safety checking service.

    Detects dangerous operations and requests confirmation.
    """

    def is_write_tool(self, tool_name: str) -> bool:
        """Check if a tool is a write/modify operation."""
        ...

    async def check_and_confirm(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check operation safety and request confirmation if needed.

        Returns:
            Tuple of (should_proceed, optional_rejection_reason)
        """
        ...

    def add_custom_pattern(
        self,
        pattern: str,
        description: str,
        risk_level: str = "HIGH",
        category: str = "custom",
    ) -> None:
        """Add a custom safety pattern from vertical extensions."""
        ...


@runtime_checkable
class AutoCommitterProtocol(Protocol):
    """Protocol for automatic git commits service.

    Handles automatic git commits for AI-assisted changes.
    """

    def is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        ...

    def has_changes(self, files: Optional[List[str]] = None) -> bool:
        """Check if there are uncommitted changes."""
        ...

    def commit_changes(
        self,
        files: Optional[List[str]] = None,
        description: str = "AI-assisted changes",
        change_type: Optional[str] = None,
        scope: Optional[str] = None,
        auto_stage: bool = True,
    ) -> Any:
        """Commit changes to git."""
        ...


@runtime_checkable
class MCPBridgeProtocol(Protocol):
    """Protocol for Model Context Protocol bridge.

    Provides access to MCP tools as Victor tools.
    """

    def configure_client(self, client: Any, prefix: str = "mcp") -> None:
        """Configure the MCP client.

        Args:
            client: MCPClient instance
            prefix: Prefix for tool names
        """
        ...

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP tools as Victor tool definitions with a name prefix."""
        ...


# =============================================================================
# Infrastructure Service Protocols
# =============================================================================


@runtime_checkable
class ToolDependencyGraphProtocol(Protocol):
    """Protocol for tool dependency graph.

    Manages tool dependencies for intelligent execution ordering.
    """

    def add_dependency(self, tool: str, depends_on: str) -> None:
        """Add a dependency relationship between tools.

        Args:
            tool: Tool name that has a dependency
            depends_on: Tool that must be executed first
        """
        ...

    def get_dependencies(self, tool: str) -> List[str]:
        """Get dependencies for a tool.

        Args:
            tool: Tool name

        Returns:
            List of tool names that this tool depends on
        """
        ...

    def get_execution_order(self, tools: List[str]) -> List[str]:
        """Get optimal execution order for a list of tools.

        Args:
            tools: List of tool names

        Returns:
            Ordered list respecting dependencies
        """
        ...


@runtime_checkable
class ToolPluginRegistryProtocol(Protocol):
    """Protocol for tool plugin registry.

    Manages dynamic tool loading from plugins.
    """

    def register_plugin(self, plugin_path: str) -> None:
        """Register a plugin directory.

        Args:
            plugin_path: Path to plugin directory
        """
        ...

    def discover_tools(self) -> List[Any]:
        """Discover and load tools from registered plugins.

        Returns:
            List of discovered tool instances
        """
        ...

    def reload_plugins(self) -> None:
        """Reload all registered plugins."""
        ...


# NOTE: SemanticToolSelectorProtocol removed in Phase 9 migration
# Use IToolSelector from victor.protocols.tool_selector instead


@runtime_checkable
class ProviderRegistryProtocol(Protocol):
    """Protocol for provider registry.

    Manages available LLM providers.
    """

    def register(self, name: str, provider_class: Any) -> None:
        """Register a provider.

        Args:
            name: Provider name
            provider_class: Provider class
        """
        ...

    def get(self, name: str) -> Any:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class
        """
        ...

    def list_providers(self) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of provider names
        """
        ...


# =============================================================================
# Analytics & Observability Protocols
# =============================================================================


@runtime_checkable
class UsageLoggerProtocol(Protocol):
    """Protocol for usage logging service.

    Logs tool and provider usage for analytics.
    """

    def log_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            success: Whether the call succeeded
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        ...

    def log_provider_call(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Log a provider API call.

        Args:
            provider: Provider name
            model: Model identifier
            tokens_used: Number of tokens consumed
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary of usage statistics
        """
        ...


@runtime_checkable
class StreamingMetricsCollectorProtocol(Protocol):
    """Protocol for streaming metrics collection.

    Collects real-time metrics during streaming responses.
    """

    def record_chunk(
        self,
        chunk_size: int,
        timestamp: float,
        **metadata: Any,
    ) -> None:
        """Record a streaming chunk.

        Args:
            chunk_size: Size of the chunk
            timestamp: Timestamp of the chunk
            **metadata: Additional metadata
        """
        ...

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary of streaming metrics
        """
        ...

    def reset(self) -> None:
        """Reset metrics for new session."""
        ...


@runtime_checkable
class IntentClassifierProtocol(Protocol):
    """Protocol for intent classification service.

    Classifies user intents using ML models.
    """

    def classify(self, text: str) -> Any:
        """Classify user intent.

        Args:
            text: User input text

        Returns:
            Classified intent (IntentType or similar)
        """
        ...

    def get_confidence(self, text: str, intent: Any) -> float:
        """Get confidence score for a specific intent.

        Args:
            text: User input text
            intent: Intent to check

        Returns:
            Confidence score (0-1)
        """
        ...


# =============================================================================
# Helper/Adapter Service Protocols
# =============================================================================


@runtime_checkable
class SystemPromptBuilderProtocol(Protocol):
    """Protocol for system prompt building service.

    Constructs system prompts from various components.
    """

    def build(
        self,
        base_prompt: str,
        tool_descriptions: Optional[str] = None,
        project_context: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Build system prompt from components.

        Args:
            base_prompt: Base system prompt
            tool_descriptions: Tool descriptions to include
            project_context: Project-specific context
            **kwargs: Additional prompt components

        Returns:
            Complete system prompt
        """
        ...


@runtime_checkable
class ParallelExecutorProtocol(Protocol):
    """Protocol for parallel tool execution service.

    Executes multiple tools in parallel.
    """

    async def execute_parallel(
        self,
        tool_calls: List[Any],
        **kwargs: Any,
    ) -> List[Any]:
        """Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool calls to execute
            **kwargs: Additional execution parameters

        Returns:
            List of tool results
        """
        ...


@runtime_checkable
class ResponseCompleterProtocol(Protocol):
    """Protocol for response completion service.

    Completes partial responses and handles tool failures.
    """

    async def complete_response(
        self,
        partial_response: str,
        context: Any,
        **kwargs: Any,
    ) -> str:
        """Complete a partial response.

        Args:
            partial_response: Partial response text
            context: Completion context
            **kwargs: Additional completion parameters

        Returns:
            Completed response
        """
        ...


@runtime_checkable
class StreamingHandlerProtocol(Protocol):
    """Protocol for streaming chat handler service.

    Handles streaming chat responses.
    """

    async def handle_stream(
        self,
        stream: AsyncIterator[Any],
        context: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Handle streaming chat response.

        Args:
            stream: Input stream
            context: Streaming context
            **kwargs: Additional handling parameters

        Yields:
            Processed stream chunks
        """
        ...


__all__ = [
    # Factory protocols
    "IAgentFactory",
    "IAgent",
    # Tool selection data classes
    "ToolSelectionContext",
    "ToolSelectorFeatures",
    # Provider protocols
    "ProviderManagerProtocol",
    # Tool protocols
    "ToolRegistryProtocol",
    "ToolSelectorProtocol",
    "ToolPipelineProtocol",
    "ToolExecutorProtocol",
    # Coordinator protocols
    "IToolSelectionCoordinator",
    # Conversation protocols
    "ConversationControllerProtocol",
    "ConversationStateMachineProtocol",
    "MessageHistoryProtocol",
    # Streaming protocols
    "StreamingToolChunk",
    "StreamingToolAdapterProtocol",
    "StreamingControllerProtocol",
    # Analysis protocols
    "TaskAnalyzerProtocol",
    "ComplexityClassifierProtocol",
    "ActionAuthorizerProtocol",
    "SearchRouterProtocol",
    # Observability protocols
    "ObservabilityProtocol",
    "MetricsCollectorProtocol",
    # Cache protocols
    "ToolCacheProtocol",
    # Task tracking protocols
    "TaskTrackerProtocol",
    # Output formatting protocols
    "ToolOutputFormatterProtocol",
    "ResponseSanitizerProtocol",
    # Utility protocols
    "ArgumentNormalizerProtocol",
    "ProjectContextProtocol",
    # Component lifecycle protocols
    "CodeExecutionManagerProtocol",
    "WorkflowRegistryProtocol",
    "ToolRegistrarProtocol",
    "UsageAnalyticsProtocol",
    "ToolSequenceTrackerProtocol",
    "ContextCompactorProtocol",
    # Mode controller protocols
    "ModeControllerProtocol",
    # Deduplication protocols
    "ToolDeduplicationTrackerProtocol",
    # Embedding store protocols
    "ConversationEmbeddingStoreProtocol",
    # Recovery protocols
    "RecoveryHandlerProtocol",
    "StreamingRecoveryCoordinatorProtocol",
    # Utility service protocols
    "DebugLoggerProtocol",
    "TaskTypeHinterProtocol",
    "ReminderManagerProtocol",
    "RLCoordinatorProtocol",
    "SafetyCheckerProtocol",
    "AutoCommitterProtocol",
    "MCPBridgeProtocol",
    # Infrastructure service protocols
    "ToolDependencyGraphProtocol",
    "ToolPluginRegistryProtocol",
    "SemanticToolSelectorProtocol",
    "ProviderRegistryProtocol",
    # Analytics & observability protocols
    "UsageLoggerProtocol",
    "StreamingMetricsCollectorProtocol",
    "IntentClassifierProtocol",
    # Helper/adapter service protocols
    "SystemPromptBuilderProtocol",
    "ParallelExecutorProtocol",
    "ResponseCompleterProtocol",
    "StreamingHandlerProtocol",
    # Tool access control protocols
    "AccessPrecedence",
    "ToolAccessDecision",
    "ToolAccessContext",
    "IToolAccessController",
    # Budget management protocols
    "BudgetType",
    "BudgetStatus",
    "BudgetConfig",
    "IBudgetManager",
    # New coordinator protocols (WS-D)
    "ToolCoordinatorProtocol",
    "StateCoordinatorProtocol",
    "PromptCoordinatorProtocol",
    # Vertical storage protocol (DIP compliance)
    "VerticalStorageProtocol",
]


# =============================================================================
# Tool Access Control Protocols
# =============================================================================

from enum import Enum


class AccessPrecedence(int, Enum):
    """Precedence levels for tool access control.

    Lower numbers = higher precedence.
    Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)
    """

    SAFETY = 0  # DangerLevel checks - highest precedence
    MODE = 1  # Mode restrictions (BUILD/PLAN/EXPLORE)
    SESSION = 2  # Session-enabled tools set
    VERTICAL = 3  # TieredToolConfig from vertical
    STAGE = 4  # Conversation stage filtering
    INTENT = 5  # DISPLAY_ONLY/READ_ONLY blocking


@dataclass
class ToolAccessDecision:
    """Result of a tool access check.

    Provides detailed information about why a tool was allowed or denied,
    and which layer made the decision.

    Attributes:
        allowed: Whether the tool is allowed
        tool_name: Name of the tool checked
        reason: Human-readable explanation
        source: Which layer made the decision
        precedence_level: Numeric precedence (lower = higher priority)
        checked_layers: Names of layers that were checked
        layer_results: Results from each checked layer
    """

    allowed: bool
    tool_name: str
    reason: str
    source: str  # Which layer decided (e.g., "mode", "safety")
    precedence_level: int = 0
    checked_layers: List[str] = field(default_factory=list)
    layer_results: Dict[str, bool] = field(default_factory=dict)

    def explain(self) -> str:
        """Get detailed explanation of the decision."""
        status = "ALLOWED" if self.allowed else "DENIED"
        layers = ", ".join(self.checked_layers) if self.checked_layers else "none"
        return (
            f"Tool '{self.tool_name}' is {status}\n"
            f"  Decision by: {self.source} (precedence {self.precedence_level})\n"
            f"  Reason: {self.reason}\n"
            f"  Layers checked: {layers}"
        )


@dataclass
class ToolAccessContext:
    """Context for tool access decisions.

    Provides all information needed by access layers to make decisions.

    Attributes:
        user_message: Current user message (for intent detection)
        conversation_stage: Current conversation stage
        intent: Detected user intent
        current_mode: Current agent mode (BUILD/PLAN/EXPLORE)
        session_enabled_tools: Tools enabled for this session
        vertical_name: Active vertical (if any)
        metadata: Additional context data
    """

    user_message: Optional[str] = None
    conversation_stage: Optional["ConversationStage"] = None
    intent: Optional[Any] = None  # ActionIntent
    current_mode: Optional[str] = None
    session_enabled_tools: Optional[Set[str]] = None
    vertical_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IToolAccessController(Protocol):
    """Protocol for unified tool access control.

    Provides a single point of control for all tool access decisions,
    replacing scattered checks throughout the codebase.

    The controller applies layers in precedence order:
    Safety (L0) > Mode (L1) > Session (L2) > Vertical (L3) > Stage (L4) > Intent (L5)

    A tool is blocked if ANY layer denies it. The first layer to deny
    becomes the authoritative source of the decision.
    """

    def check_access(
        self, tool_name: str, context: Optional[ToolAccessContext] = None
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
        self, tools: List[str], context: Optional[ToolAccessContext] = None
    ) -> Tuple[List[str], List[ToolAccessDecision]]:
        """Filter a list of tools to only allowed ones.

        Args:
            tools: List of tool names to filter
            context: Access context

        Returns:
            Tuple of (allowed_tools, denial_decisions)
        """
        ...

    def get_allowed_tools(self, context: Optional[ToolAccessContext] = None) -> Set[str]:
        """Get all tools allowed in the given context.

        Args:
            context: Access context

        Returns:
            Set of allowed tool names
        """
        ...

    def explain_decision(self, tool_name: str, context: Optional[ToolAccessContext] = None) -> str:
        """Get detailed explanation for a tool access decision.

        Args:
            tool_name: Name of the tool
            context: Access context

        Returns:
            Human-readable explanation
        """
        ...


# =============================================================================
# Budget Management Protocols
# =============================================================================


class BudgetType(str, Enum):
    """Types of budgets tracked by the budget manager.

    Attributes:
        TOOL_CALLS: Total tool calls allowed per session
        ITERATIONS: Total LLM iterations allowed
        EXPLORATION: Read/search operations (counted toward exploration limit)
        ACTION: Write/modify operations (separate from exploration)
    """

    TOOL_CALLS = "tool_calls"
    ITERATIONS = "iterations"
    EXPLORATION = "exploration"
    ACTION = "action"


@dataclass
class BudgetStatus:
    """Status of a specific budget.

    Attributes:
        budget_type: Type of budget
        current: Current usage count
        base_maximum: Base maximum before multipliers
        effective_maximum: Maximum after multipliers applied
        is_exhausted: Whether budget is fully consumed
        model_multiplier: Model-specific multiplier
        mode_multiplier: Mode-specific multiplier
        productivity_multiplier: Productivity-based multiplier
    """

    budget_type: BudgetType
    current: int = 0
    base_maximum: int = 0
    effective_maximum: int = 0
    is_exhausted: bool = False
    model_multiplier: float = 1.0
    mode_multiplier: float = 1.0
    productivity_multiplier: float = 1.0

    @property
    def remaining(self) -> int:
        """Get remaining budget."""
        return max(0, self.effective_maximum - self.current)

    @property
    def utilization(self) -> float:
        """Get budget utilization as a percentage (0.0-1.0)."""
        if self.effective_maximum == 0:
            return 0.0
        return min(1.0, self.current / self.effective_maximum)


@dataclass
class BudgetConfig:
    """Configuration for budget manager.

    Attributes:
        base_tool_calls: Base tool call budget
        base_iterations: Base iteration budget
        base_exploration: Base exploration iterations
        base_action: Base action iterations
    """

    base_tool_calls: int = 30
    base_iterations: int = 50
    base_exploration: int = 8
    base_action: int = 12


@runtime_checkable
class IBudgetManager(Protocol):
    """Protocol for unified budget management.

    Centralizes all budget tracking with consistent multiplier composition:
    effective_max = base × model_multiplier × mode_multiplier × productivity_multiplier

    Replaces scattered budget tracking in:
    - unified_task_tracker.py: exploration_iterations, action_iterations
    - orchestrator.py: tool_budget, complexity_tool_budget
    - intelligent_prompt_builder.py: recommended_tool_budget
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

        Model multipliers vary by model capability:
        - GPT-4o: 1.0 (baseline)
        - Claude Opus: 1.2 (more capable)
        - DeepSeek: 1.3 (needs more exploration)
        - Ollama local: 1.5 (needs more attempts)

        Args:
            multiplier: Model multiplier value
        """
        ...

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set the mode-specific multiplier.

        Mode multipliers:
        - BUILD: 2.0 (reading before writing)
        - PLAN: 2.5 (thorough analysis)
        - EXPLORE: 3.0 (exploration is primary goal)

        Args:
            multiplier: Mode multiplier value
        """
        ...

    def set_productivity_multiplier(self, multiplier: float) -> None:
        """Set the productivity multiplier.

        Productivity multipliers (from RL learning):
        - High productivity session: 0.8 (less budget needed)
        - Normal: 1.0
        - Low productivity: 1.2-2.0 (more attempts needed)

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

    def record_tool_call(self, tool_name: str, is_write_operation: bool = False) -> bool:
        """Record a tool call and consume appropriate budget.

        Automatically routes to EXPLORATION or ACTION budget based
        on whether the operation is a write operation.

        Args:
            tool_name: Name of the tool called
            is_write_operation: Whether this is a write/modify operation

        Returns:
            True if budget was available
        """
        ...


# =============================================================================
# Unified Memory Protocol
# =============================================================================


@runtime_checkable
class UnifiedMemoryCoordinatorProtocol(Protocol):
    """Protocol for unified memory coordinator.

    Provides federated search across multiple memory backends (entity,
    conversation, graph, embeddings) with pluggable ranking strategies.
    """

    async def search_all(
        self,
        query: str,
        limit: int = 20,
        memory_types: Optional[List[Any]] = None,
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.0,
    ) -> List[Any]:
        """Search across all registered memory providers.

        Args:
            query: Search query string
            limit: Maximum results to return
            memory_types: Optional filter for specific memory types
            session_id: Optional session ID for context
            filters: Additional provider-specific filters
            min_relevance: Minimum relevance threshold

        Returns:
            Ranked list of memory results from all providers
        """
        ...

    async def search_type(
        self,
        memory_type: Any,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> List[Any]:
        """Search a specific memory type.

        Args:
            memory_type: Type of memory to search
            query: Search query string
            limit: Maximum results to return
            **kwargs: Additional search parameters

        Returns:
            List of memory results from the specified type
        """
        ...

    async def store(
        self,
        memory_type: Any,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a value in a specific memory type.

        Args:
            memory_type: Type of memory to store in
            key: Storage key
            value: Value to store
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        ...

    async def get(
        self,
        memory_type: Any,
        key: str,
    ) -> Optional[Any]:
        """Get a value from a specific memory type.

        Args:
            memory_type: Type of memory to retrieve from
            key: Storage key

        Returns:
            Memory result or None if not found
        """
        ...

    def register_provider(self, provider: Any) -> None:
        """Register a memory provider.

        Args:
            provider: Provider implementing MemoryProviderProtocol
        """
        ...

    def unregister_provider(self, memory_type: Any) -> bool:
        """Unregister a memory provider.

        Args:
            memory_type: Type of memory provider to remove

        Returns:
            True if provider was removed
        """
        ...

    def get_registered_types(self) -> List[Any]:
        """Get list of registered memory types.

        Returns:
            List of registered MemoryType values
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics.

        Returns:
            Dictionary with query counts, errors, registered providers
        """
        ...


# =============================================================================
# New Coordinator Protocols (WS-D: Orchestrator SOLID Fixes)
# =============================================================================


@runtime_checkable
class ToolCoordinatorProtocol(Protocol):
    """Protocol for tool coordination operations.

    Coordinates tool selection, budgeting, and execution through a unified
    interface. Consolidates tool-related operations from AgentOrchestrator.
    """

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


@runtime_checkable
class StateCoordinatorProtocol(Protocol):
    """Protocol for state coordination operations.

    Coordinates conversation state and stage transitions through a unified
    interface. Consolidates state management from AgentOrchestrator.
    """

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


@runtime_checkable
class PromptCoordinatorProtocol(Protocol):
    """Protocol for prompt coordination operations.

    Coordinates system prompt assembly through a unified interface.
    Consolidates prompt building from AgentOrchestrator.
    """

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


# =============================================================================
# Vertical Storage Protocol
# =============================================================================


@runtime_checkable
class VerticalStorageProtocol(Protocol):
    """Protocol for storing vertical-specific data in orchestrator.

    This protocol addresses the DIP (Dependency Inversion Principle) violation
    where FrameworkStepHandler uses private attribute fallbacks on the orchestrator
    (e.g., _vertical_middleware_storage, _safety_patterns, _team_specs).

    By defining this protocol, consumers can depend on an abstraction rather than
    concrete implementation details, enabling:
    - Proper dependency injection
    - Easy testing via mock substitution
    - Clear contracts for vertical data storage

    Usage:
        from victor.agent.protocols import VerticalStorageProtocol

        def configure_vertical(storage: VerticalStorageProtocol) -> None:
            storage.set_middleware(middleware_list)
            storage.set_safety_patterns(patterns)
            storage.set_team_specs(specs)

        # Later retrieval
        middleware = storage.get_middleware()
    """

    def set_middleware(self, middleware: List[Any]) -> None:
        """Store middleware configuration.

        Args:
            middleware: List of MiddlewareProtocol implementations
        """
        ...

    def get_middleware(self) -> List[Any]:
        """Retrieve middleware configuration.

        Returns:
            List of middleware instances, or empty list if not set
        """
        ...

    def set_safety_patterns(self, patterns: List[Any]) -> None:
        """Store safety patterns.

        Args:
            patterns: List of SafetyPattern instances from vertical extensions
        """
        ...

    def get_safety_patterns(self) -> List[Any]:
        """Retrieve safety patterns.

        Returns:
            List of safety pattern instances, or empty list if not set
        """
        ...

    def set_team_specs(self, specs: Dict[str, Any]) -> None:
        """Store team specifications.

        Args:
            specs: Dictionary mapping team names to TeamSpec instances
        """
        ...

    def get_team_specs(self) -> Dict[str, Any]:
        """Retrieve team specifications.

        Returns:
            Dictionary of team specs, or empty dict if not set
        """
        ...


# =============================================================================
# Manager/Coordinator Refactoring Protocols (SOLID-based)
# =============================================================================


class IProviderHealthMonitor(Protocol):
    """Protocol for provider health monitoring.

    Defines interface for monitoring provider health and triggering fallbacks.
    Separated from IProviderSwitcher to follow ISP.
    """

    async def check_health(self, provider: Any) -> bool:
        """Check if provider is healthy.

        Args:
            provider: Provider instance to check

        Returns:
            True if provider is healthy, False otherwise
        """
        ...

    async def start_health_checks(
        self,
        interval: Optional[float] = None,
        provider: Optional[Any] = None,
        provider_name: Optional[str] = None,
    ) -> None:
        """Start periodic health checks.

        Args:
            interval: Interval between health checks in seconds
            provider: Provider to monitor (optional)
            provider_name: Provider name (optional)
        """
        ...

    async def stop_health_checks(self) -> None:
        """Stop health checks."""
        ...

    def is_monitoring(self) -> bool:
        """Check if health monitoring is currently active.

        Returns:
            True if monitoring is active, False otherwise
        """
        ...


class IProviderSwitcher(Protocol):
    """Protocol for provider switching operations.

    Defines interface for switching between providers and models.
    Separated from IProviderHealthMonitor to follow ISP.
    """

    def get_current_provider(self) -> Optional[Any]:
        """Get current provider instance.

        Returns:
            Current provider or None if not configured
        """
        ...

    def get_current_model(self) -> str:
        """Get current model name.

        Returns:
            Current model name or empty string if not configured
        """
        ...

    def get_current_state(self) -> Optional[Any]:
        """Get current switcher state.

        Returns:
            Current state or None if not configured
        """
        ...

    def set_initial_state(
        self,
        provider: Any,
        provider_name: str,
        model: str,
    ) -> None:
        """Set initial provider state (used during initialization).

        Args:
            provider: Provider instance
            provider_name: Provider name
            model: Model name
        """
        ...

    async def switch_provider(
        self,
        provider_name: str,
        model: str,
        reason: str = "manual",
        settings: Optional[Any] = None,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider/model.

        Args:
            provider_name: Name of provider to switch to
            model: Model identifier
            reason: Reason for switch (default "manual")
            settings: Optional settings for provider configuration
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch succeeded, False otherwise
        """
        ...

    async def switch_model(self, model: str, reason: str = "manual") -> bool:
        """Switch to a different model on current provider.

        Args:
            model: Model identifier
            reason: Reason for the switch

        Returns:
            True if switch succeeded, False otherwise
        """
        ...

    def get_switch_history(self) -> List[Dict[str, Any]]:
        """Get history of provider switches.

        Returns:
            List of switch event dictionaries
        """
        ...


class IToolAdapterCoordinator(Protocol):
    """Protocol for tool adapter coordination.

    Defines interface for initializing and managing tool adapters.
    Separated to allow independent testing and mocking.
    """

    def initialize_adapter(self) -> Any:
        """Initialize tool adapter for current provider.

        Returns:
            ToolCallingCapabilities instance

        Raises:
            ValueError: If no provider is configured
        """
        ...

    def get_capabilities(self) -> Any:
        """Get tool calling capabilities.

        Returns:
            ToolCallingCapabilities instance

        Raises:
            ValueError: If adapter not initialized
        """
        ...

    def get_adapter(self) -> Any:
        """Get current tool adapter instance.

        Returns:
            Tool adapter instance

        Raises:
            ValueError: If adapter not initialized
        """
        ...

    def is_initialized(self) -> bool:
        """Check if adapter has been initialized.

        Returns:
            True if adapter is initialized, False otherwise
        """
        ...


class IProviderEventEmitter(Protocol):
    """Protocol for provider-related events.

    Defines interface for emitting and handling provider events.
    Separated to support different event implementations.
    """

    def emit_switch_event(self, event: Dict[str, Any]) -> None:
        """Emit provider switch event.

        Args:
            event: Event dictionary with switch details
        """
        ...

    def on_switch(self, callback: Any) -> None:
        """Register callback for provider switches.

        Args:
            callback: Callable to invoke on switch
        """
        ...


class IProviderClassificationStrategy(Protocol):
    """Protocol for provider classification.

    Defines interface for classifying providers by type.
    Supports Open/Closed Principle via strategy pattern.
    """

    def is_cloud_provider(self, provider_name: str) -> bool:
        """Check if provider is cloud-based.

        Args:
            provider_name: Name of the provider

        Returns:
            True if cloud provider, False otherwise
        """
        ...

    def is_local_provider(self, provider_name: str) -> bool:
        """Check if provider is local.

        Args:
            provider_name: Name of the provider

        Returns:
            True if local provider, False otherwise
        """
        ...

    def get_provider_type(self, provider_name: str) -> str:
        """Get provider type category.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider type ("cloud", "local", "unknown")
        """
        ...


class IMessageStore(Protocol):
    """Protocol for message storage and retrieval.

    Defines interface for persisting and retrieving messages.
    Separated from other conversation concerns to follow ISP.
    """

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Add a message to storage.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
            **metadata: Additional message metadata
        """
        ...

    def get_messages(self, limit: Optional[int] = None) -> List[Any]:
        """Retrieve messages.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        ...

    def persist(self) -> bool:
        """Persist messages to storage.

        Returns:
            True if persistence succeeded, False otherwise
        """
        ...


class IContextOverflowHandler(Protocol):
    """Protocol for context overflow handling.

    Defines interface for detecting and handling context overflow.
    Separated from IMessageStore to follow ISP.
    """

    def check_overflow(self) -> bool:
        """Check if context has overflowed.

        Returns:
            True if overflow detected, False otherwise
        """
        ...

    def handle_compaction(self) -> Optional[Any]:
        """Handle context compaction.

        Returns:
            Compaction result or None
        """
        ...


class ISessionManager(Protocol):
    """Protocol for session lifecycle management.

    Defines interface for creating and managing sessions.
    Separated to support different session backends.
    """

    def create_session(self) -> str:
        """Create a new session.

        Returns:
            Session ID
        """
        ...

    def recover_session(self, session_id: str) -> bool:
        """Recover an existing session.

        Args:
            session_id: Session ID to recover

        Returns:
            True if recovery succeeded, False otherwise
        """
        ...

    def persist_session(self) -> bool:
        """Persist session state.

        Returns:
            True if persistence succeeded, False otherwise
        """
        ...


class IEmbeddingManager(Protocol):
    """Protocol for embedding and semantic search.

    Defines interface for semantic search over conversations.
    Separated because not all conversations need embeddings.
    """

    def initialize_embeddings(self) -> None:
        """Initialize embedding store."""
        ...

    def semantic_search(self, query: str, k: int = 5) -> List[Any]:
        """Perform semantic search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of search results
        """
        ...


class IBudgetTracker(Protocol):
    """Protocol for budget tracking.

    Defines interface for tracking and consuming budget.
    Core budget functionality, separated from other concerns.
    """

    def consume(self, budget_type: Any, amount: int) -> bool:
        """Consume from budget.

        Args:
            budget_type: Type of budget to consume from
            amount: Amount to consume

        Returns:
            True if consumption succeeded, False if exhausted
        """
        ...

    def get_status(self, budget_type: Any) -> Any:
        """Get current budget status.

        Args:
            budget_type: Type of budget to query

        Returns:
            BudgetStatus instance
        """
        ...

    def reset(self) -> None:
        """Reset all budgets."""
        ...


class IMultiplierCalculator(Protocol):
    """Protocol for budget multiplier calculation.

    Defines interface for calculating effective budget with multipliers.
    Separated from IBudgetTracker to follow ISP.
    """

    def calculate_effective_max(self, base_max: int) -> int:
        """Calculate effective maximum with multipliers.

        Args:
            base_max: Base maximum budget

        Returns:
            Effective maximum after applying multipliers
        """
        ...

    def set_model_multiplier(self, multiplier: float) -> None:
        """Set model-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-1.5)
        """
        ...

    def set_mode_multiplier(self, multiplier: float) -> None:
        """Set mode-specific multiplier.

        Args:
            multiplier: Multiplier value (e.g., 1.0-3.0)
        """
        ...


class IModeCompletionChecker(Protocol):
    """Protocol for mode completion detection.

    Defines interface for checking if mode should complete early.
    Separated from budget tracking to follow ISP.
    """

    def should_early_exit(self, mode: str, response: str) -> Tuple[bool, str]:
        """Check if should exit mode early.

        Args:
            mode: Current mode
            response: Response to check

        Returns:
            Tuple of (should_exit, reason)
        """
        ...


class IToolCallClassifier(Protocol):
    """Protocol for classifying tool calls.

    Defines interface for classifying tools by operation type.
    Supports Open/Closed Principle via strategy pattern.
    """

    def is_write_operation(self, tool_name: str) -> bool:
        """Check if tool is a write operation.

        Args:
            tool_name: Name of the tool

        Returns:
            True if write operation, False otherwise
        """
        ...

    def classify_operation(self, tool_name: str) -> str:
        """Classify tool operation type.

        Args:
            tool_name: Name of the tool

        Returns:
            Operation type category
        """
        ...

    def add_write_tool(self, tool_name: str) -> None:
        """Add a tool to the write operation classification.

        Args:
            tool_name: Name of the tool to add
        """
        ...
