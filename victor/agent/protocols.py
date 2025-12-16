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
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.agent.conversation_state import ConversationStage
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.tools.base import CostTier


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


@runtime_checkable
class ToolSelectorProtocol(Protocol):
    """Protocol for tool selection.

    Selects appropriate tools based on user prompt and context.
    """

    def select_tools(
        self,
        prompt: str,
        max_tools: int,
        stage: Optional["ConversationStage"] = None,
    ) -> List[str]:
        """Select tools for a given prompt.

        Args:
            prompt: User prompt
            max_tools: Maximum tools to return
            stage: Current conversation stage

        Returns:
            List of selected tool names
        """
        ...

    def get_tools_for_stage(self, stage: "ConversationStage") -> Set[str]:
        """Get recommended tools for a conversation stage."""
        ...


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
    """Protocol for tool execution.

    Handles individual tool execution with validation and retry.
    """

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: Optional[str] = None,
    ) -> Any:
        """Execute a single tool.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            tool_call_id: Optional identifier for the call

        Returns:
            Tool result
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
        session_time_limit: float,
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
        session_time_limit: float,
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


__all__ = [
    # Provider protocols
    "ProviderManagerProtocol",
    # Tool protocols
    "ToolRegistryProtocol",
    "ToolSelectorProtocol",
    "ToolPipelineProtocol",
    "ToolExecutorProtocol",
    # Conversation protocols
    "ConversationControllerProtocol",
    "ConversationStateMachineProtocol",
    "MessageHistoryProtocol",
    # Streaming protocols
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
    # Recovery protocols
    "RecoveryHandlerProtocol",
]
