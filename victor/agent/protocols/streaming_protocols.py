"""Protocol definitions for streaming protocols."""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    Tuple,
    runtime_checkable,
)

from dataclasses import dataclass, field


__all__ = [
    "StreamingToolChunk",
    "StreamingToolAdapterProtocol",
    "StreamingControllerProtocol",
    "StreamingRecoveryCoordinatorProtocol",
    "ChunkGeneratorProtocol",
    "StreamingHandlerProtocol",
    "StreamingMetricsCollectorProtocol",
]

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

