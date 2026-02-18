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

"""Protocol definitions for ChatCoordinator dependency injection.

Defines focused sub-protocols that formalize the implicit contract between
ChatCoordinator and AgentOrchestrator. This enables:

- Unit testing ChatCoordinator with lightweight mocks instead of full orchestrator
- Clear documentation of which orchestrator capabilities ChatCoordinator uses
- Compile-time detection of interface drift via structural typing

Design:
    3 sub-protocols group ~47 attributes by responsibility:
    - ChatContextProtocol: conversation, messages, settings, context management
    - ToolContextProtocol: tool selection, execution, budget
    - ProviderContextProtocol: LLM provider, model params, streaming state

    ChatOrchestratorProtocol composes all three plus remaining component
    accessors (task, recovery, presentation, session state).

    All protocols use ``Any`` for complex types (same pattern as
    ``core/protocols.py`` and ``agent/protocols.py``) to avoid importing
    30+ concrete classes.

Phase 6 of DIP Hardening (docs/architecture-analysis-phase3.md:186).
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

# =============================================================================
# Sub-Protocol: Chat Context
# =============================================================================


class ChatContextProtocol(Protocol):
    """Conversation/message management, settings, and context compaction.

    Groups attributes ChatCoordinator uses for managing conversation state,
    message history, system prompts, and context window management.
    """

    # -- Conversation & messages --
    conversation: Any
    """ConversationManager instance for system prompt and message count."""

    messages: Any
    """List[Message] - current conversation message history."""

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        ...

    _system_added: bool
    """Flag indicating whether system prompt has been added."""

    # -- Controllers --
    conversation_controller: Any
    """ConversationController for history compaction and intent guards."""

    _conversation_controller: Any
    """Internal ConversationController reference."""

    _context_compactor: Any
    """Optional ContextCompactor for automatic context window management."""

    _context_manager: Any
    """Optional ContextManager for background compaction."""

    # -- Settings & session --
    settings: Any
    """VictorSettings instance."""

    _session_state: Any
    """SessionStateManager for per-turn state reset."""

    # Note: _current_stream_context is omitted — it is lazily set by
    # ChatCoordinator during stream_chat, not a constructor-time dependency.

    # -- Context management methods --
    def _check_context_overflow(self, max_context: int) -> bool:
        """Check if conversation exceeds context window limit."""
        ...

    def _get_max_context_chars(self) -> int:
        """Get maximum context length in characters for current provider/model."""
        ...

    def _get_thinking_disabled_prompt(self, prompt: str) -> str:
        """Wrap prompt to disable thinking for completion requests."""
        ...

    # -- Token tracking --
    _cumulative_token_usage: Dict[str, int]
    """Accumulated token usage across stream iterations."""


# =============================================================================
# Sub-Protocol: Tool Context
# =============================================================================


class ToolContextProtocol(Protocol):
    """Tool selection, execution, and budget management.

    Groups attributes ChatCoordinator uses for selecting tools,
    executing tool calls, and tracking tool budget consumption.
    """

    # -- Tool components --
    tool_selector: Any
    """ToolSelector for semantic/stage-based tool selection."""

    tool_adapter: Any
    """ToolCallAdapter for normalizing tool call formats."""

    _tool_coordinator: Any
    """ToolCoordinator for parse_and_validate_tool_calls."""

    _tool_planner: Any
    """ToolPlanner for goal-based tool planning and intent filtering."""

    # -- Budget & tracking --
    tool_budget: int
    """Maximum number of tool calls allowed for this conversation."""

    tool_calls_used: int
    """Number of tool calls consumed so far."""

    use_semantic_selection: bool
    """Whether to use semantic (embedding-based) tool selection."""

    # -- File tracking --
    observed_files: Any
    """Set of file paths observed during tool execution."""

    # Note: _current_intent is omitted — it is lazily set by ChatCoordinator
    # during stream_chat via task_coordinator, not a constructor-time dependency.

    # -- Tool execution --
    async def _handle_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        ...

    def _model_supports_tool_calls(self) -> bool:
        """Check if current model supports native tool calling."""
        ...


# =============================================================================
# Sub-Protocol: Provider Context
# =============================================================================


class ProviderContextProtocol(Protocol):
    """LLM provider access, model parameters, and streaming state.

    Groups attributes ChatCoordinator uses for making provider API calls
    and managing cancellation/streaming state.
    """

    # -- Provider & model --
    provider: Any
    """LLM provider instance (supports chat/stream/supports_tools)."""

    model: str
    """Current model identifier."""

    temperature: float
    """Sampling temperature for LLM calls."""

    max_tokens: int
    """Maximum tokens for LLM responses."""

    thinking: Any
    """Optional thinking configuration for extended reasoning."""

    # -- Provider coordination --
    _provider_coordinator: Any
    """ProviderCoordinator for rate limit handling."""

    # -- Streaming & cancellation --
    _cancel_event: Any
    """asyncio.Event for stream cancellation."""

    _is_streaming: bool
    """Whether a stream is currently active."""

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested."""
        ...


# =============================================================================
# Composite Protocol: ChatOrchestratorProtocol
# =============================================================================


@runtime_checkable
class ChatOrchestratorProtocol(
    ChatContextProtocol,
    ToolContextProtocol,
    ProviderContextProtocol,
    Protocol,
):
    """Complete protocol for ChatCoordinator's orchestrator dependency.

    Composes the three sub-protocols plus remaining component accessors
    that don't warrant their own protocol (ChatCoordinator is their only
    consumer).

    AgentOrchestrator satisfies this protocol structurally — no changes
    to its class hierarchy are needed.
    """

    # -- Task classification & coordination --
    task_classifier: Any
    """TaskClassifier for complexity-based budgeting."""

    task_coordinator: Any
    """TaskCoordinator for task preparation and intent guards."""

    _task_analyzer: Any
    """TaskAnalyzer for keyword-based task classification."""

    unified_tracker: Any
    """UnifiedTaskTracker for progress, loop detection, and metrics."""

    _task_completion_detector: Any
    """Optional TaskCompletionDetector for active completion signals."""

    # -- Recovery --
    _recovery_coordinator: Any
    """RecoveryCoordinator for empty response and natural completion handling."""

    _recovery_integration: Any
    """RecoveryIntegration for outcome recording."""

    async def _create_recovery_context(self, stream_ctx: Any) -> Any:
        """Create StreamingRecoveryContext from current state."""
        ...

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: Any,
        full_content: str,
        tool_calls: Any,
        mentioned_tools: Optional[List[str]] = None,
    ) -> Any:
        """Handle recovery using the recovery integration system."""
        ...

    def _apply_recovery_action(self, recovery_action: Any, stream_ctx: Any) -> Any:
        """Apply a recovery action and return optional StreamChunk."""
        ...

    def _record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> None:
        """Record outcome for intelligent pipeline metrics."""
        ...

    async def _validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate response quality using intelligent pipeline."""
        ...

    async def _prepare_intelligent_request(
        self,
        task: str,
        task_type: str,
    ) -> Any:
        """Prepare intelligent pipeline context before provider call."""
        ...

    # -- Presentation & output --
    sanitizer: Any
    """ResponseSanitizer for cleaning model output."""

    response_completer: Any
    """ResponseCompleter for ensuring non-empty responses."""

    debug_logger: Any
    """DebugLogger for iteration and limit logging."""

    reminder_manager: Any
    """ContextReminderManager for periodic context reminders."""

    _chunk_generator: Any
    """ChunkGenerator for creating StreamChunk objects."""

    _presentation: Any
    """PresentationManager for icons and formatting."""

    _metrics_collector: Any
    """MetricsCollector for stream performance metrics."""

    _streaming_handler: Any
    """StreamingHandler for loop warning handling."""

    # -- Session state --
    _required_files: List[str]
    """File paths extracted from user prompt for task completion tracking."""

    _required_outputs: List[str]
    """Required outputs extracted from user prompt."""

    _read_files_session: Any
    """Set of files read during this session."""

    _all_files_read_nudge_sent: bool
    """Whether the 'all files read' nudge has been sent."""

    # Note: _force_finalize is omitted — it is lazily set by ChatCoordinator
    # during stream_chat, not a constructor-time dependency.

    _usage_analytics: Any
    """Optional UsageAnalytics for session-level tracking."""

    _sequence_tracker: Any
    """Optional ToolSequenceTracker for tool sequence analysis."""
