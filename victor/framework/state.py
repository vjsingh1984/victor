"""State - Observable conversation state.

This module provides State and StateHooks for observing and
reacting to conversation state changes.

Phase 7.2 Refactoring:
- Uses protocol methods from OrchestratorProtocol when available
- Falls back gracefully for backward compatibility
- Eliminates most hasattr/getattr duck-typing

Stage Unification (Post-Phase 9):
- Stage is now an alias to ConversationStage from conversation_state.py
- ConversationStage is the canonical source for conversation stages
- This eliminates duplicate enum definitions

Example:
    # Observe current state
    print(f"Stage: {agent.state.stage}")
    print(f"Tools used: {agent.state.tool_calls_used}")

    # React to state changes
    agent.on_state_change(lambda old, new: print(f"{old} -> {new}"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Union

# Import canonical ConversationStage from conversation_state.py (single source of truth)
from victor.agent.conversation_state import ConversationStage

if TYPE_CHECKING:
    from victor.framework.protocols import OrchestratorProtocol


# Stage is an alias to ConversationStage for framework API compatibility
# This maintains backward compatibility while eliminating duplicate enums
Stage = ConversationStage
"""Conversation stage in the agent's workflow.

This is an alias to `victor.agent.conversation_state.ConversationStage`.

Stages represent different phases of how the agent
approaches a task. They are detected automatically
based on tool usage and conversation patterns.

The typical flow is:
INITIAL -> PLANNING -> READING -> ANALYSIS -> EXECUTION -> VERIFICATION -> COMPLETION

Values:
    INITIAL: Initial state - conversation just started.
    PLANNING: Agent is planning its approach.
    READING: Agent is reading files and gathering information.
    ANALYSIS: Agent is analyzing code or information.
    EXECUTION: Agent is executing changes or running commands.
    VERIFICATION: Agent is verifying its changes work correctly.
    COMPLETION: Task has been completed.
"""


# Type alias for state change observers
StateObserver = Callable[["State", "State"], None]


@dataclass
class StateHooks:
    """Hook callbacks for state machine transitions.

    Use these hooks to react to state changes in real-time.
    Hooks are invoked synchronously during state transitions.

    Attributes:
        on_enter: Called when entering a new stage
        on_exit: Called when exiting a stage
        on_transition: Called for any stage transition

    Example:
        hooks = StateHooks(
            on_enter=lambda stage, ctx: print(f"Entering {stage}"),
            on_exit=lambda stage, ctx: print(f"Exiting {stage}"),
            on_transition=lambda old, new, ctx: print(f"{old} -> {new}")
        )
    """

    on_enter: Optional[Callable[[str, Dict[str, Any]], None]] = None
    """Called when entering a stage. Args: (stage_name, context)"""

    on_exit: Optional[Callable[[str, Dict[str, Any]], None]] = None
    """Called when exiting a stage. Args: (stage_name, context)"""

    on_transition: Optional[Callable[[str, str, Dict[str, Any]], None]] = None
    """Called on any transition. Args: (old_stage, new_stage, context)"""


def _has_protocol_method(obj: Any, method_name: str) -> bool:
    """Check if object has a callable protocol method.

    Args:
        obj: Object to check
        method_name: Name of method to look for

    Returns:
        True if method exists and is callable
    """
    attr = getattr(obj, method_name, None)
    return attr is not None and callable(attr)


class State:
    """Observable agent state wrapper.

    Provides a simplified view of the agent's current state,
    wrapping the internal orchestrator via OrchestratorProtocol.

    This class is read-only from the user's perspective -
    state changes are managed internally by the agent.

    Protocol Integration (Phase 7.2):
        This class now prefers protocol methods when available:
        - get_stage() -> stage property
        - get_tool_calls_count() -> tool_calls_used property
        - get_tool_budget() -> tool_budget property
        - get_observed_files() -> files_observed property
        - get_modified_files() -> files_modified property
        - get_iteration_count() -> iteration_count property
        - get_max_iterations() -> max_iterations property
        - get_message_count() -> message_count property
        - is_streaming() -> is_streaming property
        - current_provider -> provider property
        - current_model -> model property

    Attributes:
        stage: Current conversation stage
        tool_calls_used: Number of tool calls made
        tool_budget: Maximum allowed tool calls
        tools_remaining: Remaining tool calls
        files_observed: Files that have been read
        files_modified: Files that have been changed
        message_count: Messages in conversation
        is_streaming: Whether agent is currently streaming
        provider: Current LLM provider name
        model: Current model name

    Example:
        state = agent.state
        print(f"Stage: {state.stage}")
        print(f"Progress: {state.tool_calls_used}/{state.tool_budget}")
        print(f"Files touched: {state.files_observed | state.files_modified}")
    """

    def __init__(self, orchestrator: Union["OrchestratorProtocol", Any]) -> None:
        """Initialize State wrapper.

        Args:
            orchestrator: AgentOrchestrator or OrchestratorProtocol instance
        """
        self._orchestrator = orchestrator

    @property
    def stage(self) -> Stage:
        """Get current conversation stage.

        Uses protocol method get_stage() when available.
        Since Stage is now an alias to ConversationStage,
        we can return the internal stage directly.

        Returns:
            Current Stage enum value
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_stage"):
            try:
                # Stage is now same type as ConversationStage, return directly
                return self._orchestrator.get_stage()
            except AttributeError:
                pass
        return Stage.INITIAL

    @property
    def tool_calls_used(self) -> int:
        """Get number of tool calls made in this session.

        Uses protocol method get_tool_calls_count() when available.

        Returns:
            Count of tool calls
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_tool_calls_count"):
            return self._orchestrator.get_tool_calls_count()
        return 0

    @property
    def tool_budget(self) -> int:
        """Get total tool call budget.

        Uses protocol method get_tool_budget() when available.

        Returns:
            Maximum allowed tool calls
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_tool_budget"):
            return self._orchestrator.get_tool_budget()
        return 50  # Default budget

    @property
    def tools_remaining(self) -> int:
        """Get remaining tool calls.

        Returns:
            Number of tool calls still available
        """
        return max(0, self.tool_budget - self.tool_calls_used)

    @property
    def files_observed(self) -> Set[str]:
        """Get files that have been read.

        Uses protocol method get_observed_files() when available.

        Returns:
            Set of file paths that were read
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_observed_files"):
            return self._orchestrator.get_observed_files()
        return set()

    @property
    def files_modified(self) -> Set[str]:
        """Get files that have been modified.

        Uses protocol method get_modified_files() when available.

        Returns:
            Set of file paths that were written or edited
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_modified_files"):
            return self._orchestrator.get_modified_files()
        return set()

    @property
    def message_count(self) -> int:
        """Get number of messages in conversation.

        Uses protocol method get_message_count() when available.

        Returns:
            Count of messages
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_message_count"):
            return self._orchestrator.get_message_count()
        return 0

    @property
    def is_streaming(self) -> bool:
        """Check if agent is currently streaming.

        Uses protocol method is_streaming() when available.

        Returns:
            True if streaming is in progress
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "is_streaming"):
            return self._orchestrator.is_streaming()
        return False

    @property
    def provider(self) -> str:
        """Get current provider name.

        Uses protocol property current_provider when available.

        Returns:
            Provider name (e.g., "anthropic", "openai")
        """
        # Prefer protocol property
        if hasattr(self._orchestrator, "current_provider"):
            return self._orchestrator.current_provider
        return "unknown"

    @property
    def model(self) -> str:
        """Get current model name.

        Uses protocol property current_model when available.

        Returns:
            Model identifier
        """
        # Prefer protocol property
        if hasattr(self._orchestrator, "current_model"):
            return self._orchestrator.current_model
        return "unknown"

    @property
    def iteration_count(self) -> int:
        """Get current iteration count.

        Uses protocol method get_iteration_count() when available.

        Returns:
            Number of agent loop iterations
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_iteration_count"):
            return self._orchestrator.get_iteration_count()
        return 0

    @property
    def max_iterations(self) -> int:
        """Get maximum allowed iterations.

        Uses protocol method get_max_iterations() when available.

        Returns:
            Maximum iteration limit
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "get_max_iterations"):
            return self._orchestrator.get_max_iterations()
        return 25  # Default

    def reset(self) -> None:
        """Reset state to initial.

        Uses protocol method reset() when available.
        This is called internally when conversation is reset.
        """
        # Prefer protocol method
        if _has_protocol_method(self._orchestrator, "reset"):
            self._orchestrator.reset()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of current state
        """
        return {
            "stage": self.stage.value,
            "tool_calls_used": self.tool_calls_used,
            "tool_budget": self.tool_budget,
            "tools_remaining": self.tools_remaining,
            "files_observed": list(self.files_observed),
            "files_modified": list(self.files_modified),
            "message_count": self.message_count,
            "is_streaming": self.is_streaming,
            "provider": self.provider,
            "model": self.model,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
        }

    def __repr__(self) -> str:
        return (
            f"State(stage={self.stage.value}, "
            f"tools={self.tool_calls_used}/{self.tool_budget}, "
            f"messages={self.message_count})"
        )

    # =========================================================================
    # Observability Integration
    # =========================================================================

    def subscribe_to_state_changes(
        self,
        callback: Callable[[str, str, Dict[str, Any]], None],
    ) -> Optional[Callable[[], None]]:
        """Subscribe to state machine transitions via EventBus.

        This provides real-time notifications when the agent's state
        changes, using the observability EventBus.

        Args:
            callback: Function called with (old_stage, new_stage, context)

        Returns:
            Unsubscribe function, or None if observability is disabled

        Example:
            def log_transition(old, new, ctx):
                print(f"State: {old} -> {new}")

            unsubscribe = state.subscribe_to_state_changes(log_transition)
        """
        # Check for observability property (protocol-compliant)
        observability = getattr(self._orchestrator, "observability", None)
        if not observability:
            return None

        from victor.observability import EventCategory

        def handler(event: Any) -> None:
            data = event.data if hasattr(event, "data") else {}
            old_stage = data.get("old_stage", "unknown")
            new_stage = data.get("new_stage", "unknown")
            context = data.get("context", {})
            callback(old_stage, new_stage, context)

        return observability.event_bus.subscribe(EventCategory.STATE, handler)

    @property
    def transition_history(self) -> list:
        """Get state transition history.

        Returns:
            List of transition records (if available from state machine)
        """
        # This still uses direct access as it's not in the protocol
        # (transition_history is internal state machine detail)
        if hasattr(self._orchestrator, "conversation_state"):
            sm = self._orchestrator.conversation_state
            if hasattr(sm, "transition_history"):
                return list(sm.transition_history)
        return []
