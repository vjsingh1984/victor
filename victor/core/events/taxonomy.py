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

"""Unified Event Taxonomy for Victor.

This module consolidates the fragmented event types across the codebase into
a unified hierarchy. It provides a single source of truth for event types
while maintaining backward compatibility with existing event systems.

The taxonomy follows a hierarchical naming convention:
- framework.* : Framework-level events (content, thinking)
- tool.* : Tool execution events (call, result, error)
- workflow.* : Workflow lifecycle events (start, node, end)
- agent.* : Agent behavior events (thinking, response)
- system.* : System-level events (health, metrics)

Design Principles:
- Hierarchical: Events are organized by domain for easy filtering
- Extensible: New event types can be added without breaking changes
- Mappable: Existing event types can be mapped to unified taxonomy
- Backward Compatible: Old event types continue to work via mapping

Example:
    from victor.core.events.taxonomy import (
        UnifiedEventType,
        map_workflow_event,
    )
    from victor.workflows.streaming import WorkflowEventType

    # Map existing event to unified taxonomy
    workflow_event = WorkflowEventType.NODE_START
    unified = map_workflow_event(workflow_event)
    assert unified == UnifiedEventType.WORKFLOW_NODE_START

    # Use unified event types directly
    event_type = UnifiedEventType.TOOL_CALL
    print(event_type.value)  # "tool.call"
    print(event_type.category)  # "tool"
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # EventCategory removed - use topic-based routing
    from victor.workflows.streaming import WorkflowEventType


class UnifiedEventType(str, Enum):
    """Unified event type taxonomy for all Victor events.

    This enum provides a consolidated, hierarchical event type system that
    unifies the various event types used across different Victor subsystems.

    Categories:
    - framework.* : Framework-level streaming events
    - tool.* : Tool execution lifecycle events
    - workflow.* : Workflow execution lifecycle events
    - agent.* : Agent behavior and response events
    - system.* : System health and metrics events

    The string values use dot notation for hierarchical organization,
    enabling pattern-based filtering (e.g., "workflow.*" matches all
    workflow events).

    Example:
        event_type = UnifiedEventType.WORKFLOW_NODE_START
        print(event_type.value)  # "workflow.node.start"
        print(event_type.category)  # "workflow"
        print(event_type.is_workflow_event)  # True
    """

    # =========================================================================
    # Framework Events
    # Events related to the framework's streaming and processing infrastructure
    # =========================================================================

    FRAMEWORK_CONTENT = "framework.content"
    """Content streaming chunk from framework."""

    FRAMEWORK_THINKING = "framework.thinking"
    """Thinking/reasoning content from framework."""

    FRAMEWORK_CHUNK = "framework.chunk"
    """Generic streaming chunk from framework."""

    # =========================================================================
    # Tool Events
    # Events related to tool invocation and execution
    # =========================================================================

    TOOL_CALL = "tool.call"
    """Tool invocation initiated."""

    TOOL_RESULT = "tool.result"
    """Tool execution completed successfully."""

    TOOL_ERROR = "tool.error"
    """Tool execution failed with an error."""

    TOOL_START = "tool.start"
    """Tool execution starting (before invocation)."""

    TOOL_END = "tool.end"
    """Tool execution ending (after completion or error)."""

    # =========================================================================
    # Workflow Events
    # Events related to workflow execution lifecycle
    # =========================================================================

    WORKFLOW_START = "workflow.start"
    """Workflow execution has begun."""

    WORKFLOW_END = "workflow.end"
    """Workflow execution has completed (success or failure)."""

    WORKFLOW_COMPLETE = "workflow.complete"
    """Workflow execution completed successfully."""

    WORKFLOW_ERROR = "workflow.error"
    """Workflow execution failed with an error."""

    WORKFLOW_NODE_START = "workflow.node.start"
    """A workflow node has started executing."""

    WORKFLOW_NODE_COMPLETE = "workflow.node.complete"
    """A workflow node has completed successfully."""

    WORKFLOW_NODE_ERROR = "workflow.node.error"
    """A workflow node failed with an error."""

    WORKFLOW_PROGRESS = "workflow.progress"
    """Workflow progress update."""

    WORKFLOW_CHECKPOINT = "workflow.checkpoint"
    """Workflow checkpoint was saved."""

    # =========================================================================
    # Agent Events
    # Events related to agent behavior and responses
    # =========================================================================

    AGENT_THINKING = "agent.thinking"
    """Agent is processing/thinking."""

    AGENT_RESPONSE = "agent.response"
    """Agent has generated a response."""

    AGENT_CONTENT = "agent.content"
    """Streaming content from agent."""

    AGENT_TOOL_CALL = "agent.tool_call"
    """Agent is making a tool call."""

    AGENT_TOOL_RESULT = "agent.tool_result"
    """Agent received a tool result."""

    # =========================================================================
    # System Events
    # Events related to system health and metrics
    # =========================================================================

    SYSTEM_HEALTH = "system.health"
    """System health status event."""

    SYSTEM_METRICS = "system.metrics"
    """System metrics event."""

    SYSTEM_ERROR = "system.error"
    """System error event."""

    SYSTEM_WARNING = "system.warning"
    """System warning event."""

    SYSTEM_LIFECYCLE = "system.lifecycle"
    """System lifecycle event (start/stop)."""

    # =========================================================================
    # Model/Provider Events
    # Events related to LLM interactions
    # =========================================================================

    MODEL_REQUEST = "model.request"
    """Model request initiated."""

    MODEL_RESPONSE = "model.response"
    """Model response received."""

    MODEL_STREAM_CHUNK = "model.stream.chunk"
    """Model streaming chunk received."""

    MODEL_ERROR = "model.error"
    """Model request failed."""

    # =========================================================================
    # State Events
    # Events related to state machine transitions
    # =========================================================================

    STATE_TRANSITION = "state.transition"
    """State machine transition occurred."""

    STATE_ENTRY = "state.entry"
    """Entered a new state."""

    STATE_EXIT = "state.exit"
    """Exited a state."""

    # =========================================================================
    # Debug Events
    # Events related to workflow debugging and breakpoints
    # =========================================================================

    DEBUG_BREAKPOINT_SET = "debug.breakpoint.set"
    """Breakpoint was set."""

    DEBUG_BREAKPOINT_CLEARED = "debug.breakpoint.cleared"
    """Breakpoint was cleared."""

    DEBUG_BREAKPOINT_HIT = "debug.breakpoint.hit"
    """Breakpoint was hit during execution."""

    DEBUG_PAUSED = "debug.paused"
    """Execution paused at breakpoint."""

    DEBUG_RESUMED = "debug.resumed"
    """Execution resumed after pause."""

    DEBUG_STEP = "debug.step"
    """Execution step (step over/into/out)."""

    DEBUG_STATE_INSPECTED = "debug.state.inspected"
    """State was inspected for debugging."""

    # =========================================================================
    # Team Events
    # Events related to multi-agent team execution
    # =========================================================================

    TEAM_EXECUTION_STARTED = "team.execution.started"
    """Team execution has started."""

    TEAM_EXECUTION_COMPLETED = "team.execution.completed"
    """Team execution has completed."""

    TEAM_MEMBER_STARTED = "team.member.started"
    """Team member execution has started."""

    TEAM_MEMBER_COMPLETED = "team.member.completed"
    """Team member execution has completed."""

    TEAM_MEMBER_FAILED = "team.member.failed"
    """Team member execution has failed."""

    TEAM_RECURSION_DEPTH_EXCEEDED = "team.recursion.depth_exceeded"
    """Team recursion depth limit was exceeded."""

    TEAM_CONSENSUS_ACHIEVED = "team.consensus.achieved"
    """Team reached consensus (for consensus formation)."""

    TEAM_CONSENSUS_FAILED = "team.consensus.failed"
    """Team failed to reach consensus."""

    TEAM_PROGRESS_UPDATE = "team.progress.update"
    """Team execution progress update."""

    # =========================================================================
    # Custom/Extension Events
    # =========================================================================

    CUSTOM = "custom"
    """Custom user-defined event."""

    UNKNOWN = "unknown"
    """Unknown or unmapped event type."""

    @property
    def category(self) -> str:
        """Get the category (first segment) of the event type.

        Returns:
            The category portion of the event type value.

        Example:
            event = UnifiedEventType.WORKFLOW_NODE_START
            print(event.category)  # "workflow"
        """
        return self.value.split(".")[0]

    @property
    def is_workflow_event(self) -> bool:
        """Check if this is a workflow-related event.

        Returns:
            True if the event is in the workflow category.
        """
        return self.category == "workflow"

    @property
    def is_tool_event(self) -> bool:
        """Check if this is a tool-related event.

        Returns:
            True if the event is in the tool category.
        """
        return self.category == "tool"

    @property
    def is_agent_event(self) -> bool:
        """Check if this is an agent-related event.

        Returns:
            True if the event is in the agent category.
        """
        return self.category == "agent"

    @property
    def is_system_event(self) -> bool:
        """Check if this is a system-related event.

        Returns:
            True if the event is in the system category.
        """
        return self.category == "system"

    @property
    def is_error_event(self) -> bool:
        """Check if this is an error event.

        Returns:
            True if the event represents an error.
        """
        return "error" in self.value

    @classmethod
    def from_string(cls, value: str) -> "UnifiedEventType":
        """Parse a string value to UnifiedEventType.

        Attempts to find a matching enum member. Falls back to UNKNOWN
        if no match is found.

        Args:
            value: String value to parse (e.g., "workflow.node.start")

        Returns:
            Matching UnifiedEventType or UNKNOWN if not found.

        Example:
            event = UnifiedEventType.from_string("tool.call")
            assert event == UnifiedEventType.TOOL_CALL
        """
        # Try exact match
        for member in cls:
            if member.value == value:
                return member

        # Fall back to UNKNOWN
        return cls.UNKNOWN


# =============================================================================
# Mapping Functions
# =============================================================================

# Mapping from WorkflowEventType to UnifiedEventType
_WORKFLOW_EVENT_MAPPING: dict[str, UnifiedEventType] = {
    "workflow_start": UnifiedEventType.WORKFLOW_START,
    "workflow_complete": UnifiedEventType.WORKFLOW_COMPLETE,
    "workflow_error": UnifiedEventType.WORKFLOW_ERROR,
    "node_start": UnifiedEventType.WORKFLOW_NODE_START,
    "node_complete": UnifiedEventType.WORKFLOW_NODE_COMPLETE,
    "node_error": UnifiedEventType.WORKFLOW_NODE_ERROR,
    "agent_content": UnifiedEventType.AGENT_CONTENT,
    "agent_tool_call": UnifiedEventType.AGENT_TOOL_CALL,
    "agent_tool_result": UnifiedEventType.AGENT_TOOL_RESULT,
    "progress_update": UnifiedEventType.WORKFLOW_PROGRESS,
    "checkpoint_saved": UnifiedEventType.WORKFLOW_CHECKPOINT,
}

# Mapping from EventCategory to UnifiedEventType (for default event of category)
_EVENT_CATEGORY_MAPPING: dict[str, UnifiedEventType] = {
    "tool": UnifiedEventType.TOOL_CALL,
    "state": UnifiedEventType.STATE_TRANSITION,
    "model": UnifiedEventType.MODEL_REQUEST,
    "error": UnifiedEventType.SYSTEM_ERROR,
    "audit": UnifiedEventType.SYSTEM_LIFECYCLE,
    "metric": UnifiedEventType.SYSTEM_METRICS,
    "lifecycle": UnifiedEventType.SYSTEM_LIFECYCLE,
    "vertical": UnifiedEventType.CUSTOM,
    "custom": UnifiedEventType.CUSTOM,
}


def map_workflow_event(event: "WorkflowEventType") -> UnifiedEventType:
    """Map a WorkflowEventType to the unified taxonomy.

    This function converts workflow-specific event types from
    victor.workflows.streaming to the unified event taxonomy.

    Args:
        event: WorkflowEventType enum value to map.

    Returns:
        Corresponding UnifiedEventType, or UNKNOWN if no mapping exists.

    Example:
        from victor.workflows.streaming import WorkflowEventType

        workflow_event = WorkflowEventType.NODE_START
        unified = map_workflow_event(workflow_event)
        assert unified == UnifiedEventType.WORKFLOW_NODE_START
    """
    event_value = event.value if hasattr(event, "value") else str(event)
    return _WORKFLOW_EVENT_MAPPING.get(event_value, UnifiedEventType.UNKNOWN)


def map_event_category(category: str) -> UnifiedEventType:
    """Map a topic prefix to a default unified event type.

    This function maps event topic prefixes from the canonical event system
    to a representative unified event type. Since categories are broader
    than specific event types, this returns a default event for the category.

    Args:
        category: Topic prefix string to map (e.g., "tool", "state", "model").

    Returns:
        A representative UnifiedEventType for the category.

    Example:
        topic_prefix = "tool"
        unified = map_event_category(topic_prefix)
        assert unified == UnifiedEventType.TOOL_CALL
    """
    # Handle both string and legacy enum-like objects
    category_value = category.value if hasattr(category, "value") else str(category)
    return _EVENT_CATEGORY_MAPPING.get(category_value, UnifiedEventType.UNKNOWN)


def map_framework_event(event_name: str) -> UnifiedEventType:
    """Map a framework event name to the unified taxonomy.

    This function maps framework-level event names (e.g., from graph.py)
    to the unified event taxonomy.

    Args:
        event_name: Framework event name string to map.

    Returns:
        Corresponding UnifiedEventType, or UNKNOWN if no mapping exists.

    Example:
        unified = map_framework_event("content")
        assert unified == UnifiedEventType.FRAMEWORK_CONTENT
    """
    framework_mapping: dict[str, UnifiedEventType] = {
        "content": UnifiedEventType.FRAMEWORK_CONTENT,
        "thinking": UnifiedEventType.FRAMEWORK_THINKING,
        "chunk": UnifiedEventType.FRAMEWORK_CHUNK,
    }
    return framework_mapping.get(event_name.lower(), UnifiedEventType.UNKNOWN)


def map_tool_event(event_name: str) -> UnifiedEventType:
    """Map a tool event name to the unified taxonomy.

    Args:
        event_name: Tool event name (e.g., "start", "end", "error").

    Returns:
        Corresponding UnifiedEventType, or UNKNOWN if no mapping exists.

    Example:
        unified = map_tool_event("call")
        assert unified == UnifiedEventType.TOOL_CALL
    """
    tool_mapping: dict[str, UnifiedEventType] = {
        "call": UnifiedEventType.TOOL_CALL,
        "result": UnifiedEventType.TOOL_RESULT,
        "error": UnifiedEventType.TOOL_ERROR,
        "start": UnifiedEventType.TOOL_START,
        "end": UnifiedEventType.TOOL_END,
    }
    return tool_mapping.get(event_name.lower(), UnifiedEventType.UNKNOWN)


def map_agent_event(event_name: str) -> UnifiedEventType:
    """Map an agent event name to the unified taxonomy.

    Args:
        event_name: Agent event name (e.g., "thinking", "response").

    Returns:
        Corresponding UnifiedEventType, or UNKNOWN if no mapping exists.

    Example:
        unified = map_agent_event("thinking")
        assert unified == UnifiedEventType.AGENT_THINKING
    """
    agent_mapping: dict[str, UnifiedEventType] = {
        "thinking": UnifiedEventType.AGENT_THINKING,
        "response": UnifiedEventType.AGENT_RESPONSE,
        "content": UnifiedEventType.AGENT_CONTENT,
        "tool_call": UnifiedEventType.AGENT_TOOL_CALL,
        "tool_result": UnifiedEventType.AGENT_TOOL_RESULT,
    }
    return agent_mapping.get(event_name.lower(), UnifiedEventType.UNKNOWN)


def map_system_event(event_name: str) -> UnifiedEventType:
    """Map a system event name to the unified taxonomy.

    Args:
        event_name: System event name (e.g., "health", "metrics").

    Returns:
        Corresponding UnifiedEventType, or UNKNOWN if no mapping exists.

    Example:
        unified = map_system_event("health")
        assert unified == UnifiedEventType.SYSTEM_HEALTH
    """
    system_mapping: dict[str, UnifiedEventType] = {
        "health": UnifiedEventType.SYSTEM_HEALTH,
        "metrics": UnifiedEventType.SYSTEM_METRICS,
        "error": UnifiedEventType.SYSTEM_ERROR,
        "warning": UnifiedEventType.SYSTEM_WARNING,
        "lifecycle": UnifiedEventType.SYSTEM_LIFECYCLE,
    }
    return system_mapping.get(event_name.lower(), UnifiedEventType.UNKNOWN)


def map_team_event(event_name: str) -> UnifiedEventType:
    """Map a team event name to the unified taxonomy.

    Args:
        event_name: Team event name (e.g., "execution.started", "member.completed").

    Returns:
        Corresponding UnifiedEventType, or UNKNOWN if no mapping exists.

    Example:
        unified = map_team_event("execution.started")
        assert unified == UnifiedEventType.TEAM_EXECUTION_STARTED
    """
    team_mapping: dict[str, UnifiedEventType] = {
        "execution.started": UnifiedEventType.TEAM_EXECUTION_STARTED,
        "execution.completed": UnifiedEventType.TEAM_EXECUTION_COMPLETED,
        "member.started": UnifiedEventType.TEAM_MEMBER_STARTED,
        "member.completed": UnifiedEventType.TEAM_MEMBER_COMPLETED,
        "member.failed": UnifiedEventType.TEAM_MEMBER_FAILED,
        "recursion.depth_exceeded": UnifiedEventType.TEAM_RECURSION_DEPTH_EXCEEDED,
        "consensus.achieved": UnifiedEventType.TEAM_CONSENSUS_ACHIEVED,
        "consensus.failed": UnifiedEventType.TEAM_CONSENSUS_FAILED,
        "progress.update": UnifiedEventType.TEAM_PROGRESS_UPDATE,
    }
    return team_mapping.get(event_name.lower(), UnifiedEventType.UNKNOWN)


# =============================================================================
# Utility Functions
# =============================================================================


def get_all_categories() -> list[str]:
    """Get all unique event categories.

    Returns:
        List of unique category strings.

    Example:
        categories = get_all_categories()
        # Returns ["framework", "tool", "workflow", "agent", "system", ...]
    """
    categories = set()
    for event in UnifiedEventType:
        categories.add(event.category)
    return sorted(categories)


def get_events_by_category(category: str) -> list[UnifiedEventType]:
    """Get all events for a specific category.

    Args:
        category: Category name (e.g., "workflow", "tool").

    Returns:
        List of UnifiedEventType values in that category.

    Example:
        workflow_events = get_events_by_category("workflow")
        # Returns all workflow.* event types
    """
    return [event for event in UnifiedEventType if event.category == category]


def is_valid_event_type(value: str) -> bool:
    """Check if a string is a valid unified event type.

    Args:
        value: String to check.

    Returns:
        True if the string matches a valid UnifiedEventType value.

    Example:
        assert is_valid_event_type("workflow.node.start") is True
        assert is_valid_event_type("invalid.event") is False
    """
    return UnifiedEventType.from_string(value) != UnifiedEventType.UNKNOWN


# =============================================================================
# Deprecation Helpers
# =============================================================================


def emit_deprecation_warning(
    old_type: str,
    new_type: UnifiedEventType,
    module: str = "unknown",
) -> None:
    """Emit a deprecation warning for old event types.

    This helper function standardizes deprecation warnings across the codebase
    when migrating from fragmented event types to the unified taxonomy.

    Args:
        old_type: The deprecated event type string.
        new_type: The new UnifiedEventType to use instead.
        module: The module where the deprecated type is defined.

    Example:
        # In victor/workflows/streaming.py
        emit_deprecation_warning(
            "workflow_start",
            UnifiedEventType.WORKFLOW_START,
            "victor.workflows.streaming",
        )
    """
    warnings.warn(
        f"Event type '{old_type}' from {module} is deprecated. "
        f"Use UnifiedEventType.{new_type.name} ('{new_type.value}') from "
        f"victor.core.events.taxonomy instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core enum
    "UnifiedEventType",
    # Mapping functions
    "map_workflow_event",
    "map_event_category",
    "map_framework_event",
    "map_tool_event",
    "map_agent_event",
    "map_system_event",
    "map_team_event",
    # Utility functions
    "get_all_categories",
    "get_events_by_category",
    "is_valid_event_type",
    # Deprecation helpers
    "emit_deprecation_warning",
]
