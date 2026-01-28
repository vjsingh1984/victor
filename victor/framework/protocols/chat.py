"""Phase 1: Workflow chat protocols and implementations.

This module provides domain-agnostic protocols and implementations for
workflow-based chat execution. These components enable the framework
to execute chat workflows without any domain-specific knowledge.

Architecture:
    - ChatStateProtocol: Interface for chat workflow state
    - ChatResultProtocol: Interface for chat workflow results
    - WorkflowChatProtocol: Interface for workflow-based chat execution
    - ChatResult: Immutable result implementation
    - MutableChatState: Thread-safe, copy-on-write state implementation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: Workflow Chat Protocols (Domain-Agnostic)
# =============================================================================


@runtime_checkable
class ChatStateProtocol(Protocol):
    """Protocol for chat workflow state (domain-agnostic).

    Phase 1: Foundation - Domain-Agnostic Workflow Chat
    This protocol defines the interface for chat state that can be
    extended by verticals (e.g., CodingChatState) for domain-specific
    state management.

    The state is passed between workflow nodes and maintains the
    conversation history, iteration count, and metadata.

    Attributes:
        messages: List of conversation messages
        iteration_count: Current iteration number

    Methods:
        add_message: Add a message to the conversation
        increment_iteration: Increment the iteration counter
        set_metadata: Set metadata key-value pair
        get_metadata: Get metadata value by key
    """

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get the conversation messages.

        Returns:
            List of message dictionaries with role and content
        """
        ...

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count.

        Returns:
            Number of iterations that have been executed
        """
        ...

    def add_message(
        self, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add a message to the conversation history."""
        ...

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        ...

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata key-value pair."""
        ...

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        ...


@runtime_checkable
class ChatResultProtocol(Protocol):
    """Protocol for chat workflow execution results.

    Phase 1: Foundation - Domain-Agnostic Workflow Chat
    This protocol defines the interface for results returned from
    workflow-based chat execution.

    Attributes:
        content: The final response content
        iteration_count: Number of iterations executed
        metadata: Additional metadata about the execution

    Methods:
        to_dict: Convert result to dictionary
        get_summary: Get a summary of the execution
    """

    @property
    def content(self) -> str:
        """Get the final response content.

        Returns:
            String content of the response
        """
        ...

    @property
    def iteration_count(self) -> int:
        """Get the number of iterations executed.

        Returns:
            Number of iterations
        """
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get execution metadata.

        Returns:
            Dictionary with metadata like tools_used, files_modified, etc.
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        ...

    def get_summary(self) -> str:
        """Get a summary of the execution.

        Returns:
            String summary with key statistics
        """
        ...


@runtime_checkable
class WorkflowChatProtocol(Protocol):
    """Protocol for workflow-based chat execution.

    Phase 1: Foundation - Domain-Agnostic Workflow Chat
    This protocol defines the interface for domain-agnostic workflow-based
    chat execution using StateGraph.

    Implementations of this protocol should:
    - Use workflow coordinator to load workflows
    - Use graph coordinator to execute workflows
    - Return results conforming to ChatResultProtocol
    - Support streaming via AsyncIterator

    Methods:
        execute_chat_workflow: Execute a chat workflow
        stream_chat_workflow: Stream chat workflow execution
        list_workflows: List available workflows
        get_workflow_info: Get information about a specific workflow
    """

    async def execute_chat_workflow(
        self, workflow_name: str, initial_state: Dict[str, Any]
    ) -> ChatResultProtocol:
        """Execute chat workflow and return result.

        Args:
            workflow_name: Name of the workflow to execute
            initial_state: Initial state for the workflow

        Returns:
            ChatResultProtocol with execution result
        """
        ...

    async def stream_chat_workflow(
        self, workflow_name: str, initial_state: Dict[str, Any]
    ) -> AsyncIterator[ChatResultProtocol]:
        """Stream chat workflow execution.

        Yields intermediate results during execution.

        Args:
            workflow_name: Name of the workflow to execute
            initial_state: Initial state for the workflow

        Yields:
            ChatResultProtocol with intermediate results
        """
        ...
        yield  # type: ignore[misc]

    def list_workflows(self) -> List[str]:
        """List available chat workflows.

        Returns:
            List of workflow names
        """
        ...

    def get_workflow_info(self, workflow_name: str) -> Dict[str, Any]:
        """Get information about a workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Dictionary with workflow information (description, version, etc.)
        """
        ...


# =============================================================================
# Phase 1: Chat State and Result Implementations
# =============================================================================


@dataclass(frozen=True)
class ChatResult:
    """Immutable result of chat workflow execution.

    Phase 1: Foundation - Domain-Agnostic Workflow Chat
    This class implements ChatResultProtocol for workflow execution results.

    Attributes:
        content: The final response content
        iteration_count: Number of iterations executed
        metadata: Additional metadata about the execution
    """

    content: str
    iteration_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation of result
        """
        return {
            "content": self.content,
            "iteration_count": self.iteration_count,
            "metadata": self.metadata,
        }

    def get_summary(self) -> str:
        """Get a summary of the execution.

        Returns:
            String summary with key statistics
        """
        return (
            f"ChatResult(iterations={self.iteration_count}, "
            f"content_length={len(self.content)}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )


@dataclass
class MutableChatState(ChatStateProtocol):
    """Default implementation of chat state protocol.

    Phase 1: Foundation - Domain-Agnostic Workflow Chat
    This class provides state management for chat workflows.

    Features:
    - Message history tracking
    - Iteration counting
    - Extensible metadata

    Note: This implementation is not thread-safe. For concurrent access,
    use external locking or create a thread-safe subclass.

    Example:
        state = MutableChatState()
        state.add_message("user", "Hello!")
        state.increment_iteration()
        state.set_metadata("task_type", "coding")
    """

    _messages: List[Dict[str, Any]] = field(default_factory=list)
    _iteration_count: int = 0
    _metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get the conversation messages.

        Returns:
            List of message dictionaries
        """
        # Return copy to prevent external modification
        return list(self._messages)

    @property
    def iteration_count(self) -> int:
        """Get the current iteration count.

        Returns:
            Number of iterations
        """
        return self._iteration_count

    def add_message(
        self, role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, tool, system)
            content: Message content
            tool_calls: Optional list of tool calls
        """
        message = {
            "role": role,
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        self._messages.append(message)
        logger.debug(f"Added message: role={role}, content_length={len(content)}")

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self._iteration_count += 1
        logger.debug(f"Iteration count: {self._iteration_count}")

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        logger.debug(f"Set metadata: {key}={value}")

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        return {
            "messages": list(self._messages),
            "iteration_count": self._iteration_count,
            "metadata": dict(self._metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MutableChatState":
        """Create state from dictionary.

        Args:
            data: Dictionary representation of state

        Returns:
            New MutableChatState instance
        """
        state = cls()
        state._messages = list(data.get("messages", []))
        state._iteration_count = data.get("iteration_count", 0)
        state._metadata = dict(data.get("metadata", {}))
        return state

    def clear(self) -> None:
        """Clear all state (reset to initial state).

        This method is useful for starting fresh with a new conversation.
        """
        self._messages.clear()
        self._iteration_count = 0
        self._metadata.clear()
        logger.debug("Cleared chat state")

    def __repr__(self) -> str:
        """Return string representation of state.

        Returns:
            String representation
        """
        return (
            f"MutableChatState("
            f"messages={len(self._messages)}, "
            f"iteration={self._iteration_count}, "
            f"metadata_keys={list(self._metadata.keys())})"
        )
