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

"""Chat State Implementation for Phase 1.

This module provides the MutableChatState class which implements
the ChatStateProtocol for domain-agnostic chat state management.

Phase 1: Foundation - Domain-Agnostic Workflow Chat
====================================================
Provides thread-safe, copy-on-write state management for chat workflows.

Design Pattern:
- Thread-Safe: Uses threading.Lock for concurrent access
- Copy-on-Write: State mutations create new versions
- Extensible: Metadata allows verticals to add custom fields

Usage:
    state = MutableChatState()
    state.add_message("user", "Hello!")
    state.increment_iteration()
    state.set_metadata("tool_calls", ["read_file"])
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.framework.protocols import ChatStateProtocol

logger = logging.getLogger(__name__)


# =============================================================================
# Chat Result (Immutable)
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


# =============================================================================
# Mutable Chat State
# =============================================================================


@dataclass
class MutableChatState(ChatStateProtocol):
    """Default implementation of chat state protocol.

    Phase 1: Foundation - Domain-Agnostic Workflow Chat
    This class provides thread-safe, copy-on-write state management
    for chat workflows.

    Features:
    - Thread-safe with locking
    - Message history tracking
    - Iteration counting
    - Extensible metadata

    Example:
        state = MutableChatState()
        state.add_message("user", "Hello!")
        state.increment_iteration()
        state.set_metadata("task_type", "coding")

        # Convert to dict for serialization
        state_dict = state.to_dict()
    """

    _messages: List[Dict[str, Any]] = field(default_factory=list)
    _iteration_count: int = 0
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get the conversation messages.

        Returns:
            List of message dictionaries
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._iteration_count += 1
            logger.debug(f"Iteration count: {self._iteration_count}")

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value
        """
        with self._lock:
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
        with self._lock:
            return self._metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        with self._lock:
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
        with state._lock:
            state._messages = list(data.get("messages", []))
            state._iteration_count = data.get("iteration_count", 0)
            state._metadata = dict(data.get("metadata", {}))
        return state

    def clear(self) -> None:
        """Clear all state (reset to initial state).

        This method is useful for starting fresh with a new conversation.
        """
        with self._lock:
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


__all__ = [
    "ChatResult",
    "MutableChatState",
]
