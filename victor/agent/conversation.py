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

"""Conversation management for maintaining chat history and context."""

import logging
from typing import Any, Dict, List, Optional

from victor.providers.base import Message

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and message formatting.

    Responsibilities:
    - Maintain conversation history
    - Add system prompts
    - Format messages for providers
    - Manage context window limits
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_history_messages: int = 100,
    ):
        """Initialize conversation manager.

        Args:
            system_prompt: Initial system prompt for the conversation
            max_history_messages: Maximum messages to retain in history
        """
        self._system_prompt = system_prompt
        self._system_added = False
        self._messages: List[Message] = []
        self._max_history = max_history_messages

    @property
    def messages(self) -> List[Message]:
        """Get all messages in conversation history."""
        return self._messages.copy()

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set the system prompt."""
        self._system_prompt = value

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message to conversation history.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            **kwargs: Additional message fields (tool_calls, tool_call_id, name)

        Returns:
            The created Message object
        """
        message = Message(role=role, content=content, **kwargs)
        self._messages.append(message)
        self._trim_history()
        return message

    def add_user_message(self, content: str) -> Message:
        """Add a user message to conversation history."""
        return self.add_message("user", content)

    def add_assistant_message(
        self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> Message:
        """Add an assistant message to conversation history."""
        return self.add_message("assistant", content, tool_calls=tool_calls)

    def add_tool_result(
        self, tool_call_id: str, content: str, tool_name: Optional[str] = None
    ) -> Message:
        """Add a tool result message to conversation history.

        Args:
            tool_call_id: ID of the tool call being responded to
            content: Tool execution result
            tool_name: Optional name of the tool

        Returns:
            The created Message object
        """
        return self.add_message("tool", content, tool_call_id=tool_call_id, name=tool_name)

    def ensure_system_prompt(self) -> None:
        """Ensure system prompt is added to the beginning of conversation."""
        if not self._system_added and self._system_prompt:
            # Insert at the beginning
            system_message = Message(role="system", content=self._system_prompt)
            self._messages.insert(0, system_message)
            self._system_added = True

    def get_messages_for_provider(self) -> List[Message]:
        """Get messages formatted for sending to provider.

        Ensures system prompt is included and history is within limits.

        Returns:
            List of messages ready for provider
        """
        self.ensure_system_prompt()
        return self._messages.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
        self._system_added = False

    def _trim_history(self) -> None:
        """Trim history to max_history_messages, preserving system prompt."""
        if len(self._messages) <= self._max_history:
            return

        # Keep system message if present
        system_msg = None
        if self._messages and self._messages[0].role == "system":
            system_msg = self._messages[0]

        # Trim from the front (oldest messages), keeping recent ones
        excess = len(self._messages) - self._max_history
        self._messages = self._messages[excess:]

        # Re-add system message at front if it was removed
        if system_msg and (not self._messages or self._messages[0].role != "system"):
            self._messages.insert(0, system_msg)

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message."""
        for msg in reversed(self._messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def message_count(self) -> int:
        """Get the number of messages in history."""
        return len(self._messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation state to dictionary."""
        return {
            "system_prompt": self._system_prompt,
            "system_added": self._system_added,
            "messages": [msg.model_dump() for msg in self._messages],
            "max_history": self._max_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationManager":
        """Deserialize conversation state from dictionary."""
        manager = cls(
            system_prompt=data.get("system_prompt", ""),
            max_history_messages=data.get("max_history", 100),
        )
        manager._system_added = data.get("system_added", False)
        for msg_data in data.get("messages", []):
            manager._messages.append(Message(**msg_data))
        return manager
