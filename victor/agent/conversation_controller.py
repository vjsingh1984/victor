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

"""Conversation Controller - Manages message history and conversation state.

This module extracts conversation management responsibilities from AgentOrchestrator:
- Message history management
- Context size tracking and overflow detection
- Conversation stage tracking
- System prompt management

Design Principles:
- Single Responsibility: Only handles conversation state
- Composable: Works alongside other controllers
- Observable: Emits events for state changes
- Testable: No external dependencies beyond data classes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.message_history import MessageHistory
from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.providers.base import Message

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ContextMetrics:
    """Metrics about conversation context size."""

    char_count: int
    estimated_tokens: int
    message_count: int
    is_overflow_risk: bool = False
    max_context_chars: int = 200000

    @property
    def utilization(self) -> float:
        """Calculate context utilization percentage."""
        if self.max_context_chars == 0:
            return 0.0
        return min(1.0, self.char_count / self.max_context_chars)


@dataclass
class ConversationConfig:
    """Configuration for conversation controller."""

    max_context_chars: int = 200000
    chars_per_token_estimate: int = 4
    enable_stage_tracking: bool = True
    enable_context_monitoring: bool = True


class ConversationController:
    """Manages conversation history and state.

    This controller handles all conversation-related concerns:
    - Adding/retrieving messages
    - Tracking conversation stages (exploring, implementing, etc.)
    - Monitoring context size and overflow risk
    - Managing system prompts

    Example:
        controller = ConversationController(config)
        controller.set_system_prompt("You are a helpful assistant.")
        controller.add_user_message("Hello!")
        controller.add_assistant_message("Hi there!")

        metrics = controller.get_context_metrics()
        if metrics.is_overflow_risk:
            controller.compact_history()
    """

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        message_history: Optional[MessageHistory] = None,
        state_machine: Optional[ConversationStateMachine] = None,
    ):
        """Initialize conversation controller.

        Args:
            config: Conversation configuration
            message_history: Optional pre-existing message history
            state_machine: Optional pre-existing state machine
        """
        self.config = config or ConversationConfig()
        self._history = message_history or MessageHistory()
        self._state_machine = state_machine or ConversationStateMachine()
        self._system_prompt: Optional[str] = None
        self._system_added = False
        self._context_callbacks: List[Callable[[ContextMetrics], None]] = []

    @property
    def messages(self) -> List[Message]:
        """Get all messages in conversation history."""
        return self._history.messages

    @property
    def message_count(self) -> int:
        """Get number of messages in history."""
        return len(self._history.messages)

    @property
    def stage(self) -> ConversationStage:
        """Get current conversation stage."""
        return self._state_machine.get_stage()

    @property
    def system_prompt(self) -> Optional[str]:
        """Get the system prompt."""
        return self._system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt.

        Args:
            prompt: System prompt text
        """
        self._system_prompt = prompt
        self._system_added = False

    def ensure_system_message(self) -> None:
        """Ensure system message is added to history if not already present."""
        if self._system_added or not self._system_prompt:
            return

        # Check if system message already exists
        if self._history.messages and self._history.messages[0].role == "system":
            self._system_added = True
            return

        # Add system message at the beginning
        system_msg = Message(role="system", content=self._system_prompt)
        self._history._messages.insert(0, system_msg)
        self._system_added = True

    def add_user_message(self, content: str) -> Message:
        """Add a user message to history.

        Args:
            content: Message content

        Returns:
            The created message
        """
        self.ensure_system_message()
        message = self._history.add_user_message(content)

        # Update conversation state
        if self.config.enable_stage_tracking:
            self._state_machine.record_message(content, is_user=True)

        # Check context size
        if self.config.enable_context_monitoring:
            metrics = self.get_context_metrics()
            if metrics.is_overflow_risk:
                self._notify_context_callbacks(metrics)

        return message

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Add an assistant message to history.

        Args:
            content: Message content
            tool_calls: Optional tool calls made by assistant

        Returns:
            The created message
        """
        message = self._history.add_assistant_message(content, tool_calls=tool_calls)

        # Update conversation state
        if self.config.enable_stage_tracking:
            self._state_machine.record_message(content, is_user=False)

        return message

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> Message:
        """Add a tool result message to history.

        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            result: Tool execution result

        Returns:
            The created message
        """
        message = self._history.add_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=result,
        )
        return message

    def add_message(self, role: str, content: str) -> Message:
        """Add a message with specified role (backward compatibility).

        Args:
            role: Message role (user, assistant, system)
            content: Message content

        Returns:
            The created message
        """
        if role == "user":
            return self.add_user_message(content)
        elif role == "assistant":
            return self.add_assistant_message(content)
        elif role == "system":
            self.set_system_prompt(content)
            self.ensure_system_message()
            return self.messages[0]
        else:
            # For other roles, use add_message from history
            return self._history.add_message(role, content)

    def get_context_metrics(self) -> ContextMetrics:
        """Calculate current context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        total_chars = sum(len(m.content) for m in self.messages)
        estimated_tokens = total_chars // self.config.chars_per_token_estimate

        return ContextMetrics(
            char_count=total_chars,
            estimated_tokens=estimated_tokens,
            message_count=len(self.messages),
            is_overflow_risk=total_chars > self.config.max_context_chars,
            max_context_chars=self.config.max_context_chars,
        )

    def check_context_overflow(self) -> bool:
        """Check if context is at risk of overflow.

        Returns:
            True if context is dangerously large
        """
        metrics = self.get_context_metrics()
        if metrics.is_overflow_risk:
            logger.warning(
                f"Context overflow risk: {metrics.char_count:,} chars "
                f"(~{metrics.estimated_tokens:,} tokens). "
                f"Max: {metrics.max_context_chars:,} chars"
            )
        return metrics.is_overflow_risk

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for current conversation stage.

        Returns:
            Set of recommended tool names
        """
        return self._state_machine.get_stage_tools()

    def reset(self) -> None:
        """Reset conversation to initial state."""
        self._history.clear()
        self._state_machine.reset()
        self._system_added = False
        logger.info("Conversation reset")

    def compact_history(self, keep_recent: int = 10) -> int:
        """Compact history by removing old messages.

        Keeps system message and most recent messages.

        Args:
            keep_recent: Number of recent messages to keep

        Returns:
            Number of messages removed
        """
        if len(self.messages) <= keep_recent + 1:  # +1 for system message
            return 0

        # Keep system message and recent messages
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]

        recent = self.messages[-keep_recent:]
        removed = len(self.messages) - keep_recent - (1 if system_msg else 0)

        self._history.clear()
        if system_msg:
            self._history._messages.append(system_msg)
        for msg in recent:
            self._history._messages.append(msg)

        logger.info(f"Compacted history: removed {removed} messages")
        return removed

    def on_context_overflow(self, callback: Callable[[ContextMetrics], None]) -> None:
        """Register callback for context overflow events.

        Args:
            callback: Function to call when overflow is detected
        """
        self._context_callbacks.append(callback)

    def _notify_context_callbacks(self, metrics: ContextMetrics) -> None:
        """Notify registered callbacks about context state."""
        for callback in self._context_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.warning(f"Context callback failed: {e}")

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message.

        Returns:
            Last user message content or None
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message.

        Returns:
            Last assistant message content or None
        """
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export conversation state as dictionary.

        Returns:
            Dictionary representation of conversation
        """
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "stage": self.stage.value,
            "metrics": {
                "char_count": self.get_context_metrics().char_count,
                "message_count": self.message_count,
            },
        }

    @classmethod
    def from_messages(
        cls,
        messages: List[Message],
        config: Optional[ConversationConfig] = None,
    ) -> "ConversationController":
        """Create controller from existing messages.

        Args:
            messages: List of messages to initialize with
            config: Optional configuration

        Returns:
            New ConversationController with messages
        """
        controller = cls(config=config)
        for msg in messages:
            controller._history._messages.append(msg)
        if messages and messages[0].role == "system":
            controller._system_prompt = messages[0].content
            controller._system_added = True
        return controller
