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

"""Context service implementation.

Extracts context management from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Context size monitoring and metrics
- Context overflow detection and prevention
- Context compaction and optimization
- Message history management
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Message is a dict with 'role' and 'content' keys
Message = Dict[str, Any]


class ContextServiceConfig:
    """Configuration for ContextService.

    Attributes:
        max_tokens: Maximum context tokens
        min_messages_to_keep: Minimum messages to retain after compaction
        default_compaction_strategy: Default compaction strategy
        overflow_threshold_percent: Threshold for overflow warning
    """

    def __init__(
        self,
        max_tokens: int = 100000,
        min_messages_to_keep: int = 6,
        default_compaction_strategy: str = "tiered",
        overflow_threshold_percent: float = 90.0,
    ):
        self.max_tokens = max_tokens
        self.min_messages_to_keep = min_messages_to_keep
        self.default_compaction_strategy = default_compaction_strategy
        self.overflow_threshold_percent = overflow_threshold_percent


class ContextMetricsImpl:
    """Implementation of context metrics."""

    def __init__(
        self,
        total_tokens: int,
        message_count: int,
        user_message_count: int,
        assistant_message_count: int,
        tool_result_count: int,
        system_prompt_tokens: int,
        max_tokens: int,
    ):
        self.total_tokens = total_tokens
        self.message_count = message_count
        self.user_message_count = user_message_count
        self.assistant_message_count = assistant_message_count
        self.tool_result_count = tool_result_count
        self.system_prompt_tokens = system_prompt_tokens
        self._max_tokens = max_tokens

    @property
    def utilization_percent(self) -> float:
        """Context utilization as percentage."""
        return (
            (self.total_tokens / self._max_tokens * 100) if self._max_tokens > 0 else 0
        )


class ContextService:
    """Service for context and state management.

    Extracted from AgentOrchestrator to handle:
    - Context size monitoring and metrics
    - Context overflow detection and prevention
    - Context compaction and optimization
    - Message history management

    This service follows SOLID principles:
    - SRP: Only handles context operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements ContextServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        config = ContextServiceConfig()
        service = ContextService(config=config)

        metrics = await service.get_context_metrics()
        if await service.check_context_overflow():
            await service.compact_context()
    """

    def __init__(self, config: ContextServiceConfig):
        """Initialize the context service.

        Args:
            config: Service configuration
        """
        self._config = config
        self._messages: List["Message"] = []
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    async def get_context_metrics(self) -> ContextMetricsImpl:
        """Get current context metrics.

        Returns:
            ContextMetrics with current context information
        """
        total_tokens = sum(
            self._estimate_tokens(getattr(m, "content", "")) for m in self._messages
        )

        user_count = sum(1 for m in self._messages if getattr(m, "role", "") == "user")
        assistant_count = sum(
            1 for m in self._messages if getattr(m, "role", "") == "assistant"
        )
        tool_count = sum(1 for m in self._messages if getattr(m, "role", "") == "tool")

        system_tokens = self._estimate_tokens(
            next(
                (
                    m.content
                    for m in self._messages
                    if getattr(m, "role", "") == "system"
                ),
                "",
            )
        )

        return ContextMetricsImpl(
            total_tokens=total_tokens,
            message_count=len(self._messages),
            user_message_count=user_count,
            assistant_message_count=assistant_count,
            tool_result_count=tool_count,
            system_prompt_tokens=system_tokens,
            max_tokens=self._config.max_tokens,
        )

    async def check_context_overflow(self) -> bool:
        """Check if context exceeds limits.

        Returns:
            True if context exceeds limits, False otherwise
        """
        metrics = await self.get_context_metrics()
        return metrics.total_tokens > self._config.max_tokens

    async def compact_context(
        self,
        strategy: str = "tiered",
        min_messages: int = 6,
    ) -> int:
        """Compact context to fit within limits.

        Args:
            strategy: Compaction strategy
            min_messages: Minimum messages to retain

        Returns:
            Number of messages removed
        """
        original_count = len(self._messages)

        if original_count <= min_messages:
            return 0

        # Keep the most recent messages
        self._messages = self._messages[-min_messages:]

        removed = original_count - len(self._messages)
        self._logger.info(f"Compacted context: removed {removed} messages")

        return removed

    def add_message(self, message: "Message") -> None:
        """Add a message to the context.

        Args:
            message: Message to add
        """
        self._messages.append(message)

    def add_messages(self, messages: List["Message"]) -> None:
        """Add multiple messages to the context.

        Args:
            messages: Messages to add
        """
        self._messages.extend(messages)

    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List["Message"]:
        """Get messages from context.

        Args:
            limit: Maximum number of messages to return
            role: Filter by message role

        Returns:
            List of messages
        """
        messages = self._messages

        if role:
            messages = [m for m in messages if getattr(m, "role", "") == role]

        if limit:
            messages = messages[-limit:]

        return messages

    def clear_messages(self, retain_system: bool = True) -> None:
        """Clear messages from context.

        Args:
            retain_system: If True, retain system prompt
        """
        if retain_system:
            system_messages = [
                m for m in self._messages if getattr(m, "role", "") == "system"
            ]
            self._messages = system_messages
        else:
            self._messages.clear()

    def get_max_tokens(self) -> int:
        """Get the maximum context token limit.

        Returns:
            Maximum tokens allowed in context
        """
        return self._config.max_tokens

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the maximum context token limit.

        Args:
            max_tokens: Maximum tokens allowed

        Raises:
            ValueError: If max_tokens is negative
        """
        if max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative: {max_tokens}")

        self._config.max_tokens = max_tokens
        self._logger.info(f"Max tokens updated to {max_tokens}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple heuristic: ~4 characters per token
        return len(text) // 4

    def is_healthy(self) -> bool:
        """Check if the context service is healthy.

        Returns:
            True if the service is healthy
        """
        return self._config.max_tokens > 0

    def _estimate_tokens(self, text: str) -> int:
        """Internal token estimation."""
        return self.estimate_tokens(text)
