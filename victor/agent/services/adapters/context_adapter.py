"""Context service adapter that wraps ConversationController and ContextCompactor.

Implements ContextServiceProtocol by delegating to the existing
conversation controller and context compactor components.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor

logger = logging.getLogger(__name__)


class ContextServiceAdapter:
    """Adapts ConversationController + ContextCompactor to ContextServiceProtocol.

    This adapter bridges existing conversation management components
    to the service protocol interface.
    """

    def __init__(
        self,
        conversation_controller: "ConversationController",
        context_compactor: Optional["ContextCompactor"] = None,
    ) -> None:
        self._conversation_controller = conversation_controller
        self._context_compactor = context_compactor

    def get_context_metrics(self) -> Any:
        """Get current context metrics from the conversation controller."""
        return self._conversation_controller.get_context_metrics()

    async def check_context_overflow(self) -> bool:
        """Check whether the underlying conversation is at overflow risk."""
        return self._conversation_controller.check_context_overflow()

    async def compact_context(
        self,
        strategy: str = "tiered",
        min_messages: int = 6,
    ) -> int:
        """Compact context via the context compactor."""
        if self._context_compactor is None:
            if hasattr(self._conversation_controller, "compact_history"):
                return self._conversation_controller.compact_history(keep_recent=min_messages)
            return 0
        result = await self._context_compactor.check_and_compact()
        return result if isinstance(result, int) else 0

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        self._conversation_controller.add_message(role, content, **metadata)

    def add_messages(self, messages: List[Any]) -> None:
        """Add multiple messages to the conversation."""
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
                metadata = {
                    key: value for key, value in message.items() if key not in {"role", "content"}
                }
            else:
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
                metadata = {}
                for key in ("name", "tool_calls", "tool_call_id", "metadata"):
                    value = getattr(message, key, None)
                    if value is not None:
                        metadata[key] = value

            if role is None or content is None:
                continue
            self.add_message(role, content, **metadata)

    def get_messages(
        self,
        limit: Optional[int] = None,
        role: Optional[str] = None,
    ) -> List[Any]:
        """Get messages from conversation history."""
        messages = self._conversation_controller.messages
        if role:
            messages = [m for m in messages if getattr(m, "role", None) == role]
        if limit:
            messages = messages[-limit:]
        return messages

    def clear_messages(self, retain_system: bool = True) -> None:
        """Clear conversation messages while optionally preserving the system prompt."""
        self._conversation_controller.reset()
        if retain_system and hasattr(self._conversation_controller, "ensure_system_message"):
            self._conversation_controller.ensure_system_message()

    def get_max_tokens(self) -> int:
        """Return the approximate max token budget from controller configuration."""
        config = getattr(self._conversation_controller, "config", None)
        if config is None:
            return 0
        max_chars = getattr(config, "max_context_chars", 0)
        chars_per_token = max(getattr(config, "chars_per_token_estimate", 4), 1)
        return max_chars // chars_per_token

    def set_max_tokens(self, max_tokens: int) -> None:
        """Set the approximate max token budget on controller configuration."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")

        config = getattr(self._conversation_controller, "config", None)
        if config is None:
            return
        chars_per_token = max(getattr(config, "chars_per_token_estimate", 4), 1)
        config.max_context_chars = max_tokens * chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the controller configuration heuristic."""
        config = getattr(self._conversation_controller, "config", None)
        chars_per_token = max(getattr(config, "chars_per_token_estimate", 4), 1)
        return len(text) // chars_per_token

    def is_healthy(self) -> bool:
        """Check if the context service is healthy."""
        return self._conversation_controller is not None
