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

    async def compact_context(
        self,
        strategy: str = "tiered",
        min_messages: int = 6,
    ) -> int:
        """Compact context via the context compactor."""
        if self._context_compactor is None:
            return 0
        result = await self._context_compactor.check_and_compact()
        return result if isinstance(result, int) else 0

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        self._conversation_controller.add_message(role, content, **metadata)

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

    def is_healthy(self) -> bool:
        """Check if the context service is healthy."""
        return self._conversation_controller is not None
