"""Chat service adapter that wraps ChatCoordinator.

Implements ChatServiceProtocol by delegating to the existing
ChatCoordinator, enabling feature-flagged service layer migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from victor.agent.coordinators.chat_coordinator import ChatCoordinator
    from victor.providers.base import CompletionResponse, StreamChunk

logger = logging.getLogger(__name__)


class ChatServiceAdapter:
    """Adapts ChatCoordinator to ChatServiceProtocol.

    This adapter delegates all chat operations to the existing
    ChatCoordinator, providing a service-layer interface without
    changing behavior.
    """

    def __init__(self, chat_coordinator: "ChatCoordinator") -> None:
        self._chat_coordinator = chat_coordinator

    async def chat(
        self,
        user_message: str,
        *,
        stream: bool = False,
        **kwargs,
    ) -> "CompletionResponse":
        """Process a chat message via the coordinator."""
        return await self._chat_coordinator.chat(user_message)

    async def stream_chat(
        self,
        user_message: str,
        **kwargs,
    ) -> AsyncIterator["StreamChunk"]:
        """Stream chat response via the coordinator."""
        async for chunk in self._chat_coordinator.stream_chat(user_message):
            yield chunk

    async def chat_with_planning(
        self,
        user_message: str,
        use_planning: bool | None = None,
    ) -> "CompletionResponse":
        """Process a chat message with planning via the coordinator."""
        return await self._chat_coordinator.chat_with_planning(
            user_message, use_planning
        )

    def reset_conversation(self) -> None:
        """Reset conversation state."""
        if hasattr(self._chat_coordinator, "reset_conversation"):
            self._chat_coordinator.reset_conversation()

    def is_healthy(self) -> bool:
        """Check if the chat service is healthy."""
        return self._chat_coordinator is not None
