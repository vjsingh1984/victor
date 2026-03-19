"""Session service adapter that wraps SessionCoordinator.

Implements SessionServiceProtocol by delegating to the existing
SessionCoordinator, enabling feature-flagged service layer migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.coordinators.session_coordinator import SessionCoordinator

logger = logging.getLogger(__name__)


class SessionServiceAdapter:
    """Adapts SessionCoordinator to SessionServiceProtocol.

    This adapter delegates all session operations to the existing
    SessionCoordinator, providing a service-layer interface without
    changing behavior.
    """

    def __init__(self, session_coordinator: "SessionCoordinator") -> None:
        self._session_coordinator = session_coordinator

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions."""
        return self._session_coordinator.get_recent_sessions(limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session."""
        return self._session_coordinator.recover_session(session_id)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self._session_coordinator.get_session_stats()

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get memory context for the current session."""
        return self._session_coordinator.get_memory_context(
            max_tokens=max_tokens,
            messages=messages,
        )

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save a conversation checkpoint."""
        return await self._session_coordinator.save_checkpoint(description, tags)

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint."""
        return await self._session_coordinator.restore_checkpoint(checkpoint_id)

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met."""
        return await self._session_coordinator.maybe_auto_checkpoint()

    def is_healthy(self) -> bool:
        """Check if the session service is healthy."""
        return self._session_coordinator is not None
