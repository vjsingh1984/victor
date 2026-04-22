"""Session service adapter shim.

Provides a service-shaped compatibility wrapper that prefers the canonical
SessionService when available and falls back to the deprecated SessionCoordinator.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.agent.coordinators.session_coordinator import SessionCoordinator

logger = logging.getLogger(__name__)


class SessionServiceAdapter:
    """Compatibility shim that routes session calls to the service first."""

    def __init__(
        self,
        session_service: Optional[Any] = None,
        session_coordinator: Optional["SessionCoordinator"] = None,
    ) -> None:
        if session_service is None and session_coordinator is not None:
            warnings.warn(
                "SessionServiceAdapter configured with a coordinator fallback only. "
                "This compatibility path is deprecated; prefer SessionService.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._session_service = session_service
        self._session_coordinator = session_coordinator

    def _delegate(self, method_name: str) -> Any:
        service = self._session_service
        if service is not None:
            method = getattr(service, method_name, None)
            if callable(method):
                return method

        coordinator = self._session_coordinator
        if coordinator is not None:
            method = getattr(coordinator, method_name, None)
            if callable(method):
                return method

        raise AttributeError(f"Session service adapter has no delegate for {method_name}")

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation sessions."""
        return self._delegate("get_recent_sessions")(limit)

    def recover_session(self, session_id: str) -> bool:
        """Recover a previous conversation session."""
        return self._delegate("recover_session")(session_id)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self._delegate("get_session_stats")()

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
        messages: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get memory context for the current session."""
        return self._delegate("get_memory_context")(
            max_tokens=max_tokens,
            messages=messages,
        )

    async def save_checkpoint(
        self,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Save a conversation checkpoint."""
        return await self._delegate("save_checkpoint")(description, tags)

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint."""
        return await self._delegate("restore_checkpoint")(checkpoint_id)

    async def maybe_auto_checkpoint(self) -> Optional[str]:
        """Trigger auto-checkpoint if interval threshold is met."""
        return await self._delegate("maybe_auto_checkpoint")()

    def is_healthy(self) -> bool:
        """Check if the session service is healthy."""
        return self._session_service is not None or self._session_coordinator is not None
