"""Session service adapter shim.

Provides a service-shaped compatibility wrapper that prefers the canonical
SessionService when available and falls back to the deprecated SessionCoordinator.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionServiceAdapter:
    """Compatibility shim that routes session calls to the service first."""

    def __init__(
        self,
        *args: Any,
        session_service: Optional[Any] = None,
        session_coordinator: Optional[Any] = None,
        deprecated_session_coordinator: Optional[Any] = None,
    ) -> None:
        if len(args) > 2:
            raise TypeError("SessionServiceAdapter accepts at most 2 positional arguments.")

        if args:
            warnings.warn(
                "Positional SessionServiceAdapter(...) construction is deprecated. "
                "Use session_service=... for canonical service wiring or "
                "deprecated_session_coordinator=... for explicit compatibility mode.",
                DeprecationWarning,
                stacklevel=2,
            )
            if session_service is not None:
                raise TypeError(
                    "Use either positional session_service or keyword session_service, not both."
                )
            session_service = args[0]
            if len(args) == 2:
                if session_coordinator is not None or deprecated_session_coordinator is not None:
                    raise TypeError(
                        "Use either positional session_coordinator or keyword compatibility "
                        "arguments, not both."
                    )
                session_coordinator = args[1]

        if session_coordinator is not None and deprecated_session_coordinator is not None:
            raise TypeError(
                "Use only one of session_coordinator or deprecated_session_coordinator."
            )

        resolved_deprecated_session_coordinator = deprecated_session_coordinator
        if session_coordinator is not None:
            warnings.warn(
                "SessionServiceAdapter(session_coordinator=...) is deprecated. "
                "Use deprecated_session_coordinator=... for explicit compatibility "
                "mode or prefer SessionService.",
                DeprecationWarning,
                stacklevel=2,
            )
            resolved_deprecated_session_coordinator = session_coordinator

        if session_service is None and resolved_deprecated_session_coordinator is not None:
            warnings.warn(
                "SessionServiceAdapter configured with a coordinator fallback only. "
                "This compatibility path is deprecated; prefer SessionService.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._session_service = session_service
        self._deprecated_session_coordinator = resolved_deprecated_session_coordinator

    def _delegate(self, method_name: str) -> Any:
        service = self._session_service
        if service is not None:
            method = getattr(service, method_name, None)
            if callable(method):
                return method

        coordinator = self._deprecated_session_coordinator
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
        return self._session_service is not None or self._deprecated_session_coordinator is not None
