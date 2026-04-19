"""Recovery controller extracted from ChatCoordinator.

Handles provider errors, token overflow, and retry-with-fallback logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Actions the recovery controller can recommend."""

    RETRY = "retry"
    FALLBACK = "fallback"
    TRUNCATE = "truncate"
    ABORT = "abort"
    COMPACT = "compact"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    action: RecoveryAction
    success: bool = False
    response: Optional[dict] = None
    message: str = ""


class RecoveryController:
    """[DEPRECATED] Handles error recovery for provider interactions.

    This class is being superseded by RecoveryService as part of the
    state-passed architectural migration. It remains for backward
    compatibility with facade-driven components.
    """

    def __init__(self, max_retries: int = 3, fallback_providers: list[str] = None):
        self._max_retries = max_retries
        self._fallback_providers = fallback_providers or []
        self._retry_count = 0

    async def handle_provider_error(self, error: Exception) -> RecoveryAction:
        """Determine recovery action for a provider error.

        Args:
            error: The provider error.

        Returns:
            Recommended RecoveryAction.
        """
        error_str = str(error).lower()

        # Rate limiting
        if "rate" in error_str or "429" in error_str:
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                return RecoveryAction.RETRY
            return RecoveryAction.FALLBACK

        # Context too long
        if "context" in error_str or "token" in error_str or "length" in error_str:
            return RecoveryAction.TRUNCATE

        # Server errors
        if any(code in error_str for code in ("500", "502", "503")):
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                return RecoveryAction.RETRY
            return RecoveryAction.FALLBACK

        # Authentication errors - abort
        if "auth" in error_str or "401" in error_str or "403" in error_str:
            return RecoveryAction.ABORT

        return RecoveryAction.ABORT

    async def handle_token_overflow(self, usage: dict) -> RecoveryAction:
        """Handle token limit overflow.

        Args:
            usage: Token usage dict with 'total', 'limit' keys.

        Returns:
            Recommended RecoveryAction.
        """
        total = usage.get("total", 0)
        limit = usage.get("limit", float("inf"))

        if total > limit * 0.95:
            return RecoveryAction.COMPACT
        elif total > limit * 0.80:
            return RecoveryAction.TRUNCATE

        return RecoveryAction.RETRY

    async def retry_with_fallback(self, request: dict) -> Optional[dict]:
        """Attempt retry, falling back to alternative providers.

        Args:
            request: The original request dict.

        Returns:
            Response dict if successful, None otherwise.
        """
        # This is a placeholder for integration with the provider coordinator
        logger.info(
            f"Retry attempt {self._retry_count}/{self._max_retries} "
            f"with {len(self._fallback_providers)} fallbacks available"
        )
        return None

    def reset(self) -> None:
        """Reset retry state."""
        self._retry_count = 0
