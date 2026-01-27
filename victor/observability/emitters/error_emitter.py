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

"""Error event emitter for tracking errors.

SOLID Principles:
- SRP: Focused solely on error events
- OCP: Extensible via inheritance (can add custom error tracking)
- LSP: Substitutable with IErrorEventEmitter
- ISP: Implements focused interface
- DIP: Depends on ObservabilityBus abstraction, not concrete implementation

Migration Notes:
- Migrated from legacy EventBus to canonical core/events system
- Uses topic-based routing ("error.raised") instead of category-based
- Fully async API with sync wrappers for gradual migration
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

from victor.observability.emitters.base import IErrorEventEmitter
from victor.core.events import ObservabilityBus, SyncEventWrapper

logger = logging.getLogger(__name__)


class ErrorEventEmitter(IErrorEventEmitter):
    """Emits error events to ObservabilityBus.

    Thread-safe, performant error tracking.
    Handles errors gracefully to avoid cascading failures.

    Example:
        >>> emitter = ErrorEventEmitter()
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     emitter.error(e, recoverable=True, context={"component": "tool_executor"})

    Migration:
        This emitter now uses the canonical core/events system instead of
        the legacy observability/event_bus.py. Events use topic-based routing
        ("error.raised") instead of category-based routing.
    """

    def __init__(self, bus: Optional[ObservabilityBus] = None):
        """Initialize the error event emitter.

        Args:
            bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self._bus = bus
        self._sync_wrapper: Optional[SyncEventWrapper] = None
        self._enabled = True

    def _get_bus(self) -> Optional[ObservabilityBus]:
        """Get ObservabilityBus instance.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        if self._bus:
            return self._bus

        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def _get_sync_wrapper(self) -> Optional[SyncEventWrapper]:
        """Get sync wrapper for gradual migration.

        Returns:
            SyncEventWrapper instance or None if unavailable
        """
        if self._sync_wrapper:
            return self._sync_wrapper

        bus = self._get_bus()
        if bus:
            self._sync_wrapper = SyncEventWrapper(bus.backend)
            return self._sync_wrapper

        return None

    async def emit_async(
        self,
        topic: str,
        data: Dict[str, Any],
    ) -> bool:
        """Emit an error event asynchronously.

        Args:
            topic: Event topic (e.g., "error.raised")
            data: Event payload

        Returns:
            True if emission succeeded, False otherwise
        """
        if not self._enabled:
            return False

        bus = self._get_bus()
        if bus:
            try:
                # Add category metadata for observability features
                data_with_category = {
                    **data,
                    "category": "error",
                }
                return await bus.emit(topic, data_with_category)
            except Exception as e:
                logger.debug(f"Failed to emit error event: {e}")
                return False

        return False

    def emit(
        self,
        event: MessagingEvent,
    ) -> None:
        """Emit an error event synchronously (for gradual migration).

        Args:
            event: The error event to emit
        """
        try:
            # For backward compatibility, convert to topic/data format
            self._event_bus.emit(event.topic, event.data)  # type: ignore[attr-defined]
        except Exception as e:
            logger.debug(f"Failed to emit error event: {e}")

        try:
            from victor.core.events.emit_helper import emit_event_sync

            bus = self._get_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic=topic,
                    data=data,
                    source="ErrorEventEmitter",
                )
        except Exception as e:
            logger.debug(f"Failed to emit error event: {e}")

    async def error_async(
        self,
        error: Exception,
        recoverable: bool,
        context: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> bool:
        """Emit error event asynchronously.

        Args:
            error: The exception that occurred
            recoverable: Whether the error is recoverable
            context: Additional error context
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        # Get traceback as string
        traceback_str = traceback.format_exc()

        return await self.emit_async(
            topic="error.raised",
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "recoverable": recoverable,
                "context": context or {},
                "traceback": traceback_str[-2000:] if traceback_str else None,  # Last 2000 chars
                **metadata,
            },
        )

    def error(
        self,
        error: Exception,
        recoverable: bool,
        context: Optional[Dict[str, Any]] = None,
        **metadata: Any,
    ) -> None:
        """Emit error event (sync wrapper).

        Args:
            error: The exception that occurred
            recoverable: Whether the error is recoverable
            context: Additional error context
            **metadata: Additional metadata
        """
        # Get traceback as string
        traceback_str = traceback.format_exc()

        self.emit(
            topic="error.raised",
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "recoverable": recoverable,
                "context": context or {},
                "traceback": traceback_str[-2000:] if traceback_str else None,  # Last 2000 chars
                **metadata,
            },
        )

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if event emission is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled
