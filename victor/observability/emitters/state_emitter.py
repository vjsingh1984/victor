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

"""State event emitter for tracking state transitions.

SOLID Principles:
- SRP: Focused solely on state transition events
- OCP: Extensible via inheritance (can add custom state tracking)
- LSP: Substitutable with IStateEventEmitter
- ISP: Implements focused interface
- DIP: Depends on ObservabilityBus abstraction, not concrete implementation

Migration Notes:
- Migrated from legacy EventBus to canonical core/events system
- Uses topic-based routing ("state.transition") instead of category-based
- Fully async API with sync wrappers for gradual migration
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Union

from victor.observability.emitters.base import IStateEventEmitter
from victor.core.events import ObservabilityBus, SyncEventWrapper

if TYPE_CHECKING:
    from victor.core.events import MessagingEvent

logger = logging.getLogger(__name__)


class StateEventEmitter(IStateEventEmitter):
    """Emits state transition events to ObservabilityBus.

    Thread-safe, performant state transition tracking.
    Handles errors gracefully to avoid impacting execution.

    Example:
        >>> emitter = StateEventEmitter()
        >>> emitter.state_transition("INITIAL", "PLANNING", 0.95, agent_id="agent-1")

    Migration:
        This emitter now uses the canonical core/events system instead of
        the legacy observability/event_bus.py. Events use topic-based routing
        ("state.transition") instead of category-based routing.
    """

    def __init__(self, bus: Optional[ObservabilityBus] = None):
        """Initialize the state event emitter.

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
        """Emit a state event asynchronously.

        Args:
            topic: Event topic (e.g., "state.transition")
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
                    "category": "state",
                }
                return await bus.emit(topic, data_with_category)
            except Exception as e:
                logger.debug(f"Failed to emit state event: {e}")
                return False

        return False

    def emit(
        self,
        event: Union["MessagingEvent", str],
        data: Optional[Dict[str, Any]] = None,
        *,
        topic: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        """Emit a state event synchronously (for gradual migration).

        This method wraps the async emit_async() method using emit_event_sync()
        to avoid asyncio.run() errors in running event loops.

        Args:
            event: Either a MessagingEvent object or a topic string (for backward compatibility)
            data: Event data (used with topic string or keyword arguments)
            topic: Alternative keyword argument for topic (supports emit(topic=..., data=...))
        """
        try:
            from victor.core.events.emit_helper import emit_event_sync

            # Support both MessagingEvent and backward-compatible (topic, data) form
            if topic is not None:
                final_topic = topic
                event_data = data or {}
            elif isinstance(event, str):
                final_topic = event
                event_data = data or {}
            else:
                # MessagingEvent form
                final_topic = event.topic if hasattr(event, 'topic') else "state.transition"
                event_data = event.data if hasattr(event, 'data') else {}

            bus = self._get_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic=final_topic,
                    data=event_data,
                    source="StateEventEmitter",
                )
        except Exception as e:
            logger.debug(f"Failed to emit state event: {e}")

    async def state_transition_async(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        **metadata: Any,
    ) -> bool:
        """Emit state transition event asynchronously.

        Args:
            old_stage: Previous state/stage
            new_stage: New state/stage
            confidence: Transition confidence (0.0 to 1.0)
            **metadata: Additional metadata

        Returns:
            True if emission succeeded
        """
        return await self.emit_async(
            topic="state.transition",
            data={
                "old_stage": old_stage,
                "new_stage": new_stage,
                "confidence": confidence,
                **metadata,
            },
        )

    def state_transition(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        **metadata: Any,
    ) -> None:
        """Emit state transition event (sync wrapper).

        Args:
            old_stage: Previous state/stage
            new_stage: New state/stage
            confidence: Transition confidence (0.0 to 1.0)
            **metadata: Additional metadata
        """
        self.emit(
            MessagingEvent(
                topic="state.transition",
                data={
                    "old_stage": old_stage,
                    "new_stage": new_stage,
                    "confidence": confidence,
                    **metadata,
                },
            )  # type: ignore[arg-type,call-arg]
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
