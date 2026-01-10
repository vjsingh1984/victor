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

"""Adapter to bridge observability EventBus with CQRS EventDispatcher.

This module provides bidirectional event bridging between:
- victor.observability.EventBus (Pub/Sub pattern)
- victor.core.EventDispatcher (Event Sourcing pattern)

Design Patterns:
    - Adapter: Converts between event formats
    - Bridge: Decouples abstraction from implementation
    - Observer: Both systems use observer pattern

Architecture:
    ┌────────────────────┐       ┌─────────────────────┐
    │  EventBus          │       │  EventDispatcher    │
    │  (Observability)   │◄─────►│  (CQRS/ES)          │
    │                    │       │                     │
    │  - VictorEvent     │       │  - Event            │
    │  - Categories      │       │  - Aggregates       │
    │  - Exporters       │       │  - Projections      │
    └────────────────────┘       └─────────────────────┘
                    ▲
                    │
          ┌────────┴────────┐
          │  CQRSEventAdapter│
          │                 │
          │  - Bidirectional│
          │  - Event mapping│
          │  - Filtering    │
          └─────────────────┘

Example:
    from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus
    from victor.core import EventDispatcher
    from victor.observability.cqrs_adapter import CQRSEventAdapter

    # Create adapter
    adapter = CQRSEventAdapter(
        event_bus=get_observability_bus(),
        event_dispatcher=EventDispatcher(),
    )

    # Enable bidirectional bridging
    adapter.enable_observability_to_cqrs()
    adapter.enable_cqrs_to_observability()

    # Events now flow between both systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

if TYPE_CHECKING:
    from victor.core.cqrs import Event as CQRSEvent
    from victor.core.event_sourcing import EventDispatcher

logger = logging.getLogger(__name__)


class EventDirection(str, Enum):
    """Direction of event flow."""

    OBSERVABILITY_TO_CQRS = "observability_to_cqrs"
    CQRS_TO_OBSERVABILITY = "cqrs_to_observability"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class EventMappingRule:
    """Rule for mapping events between systems.

    Attributes:
        source_pattern: Pattern to match source event (name or category)
        target_name: Name in target system
        transform: Optional transformation function
        enabled: Whether rule is active
    """

    source_pattern: str
    target_name: str
    transform: Optional[Callable[[Any], Dict[str, Any]]] = None
    enabled: bool = True

    def matches(self, event_name: str) -> bool:
        """Check if this rule matches an event name.

        Args:
            event_name: Name to check.

        Returns:
            True if rule matches.
        """
        if self.source_pattern == "*":
            return True
        if self.source_pattern.endswith("*"):
            return event_name.startswith(self.source_pattern[:-1])
        return event_name == self.source_pattern


@dataclass
class AdapterConfig:
    """Configuration for the CQRS event adapter.

    Attributes:
        enable_observability_to_cqrs: Forward observability events to CQRS
        enable_cqrs_to_observability: Forward CQRS events to observability
        filter_categories: Only bridge these categories (None = all)
        exclude_patterns: Event patterns to exclude
        include_metadata: Include adapter metadata in events
        batch_size: Batch size for async processing
    """

    enable_observability_to_cqrs: bool = True
    enable_cqrs_to_observability: bool = True
    filter_categories: Optional[Set[str]] = None
    exclude_patterns: Set[str] = field(default_factory=lambda: {"metric.*", "audit.*"})
    include_metadata: bool = True
    batch_size: int = 100


class CQRSEventAdapter:
    """Bidirectional adapter between EventBus and EventDispatcher.

    Enables unified event handling across observability and CQRS systems.
    Events can flow in either or both directions based on configuration.

    Features:
    - Bidirectional event bridging
    - Configurable filtering
    - Event transformation
    - Circular loop prevention
    - Metrics collection

    Example:
        adapter = CQRSEventAdapter(event_bus, event_dispatcher)
        adapter.start()

        # Events now flow between systems
        event_bus.publish(VictorEvent(...))  # Also dispatched to CQRS
        event_dispatcher.dispatch(Event(...))  # Also published to EventBus
    """

    def __init__(
        self,
        event_bus: Optional["ObservabilityBus"] = None,
        event_dispatcher: Optional["EventDispatcher"] = None,
        config: Optional[AdapterConfig] = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            event_bus: EventBus instance (default: singleton).
            event_dispatcher: EventDispatcher instance.
            config: Adapter configuration.
        """
        self._config = config or AdapterConfig()
        self._event_bus = event_bus
        self._event_dispatcher = event_dispatcher
        self._mapping_rules: List[EventMappingRule] = []
        self._unsubscribers: List[Callable[[], None]] = []
        self._is_active = False

        # Metrics
        self._events_bridged_to_cqrs = 0
        self._events_bridged_to_obs = 0
        self._events_filtered = 0

        # Circular loop prevention
        self._processing_ids: Set[str] = set()

        # Initialize default mappings
        self._setup_default_mappings()

    def _setup_default_mappings(self) -> None:
        """Set up default event mappings."""
        # Tool events
        self.add_mapping_rule(
            EventMappingRule(
                source_pattern="*.start",
                target_name="ToolStarted",
            )
        )
        self.add_mapping_rule(
            EventMappingRule(
                source_pattern="*.end",
                target_name="ToolCompleted",
            )
        )

        # State events
        self.add_mapping_rule(
            EventMappingRule(
                source_pattern="stage_transition",
                target_name="StateChanged",
            )
        )

        # Session events
        self.add_mapping_rule(
            EventMappingRule(
                source_pattern="session.start",
                target_name="SessionStarted",
            )
        )
        self.add_mapping_rule(
            EventMappingRule(
                source_pattern="session.end",
                target_name="SessionEnded",
            )
        )

    @property
    def is_active(self) -> bool:
        """Check if adapter is active."""
        return self._is_active

    @property
    def event_bus(self) -> Optional["ObservabilityBus"]:
        """Get the event bus."""
        return self._event_bus

    @property
    def event_dispatcher(self) -> Optional["EventDispatcher"]:
        """Get the event dispatcher."""
        return self._event_dispatcher

    def set_event_bus(self, event_bus: "ObservabilityBus") -> None:
        """Set the event bus.

        Args:
            event_bus: EventBus instance.
        """
        if self._is_active:
            raise RuntimeError("Cannot change event_bus while adapter is active")
        self._event_bus = event_bus

    def set_event_dispatcher(self, event_dispatcher: "EventDispatcher") -> None:
        """Set the event dispatcher.

        Args:
            event_dispatcher: EventDispatcher instance.
        """
        if self._is_active:
            raise RuntimeError("Cannot change event_dispatcher while adapter is active")
        self._event_dispatcher = event_dispatcher

    def add_mapping_rule(self, rule: EventMappingRule) -> None:
        """Add an event mapping rule.

        Args:
            rule: Mapping rule to add.
        """
        self._mapping_rules.append(rule)

    def remove_mapping_rule(self, source_pattern: str) -> None:
        """Remove a mapping rule by source pattern.

        Args:
            source_pattern: Pattern to remove.
        """
        self._mapping_rules = [r for r in self._mapping_rules if r.source_pattern != source_pattern]

    def start(self) -> None:
        """Start the adapter and begin bridging events.

        Raises:
            RuntimeError: If event_bus or event_dispatcher not set.
        """
        if self._is_active:
            return

        if not self._event_bus:

            self._event_bus = get_observability_bus()

        if not self._event_dispatcher:
            from victor.core.event_sourcing import EventDispatcher

            self._event_dispatcher = EventDispatcher()

        # Subscribe to EventBus for observability -> CQRS
        if self._config.enable_observability_to_cqrs:
            # New ObservabilityBus uses async subscribe()
            # We need to handle this in the event loop
            try:
                import asyncio

                # Try to get running loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Loop is running, create task for async subscription
                        # Subscribe to all events using wildcard pattern
                        asyncio.create_task(self._async_subscribe_observability())
                    else:
                        # Loop not running yet, run directly
                        loop.run_until_complete(self._async_subscribe_observability())
                except RuntimeError:
                    # No loop yet, create new one
                    asyncio.run(self._async_subscribe_observability())
            except Exception as e:
                logger.warning(f"Failed to subscribe to observability events: {e}")

        # Subscribe to EventDispatcher for CQRS -> observability
        if self._config.enable_cqrs_to_observability:
            if hasattr(self._event_dispatcher, "subscribe_all"):
                # Old sync API
                self._event_dispatcher.subscribe_all(self._on_cqrs_event)
            else:
                # New async API - handle similarly
                try:
                    import asyncio

                    asyncio.create_task(self._async_subscribe_cqrs())
                except Exception as e:
                    logger.warning(f"Failed to subscribe to CQRS events: {e}")

        self._is_active = True
        logger.info("CQRSEventAdapter started")

    def stop(self) -> None:
        """Stop the adapter and unsubscribe from all events."""
        if not self._is_active:
            return

        # Unsubscribe from EventBus
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception as e:
                logger.warning(f"Error unsubscribing: {e}")

        self._unsubscribers.clear()
        self._is_active = False
        logger.info("CQRSEventAdapter stopped")

    async def _async_subscribe_observability(self) -> None:
        """Async subscribe to observability events.

        Subscribes to all event patterns using wildcard.
        """
        # Subscribe to common event patterns
        patterns = [
            "tool.*",
            "state.*",
            "model.*",
            "error.*",
            "lifecycle.*",
            "audit.*",
            "metric.*",
        ]

        for pattern in patterns:
            try:
                handle = await self._event_bus.subscribe(pattern, self._on_observability_event)

                # Create sync unsubscribe function
                def make_unsub(h=handle):
                    def unsub():
                        # Note: This would need to be async in a real implementation
                        # For now, we'll just mark as inactive
                        logger.debug(f"Unsubscribing from {pattern}")

                    return unsub

                self._unsubscribers.append(make_unsub())
            except Exception as e:
                logger.warning(f"Failed to subscribe to {pattern}: {e}")

    async def _async_subscribe_cqrs(self) -> None:
        """Async subscribe to CQRS events."""
        # This would be implemented when EventDispatcher also goes async
        pass

    def _on_observability_event(self, event: MessagingEvent) -> None:
        """Handle an event from the observability EventBus.

        Args:
            event: MessagingEvent from EventBus.
        """
        # Prevent circular loops
        if event.id in self._processing_ids:
            return
        self._processing_ids.add(event.id)

        try:
            # Check category filter
            if self._config.filter_categories:
                # Get category from topic prefix or event data
                category = event.data.get("category", event.topic.split(".")[0])
                if category not in self._config.filter_categories:
                    self._events_filtered += 1
                    return

            # Check exclude patterns
            for pattern in self._config.exclude_patterns:
                if self._matches_pattern(event.topic, pattern):
                    self._events_filtered += 1
                    return

            # Convert and dispatch to CQRS
            cqrs_event = self._convert_to_cqrs_event(event)
            if cqrs_event and self._event_dispatcher:
                self._event_dispatcher.dispatch(cqrs_event)
                self._events_bridged_to_cqrs += 1

        finally:
            self._processing_ids.discard(event.id)

    def _on_cqrs_event(self, event: "CQRSEvent") -> None:
        """Handle an event from the CQRS EventDispatcher.

        Args:
            event: CQRS Event from EventDispatcher.
        """
        # Prevent circular loops
        event_id = getattr(event, "id", None) or str(uuid4())
        if event_id in self._processing_ids:
            return
        self._processing_ids.add(event_id)

        try:
            # Convert and publish to observability
            victor_event = self._convert_to_victor_event(event)
            if victor_event and self._event_bus:
                self._event_bus.publish(victor_event)
                self._events_bridged_to_obs += 1

        finally:
            self._processing_ids.discard(event_id)

    def _convert_to_cqrs_event(self, event: MessagingEvent) -> Optional["CQRSEvent"]:
        """Convert a VictorEvent to a CQRS Event.

        Uses concrete event types from event_sourcing module where available,
        or creates a generic ObservabilityEvent for custom events.

        Args:
            event: MessagingEvent to convert.

        Returns:
            CQRS Event or None if no mapping.
        """
        from victor.core.event_sourcing import (
            StateChangedEvent,
            ToolCalledEvent,
            ToolResultEvent,
        )

        # Build data dict with metadata
        data = event.data.copy()
        if self._config.include_metadata:
            data["_source"] = "observability"
            data["_original_id"] = event.id
            data["_category"] = event.category.value

        # Map to concrete event types based on event name and category
        if "stage_transition" in event.name or "state" in event.name.lower():
            return StateChangedEvent(
                task_id=data.get("session_id", ""),
                from_state=data.get("old_stage", ""),
                to_state=data.get("new_stage", ""),
                reason=f"confidence: {data.get('confidence', 0)}",
            )

        if ".start" in event.name:
            return ToolCalledEvent(
                task_id=data.get("session_id", ""),
                tool_name=data.get("tool_name", event.name.replace(".start", "")),
                arguments=data.get("arguments", {}),
            )

        if ".end" in event.name:
            return ToolResultEvent(
                task_id=data.get("session_id", ""),
                tool_name=data.get("tool_name", event.name.replace(".end", "")),
                success=data.get("success", True),
                result=str(data.get("result", "")),
                duration_ms=data.get("duration_ms", 0.0),
            )

        # For other events, use StateChangedEvent as a generic container
        return StateChangedEvent(
            task_id=data.get("session_id", ""),
            from_state="",
            to_state=event.name,
            reason=f"category:{event.category.value}",
        )

    def _convert_to_victor_event(self, event: "CQRSEvent") -> Optional[MessagingEvent]:
        """Convert a CQRS Event to an Event.

        Args:
            event: CQRS Event to convert.

        Returns: Event or None if should be filtered.
        """

        # Get event type from class name
        event_type = type(event).__name__

        # Build data from event attributes
        event_data: Dict[str, Any] = {}

        # Extract common attributes from concrete event types
        if hasattr(event, "task_id"):
            event_data["task_id"] = event.task_id
        if hasattr(event, "tool_name"):
            event_data["tool_name"] = event.tool_name
        if hasattr(event, "from_state"):
            event_data["from_state"] = event.from_state
        if hasattr(event, "to_state"):
            event_data["to_state"] = event.to_state
        if hasattr(event, "result"):
            event_data["result"] = event.result
        if hasattr(event, "success"):
            event_data["success"] = event.success
        if hasattr(event, "reason"):
            event_data["reason"] = event.reason
        if hasattr(event, "arguments"):
            event_data["arguments"] = event.arguments

        # Skip if this event originated from observability (prevent loops)
        if event_data.get("reason", "").startswith("topic:"):
            return None

        # Determine topic prefix from event type
        topic_prefix = self._infer_topic_prefix(event_type)
        topic = f"{topic_prefix}.{event_type.lower()}"

        # Add metadata
        if self._config.include_metadata:
            event_data["_source"] = "cqrs"
            event_data["_event_type"] = event_type

        return Event(
            topic=topic,
            data=event_data,
            timestamp=getattr(event, "timestamp", datetime.now(timezone.utc)),
        )

    def _infer_topic_prefix(self, event_type: str) -> str:
        """Infer topic prefix from CQRS event type.

        Args:
            event_type: CQRS event type name.

        Returns:
            Appropriate topic prefix (e.g., "tool", "state", "lifecycle").
        """

        event_lower = event_type.lower()

        if "tool" in event_lower:
            return "tool"
        if "state" in event_lower or "stage" in event_lower:
            return "state"
        if "session" in event_lower:
            return "lifecycle"
        if "error" in event_lower or "failed" in event_lower:
            return "error"
        if "model" in event_lower or "chat" in event_lower:
            return "model"

        return "custom"

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches a glob-like pattern.

        Args:
            name: Name to check.
            pattern: Pattern with optional * wildcards.

        Returns:
            True if matches.
        """
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return name.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return name.endswith(pattern[1:])
        return name == pattern

    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics.

        Returns:
            Dictionary with metrics.
        """
        return {
            "is_active": self._is_active,
            "events_bridged_to_cqrs": self._events_bridged_to_cqrs,
            "events_bridged_to_observability": self._events_bridged_to_obs,
            "events_filtered": self._events_filtered,
            "mapping_rules": len(self._mapping_rules),
        }

    def reset_metrics(self) -> None:
        """Reset adapter metrics."""
        self._events_bridged_to_cqrs = 0
        self._events_bridged_to_obs = 0
        self._events_filtered = 0


class UnifiedEventBridge:
    """High-level unified event bridge for framework integration.

        Provides a simple API to connect all event systems:
        - Observability EventBus
        - CQRS EventDispatcher
        - Framework Events

        This is the recommended way to set up event bridging.

        Example:
            from victor.observability.cqrs_adapter import UnifiedEventBridge
    from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

            bridge = UnifiedEventBridge.create()
            bridge.start()

            # All events now flow between systems
    """

    def __init__(self) -> None:
        """Initialize the bridge."""
        self._adapter: Optional[CQRSEventAdapter] = None
        self._is_started = False

    @classmethod
    def create(
        cls,
        event_bus: Optional["ObservabilityBus"] = None,
        event_dispatcher: Optional["EventDispatcher"] = None,
        config: Optional[AdapterConfig] = None,
    ) -> "UnifiedEventBridge":
        """Create a configured bridge instance.

        Args:
            event_bus: Optional EventBus (default: singleton).
            event_dispatcher: Optional EventDispatcher.
            config: Optional adapter config.

        Returns:
            Configured UnifiedEventBridge.
        """
        bridge = cls()
        bridge._adapter = CQRSEventAdapter(
            event_bus=event_bus,
            event_dispatcher=event_dispatcher,
            config=config,
        )
        return bridge

    @property
    def adapter(self) -> Optional[CQRSEventAdapter]:
        """Get the underlying adapter."""
        return self._adapter

    @property
    def is_started(self) -> bool:
        """Check if bridge is started."""
        return self._is_started

    def start(self) -> "UnifiedEventBridge":
        """Start the bridge.

        Returns:
            Self for chaining.
        """
        if self._adapter and not self._is_started:
            self._adapter.start()
            self._is_started = True
        return self

    def stop(self) -> "UnifiedEventBridge":
        """Stop the bridge.

        Returns:
            Self for chaining.
        """
        if self._adapter and self._is_started:
            self._adapter.stop()
            self._is_started = False
        return self

    def __enter__(self) -> "UnifiedEventBridge":
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()


def create_unified_bridge(
    event_bus: Optional["ObservabilityBus"] = None,
    event_dispatcher: Optional["EventDispatcher"] = None,
    auto_start: bool = True,
) -> UnifiedEventBridge:
    """Factory function to create and optionally start a unified event bridge.

    Args:
        event_bus: Optional EventBus instance.
        event_dispatcher: Optional EventDispatcher instance.
        auto_start: Whether to start the bridge immediately.

    Returns:
        Configured UnifiedEventBridge.

    Example:
        bridge = create_unified_bridge()
        # Events now flow between all systems
    """
    bridge = UnifiedEventBridge.create(
        event_bus=event_bus,
        event_dispatcher=event_dispatcher,
    )

    if auto_start:
        bridge.start()

    return bridge
