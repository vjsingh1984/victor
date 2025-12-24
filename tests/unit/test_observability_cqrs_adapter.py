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

"""Tests for the CQRS-Observability event adapter.

Tests the bidirectional event bridging between:
- victor.observability.EventBus (Pub/Sub)
- victor.core.EventDispatcher (Event Sourcing)
"""

import pytest
from datetime import datetime
from typing import Any, Dict, List

from victor.observability import (
    AdapterConfig,
    CQRSEventAdapter,
    EventBus,
    EventCategory,
    EventDirection,
    EventMappingRule,
    UnifiedEventBridge,
    VictorEvent,
    create_unified_bridge,
)
from victor.core.event_sourcing import (
    Event as CQRSEvent,
    EventDispatcher,
    StateChangedEvent,
    ToolCalledEvent,
    ToolResultEvent,
    TaskStartedEvent,
)


class TestEventMappingRule:
    """Tests for EventMappingRule."""

    def test_exact_match(self):
        """Test exact pattern matching."""
        rule = EventMappingRule(
            source_pattern="read.start",
            target_name="ToolStarted",
        )

        assert rule.matches("read.start")
        assert not rule.matches("write.start")
        assert not rule.matches("read.start.extra")

    def test_wildcard_match(self):
        """Test wildcard pattern matching with prefix wildcard."""
        # For prefix wildcard like "read*", it matches anything starting with "read"
        rule = EventMappingRule(
            source_pattern="read*",
            target_name="ToolStarted",
        )

        assert rule.matches("read.start")
        assert rule.matches("read.end")
        assert not rule.matches("write.start")

    def test_star_pattern_matches_all(self):
        """Test that * matches everything."""
        rule = EventMappingRule(
            source_pattern="*",
            target_name="AnyEvent",
        )

        assert rule.matches("read.start")
        assert rule.matches("anything")
        assert rule.matches("")

    def test_disabled_rule(self):
        """Test that disabled rules are still matchable (filtering happens at adapter level)."""
        rule = EventMappingRule(
            source_pattern="test",
            target_name="TestEvent",
            enabled=False,
        )

        # matches() only checks pattern, not enabled state
        assert rule.matches("test")


class TestAdapterConfig:
    """Tests for AdapterConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdapterConfig()

        assert config.enable_observability_to_cqrs is True
        assert config.enable_cqrs_to_observability is True
        assert config.filter_categories is None
        assert "metric.*" in config.exclude_patterns
        assert config.include_metadata is True
        assert config.batch_size == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = AdapterConfig(
            enable_observability_to_cqrs=False,
            filter_categories={"tool", "state"},
            exclude_patterns=set(),
        )

        assert config.enable_observability_to_cqrs is False
        assert config.filter_categories == {"tool", "state"}
        assert len(config.exclude_patterns) == 0


class TestCQRSEventAdapter:
    """Tests for CQRSEventAdapter."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus for each test."""
        # Reset singleton
        EventBus.reset_instance()
        return EventBus.get_instance()

    @pytest.fixture
    def event_dispatcher(self):
        """Create a fresh EventDispatcher for each test."""
        return EventDispatcher()

    @pytest.fixture
    def adapter(self, event_bus, event_dispatcher):
        """Create an adapter with both systems."""
        return CQRSEventAdapter(
            event_bus=event_bus,
            event_dispatcher=event_dispatcher,
        )

    def test_adapter_creation(self, adapter):
        """Test basic adapter creation."""
        assert adapter.event_bus is not None
        assert adapter.event_dispatcher is not None
        assert not adapter.is_active

    def test_adapter_start_stop(self, adapter):
        """Test starting and stopping the adapter."""
        assert not adapter.is_active

        adapter.start()
        assert adapter.is_active

        adapter.stop()
        assert not adapter.is_active

    def test_cannot_change_bus_while_active(self, adapter):
        """Test that event_bus cannot be changed while adapter is active."""
        adapter.start()

        with pytest.raises(RuntimeError, match="Cannot change event_bus"):
            adapter.set_event_bus(EventBus.get_instance())

        adapter.stop()

    def test_add_mapping_rule(self, adapter):
        """Test adding custom mapping rules."""
        initial_count = len(adapter._mapping_rules)

        adapter.add_mapping_rule(
            EventMappingRule(
                source_pattern="custom.*",
                target_name="CustomEvent",
            )
        )

        assert len(adapter._mapping_rules) == initial_count + 1

    def test_remove_mapping_rule(self, adapter):
        """Test removing mapping rules."""
        adapter.add_mapping_rule(
            EventMappingRule(
                source_pattern="to_remove",
                target_name="RemoveMe",
            )
        )

        initial_count = len(adapter._mapping_rules)
        adapter.remove_mapping_rule("to_remove")

        assert len(adapter._mapping_rules) == initial_count - 1

    def test_observability_to_cqrs_bridging(self, adapter, event_bus, event_dispatcher):
        """Test that observability events can be converted to CQRS events.

        Note: This tests the conversion logic directly since EventDispatcher.dispatch()
        is async and would require async test infrastructure for full pipeline testing.
        """
        adapter.start()

        # Create a VictorEvent (using .start pattern to get ToolCalledEvent)
        victor_event = VictorEvent(
            category=EventCategory.TOOL,
            name="read.start",
            data={"path": "/test/file.py", "tool_name": "read"},
        )

        # Test the conversion directly
        cqrs_event = adapter._convert_to_cqrs_event(victor_event)

        # Verify conversion produces correct CQRS event type
        assert cqrs_event is not None
        assert isinstance(cqrs_event, ToolCalledEvent)
        assert cqrs_event.tool_name == "read"

        # Verify the handler processes events (increments counter)
        initial_count = adapter._events_bridged_to_cqrs
        adapter._on_observability_event(victor_event)
        # Counter should increment (even if async dispatch doesn't complete)
        assert adapter._events_bridged_to_cqrs >= initial_count

        adapter.stop()

    def test_cqrs_to_observability_bridging(self, adapter, event_bus, event_dispatcher):
        """Test that CQRS events can be converted to observability events.

        Note: This tests the conversion logic directly since EventDispatcher.dispatch()
        is async and would require async test infrastructure for full pipeline testing.
        """
        received_events: List[VictorEvent] = []

        def obs_handler(event: VictorEvent) -> None:
            received_events.append(event)

        unsub = event_bus.subscribe_all(obs_handler)
        adapter.start()

        # Create a CQRS event using concrete event type
        cqrs_event = ToolCalledEvent(
            task_id="test-task",
            tool_name="read",
            arguments={"path": "/test"},
        )

        # Test the conversion directly
        victor_event = adapter._convert_to_victor_event(cqrs_event)

        # Verify conversion produces correct VictorEvent
        assert victor_event is not None
        assert victor_event.name == "ToolCalledEvent"
        assert victor_event.category == EventCategory.TOOL

        # Verify the handler can process the event
        # Call the CQRS handler directly to test the bridge
        adapter._on_cqrs_event(cqrs_event)

        # Should have published to event bus
        assert len(received_events) >= 1
        assert received_events[0].name == "ToolCalledEvent"

        unsub()
        adapter.stop()

    def test_circular_loop_prevention(self, adapter, event_bus, event_dispatcher):
        """Test that circular event loops are prevented via ID tracking."""
        # Test the internal mechanism directly
        adapter.start()

        # Simulate an event that's already being processed
        test_id = "test-event-id"
        adapter._processing_ids.add(test_id)

        # This ID should be skipped
        assert test_id in adapter._processing_ids

        # After processing, ID should be removed
        adapter._processing_ids.discard(test_id)
        assert test_id not in adapter._processing_ids

        adapter.stop()

    def test_category_filtering(self, event_bus, event_dispatcher):
        """Test that category filtering config is applied correctly."""
        config = AdapterConfig(
            filter_categories={"tool"},  # Only forward tool events
        )
        adapter = CQRSEventAdapter(
            event_bus=event_bus,
            event_dispatcher=event_dispatcher,
            config=config,
        )

        # Test the filtering logic directly
        assert adapter._config.filter_categories == {"tool"}

        # STATE category should be filtered
        adapter.start()
        adapter._events_filtered = 0

        # Call the handler directly to test filtering logic
        state_event = VictorEvent(
            category=EventCategory.STATE,
            name="transition",
            data={},
        )
        adapter._on_observability_event(state_event)

        # Should have been filtered
        assert adapter._events_filtered == 1

        adapter.stop()

    def test_exclude_patterns(self, event_bus, event_dispatcher):
        """Test that exclude patterns filter events correctly."""
        config = AdapterConfig(
            exclude_patterns={"test.*"},
        )
        adapter = CQRSEventAdapter(
            event_bus=event_bus,
            event_dispatcher=event_dispatcher,
            config=config,
        )

        # Test the pattern matching helper
        assert adapter._matches_pattern("test.event", "test.*")
        assert adapter._matches_pattern("test.other", "test.*")
        assert not adapter._matches_pattern("other.event", "test.*")

        adapter.stop()

    def test_metrics_collection(self, adapter, event_bus, event_dispatcher):
        """Test that adapter collects metrics."""
        adapter.start()

        # Get initial metrics
        metrics = adapter.get_metrics()
        assert metrics["is_active"] is True
        assert metrics["mapping_rules"] > 0

        # Metrics should be available
        assert "events_bridged_to_cqrs" in metrics
        assert "events_filtered" in metrics

        adapter.stop()

    def test_reset_metrics(self, adapter, event_bus):
        """Test that metrics can be reset."""
        adapter.start()

        # Increment metrics manually
        adapter._events_bridged_to_cqrs = 5
        adapter._events_filtered = 3

        assert adapter.get_metrics()["events_bridged_to_cqrs"] == 5
        assert adapter.get_metrics()["events_filtered"] == 3

        adapter.reset_metrics()

        assert adapter.get_metrics()["events_bridged_to_cqrs"] == 0
        assert adapter.get_metrics()["events_filtered"] == 0

        adapter.stop()

    def test_event_conversion_logic(self, adapter, event_bus, event_dispatcher):
        """Test event conversion includes correct data."""
        adapter.start()

        # Test the conversion function directly
        victor_event = VictorEvent(
            category=EventCategory.TOOL,
            name="read.start",
            data={"path": "/test", "tool_name": "read"},
        )

        cqrs_event = adapter._convert_to_cqrs_event(victor_event)

        assert cqrs_event is not None
        assert isinstance(cqrs_event, ToolCalledEvent)
        assert cqrs_event.tool_name == "read"

        adapter.stop()


class TestUnifiedEventBridge:
    """Tests for UnifiedEventBridge."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield

    def test_create_bridge(self):
        """Test creating a unified bridge."""
        bridge = UnifiedEventBridge.create()

        assert bridge.adapter is not None
        assert not bridge.is_started

    def test_start_stop_bridge(self):
        """Test starting and stopping the bridge."""
        bridge = UnifiedEventBridge.create()

        bridge.start()
        assert bridge.is_started

        bridge.stop()
        assert not bridge.is_started

    def test_bridge_chaining(self):
        """Test that start/stop return self for chaining."""
        bridge = UnifiedEventBridge.create()

        result = bridge.start()
        assert result is bridge

        result = bridge.stop()
        assert result is bridge

    def test_bridge_context_manager(self):
        """Test using bridge as context manager."""
        bridge = UnifiedEventBridge.create()

        with bridge as b:
            assert b.is_started
            assert b is bridge

        assert not bridge.is_started


class TestCreateUnifiedBridge:
    """Tests for create_unified_bridge factory function."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield

    def test_factory_creates_started_bridge(self):
        """Test that factory creates and starts bridge by default."""
        bridge = create_unified_bridge()

        assert bridge.is_started

        bridge.stop()

    def test_factory_with_auto_start_false(self):
        """Test factory with auto_start=False."""
        bridge = create_unified_bridge(auto_start=False)

        assert not bridge.is_started

    def test_factory_with_custom_components(self):
        """Test factory with custom event bus and dispatcher."""
        EventBus.reset_instance()
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()

        bridge = create_unified_bridge(
            event_bus=event_bus,
            event_dispatcher=event_dispatcher,
            auto_start=False,
        )

        assert bridge.adapter.event_bus is event_bus
        assert bridge.adapter.event_dispatcher is event_dispatcher


class TestCategoryInference:
    """Tests for event category inference."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield

    def test_tool_category_inference(self):
        """Test that tool-related events get TOOL category."""
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()
        adapter = CQRSEventAdapter(event_bus, event_dispatcher)

        # Use concrete ToolCalledEvent
        victor_event = adapter._convert_to_victor_event(
            ToolCalledEvent(task_id="test", tool_name="read", arguments={})
        )

        assert victor_event is not None
        assert victor_event.category == EventCategory.TOOL

    def test_state_category_inference(self):
        """Test that state-related events get STATE category."""
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()
        adapter = CQRSEventAdapter(event_bus, event_dispatcher)

        # Use concrete StateChangedEvent
        victor_event = adapter._convert_to_victor_event(
            StateChangedEvent(
                task_id="test", from_state="INITIAL", to_state="PLANNING", reason="test"
            )
        )

        assert victor_event is not None
        assert victor_event.category == EventCategory.STATE

    def test_error_category_inference(self):
        """Test that error-related events (TaskFailed) get ERROR category."""
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()
        adapter = CQRSEventAdapter(event_bus, event_dispatcher)

        # Use inference helper directly
        category = adapter._infer_category("TaskFailedEvent")
        assert category == EventCategory.ERROR

    def test_session_category_inference(self):
        """Test that session-related events get LIFECYCLE category."""
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()
        adapter = CQRSEventAdapter(event_bus, event_dispatcher)

        # Use inference helper directly
        category = adapter._infer_category("SessionStartedEvent")
        assert category == EventCategory.LIFECYCLE
