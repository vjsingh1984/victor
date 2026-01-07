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

"""Tests for EventCategoryRegistry and related event system fixes (Workstream C)."""

from unittest.mock import MagicMock

import pytest

# Old event_bus imports removed - migration complete
from victor.observability.event_registry import (
    CustomEventCategory,
    EventCategoryRegistry,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    # EventBus.reset_instance()  # DELETED
    EventCategoryRegistry.reset_instance()
    yield
    # EventBus.reset_instance()  # DELETED
    EventCategoryRegistry.reset_instance()

    # =============================================================================
    # C.1: get_event_bus() Factory Function Tests
    # =============================================================================

    # # TODO: MIGRATION - EventBus tests removed
    # class TestGetEventBusFactory:
    """Tests for get_event_bus() factory function (C.1)."""

    def test_get_event_bus_returns_singleton(self):
        # """get_event_bus should return the EventBus singleton."""
        bus = get_event_bus()
        # assert bus is EventBus.get_instance()

    def test_get_event_bus_returns_same_instance(self):
        """Multiple calls should return the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_get_event_bus_is_functional(self):
        # """The returned EventBus should be fully functional."""
        bus = get_event_bus()
        handler = MagicMock()

        bus.subscribe(EventCategory.TOOL, handler)
        event = VictorEvent(category=EventCategory.TOOL, name="test_event")
        bus.publish(event)

        handler.assert_called_once()
        args = handler.call_args[0]
        assert args[0].name == "test_event"


# # END EventBus tests
# =============================================================================
# C.2: emit_lifecycle_event() Method Tests
# =============================================================================


class TestEmitLifecycleEvent:
    """Tests for emit_lifecycle_event() method (C.2)."""

    def test_emit_lifecycle_event_basic(self):
        """emit_lifecycle_event should publish LIFECYCLE category events."""
        bus = get_event_bus()
        handler = MagicMock()

        bus.subscribe(EventCategory.LIFECYCLE, handler)
        bus.emit_lifecycle_event("graph_started")

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.category == EventCategory.LIFECYCLE
        assert event.name == "graph_started"
        assert event.data == {}

    def test_emit_lifecycle_event_with_data(self):
        """emit_lifecycle_event should include provided data."""
        bus = get_event_bus()
        handler = MagicMock()

        bus.subscribe(EventCategory.LIFECYCLE, handler)
        bus.emit_lifecycle_event(
            "workflow_completed",
            data={"workflow_id": "deep_research", "duration_ms": 1234},
        )

        event = handler.call_args[0][0]
        assert event.name == "workflow_completed"
        assert event.data["workflow_id"] == "deep_research"
        assert event.data["duration_ms"] == 1234

    def test_emit_lifecycle_event_with_priority(self):
        """emit_lifecycle_event should respect priority parameter."""
        bus = get_event_bus()
        handler = MagicMock()

        bus.subscribe(EventCategory.LIFECYCLE, handler)
        bus.emit_lifecycle_event(
            "session_error",
            data={"error": "Connection lost"},
            priority=EventPriority.HIGH,
        )

        event = handler.call_args[0][0]
        assert event.priority == EventPriority.HIGH

    def test_emit_lifecycle_event_default_priority(self):
        """emit_lifecycle_event should default to NORMAL priority."""
        bus = get_event_bus()
        handler = MagicMock()

        bus.subscribe(EventCategory.LIFECYCLE, handler)
        bus.emit_lifecycle_event("session_started")

        event = handler.call_args[0][0]
        assert event.priority == EventPriority.NORMAL

    def test_emit_lifecycle_event_common_events(self):
        """Common lifecycle event types should work correctly."""
        bus = get_event_bus()
        events_received = []

        def capture_event(event):
            events_received.append(event.name)

        bus.subscribe(EventCategory.LIFECYCLE, capture_event)

        # Emit common lifecycle events
        bus.emit_lifecycle_event("graph_started", {"graph_id": "main"})
        bus.emit_lifecycle_event("workflow_completed", {"status": "success"})
        bus.emit_lifecycle_event("session_started", {"session_id": "abc123"})
        bus.emit_lifecycle_event("checkpoint_saved", {"thread_id": "t1"})

        assert events_received == [
            "graph_started",
            "workflow_completed",
            "session_started",
            "checkpoint_saved",
        ]


# =============================================================================
# C.3: CustomEventCategory Tests
# =============================================================================


class TestCustomEventCategory:
    """Tests for CustomEventCategory dataclass."""

    def test_custom_category_creation(self):
        """Should create CustomEventCategory with valid attributes."""
        category = CustomEventCategory(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )
        assert category.name == "security_audit"
        assert category.description == "Security audit events"
        assert category.registered_by == "victor.security"
        assert category.registered_at is not None

    def test_custom_category_immutable(self):
        """CustomEventCategory should be immutable (frozen dataclass)."""
        category = CustomEventCategory(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )
        with pytest.raises(AttributeError):
            category.name = "other_name"

    def test_custom_category_empty_name_raises(self):
        """Empty name should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CustomEventCategory(
                name="",
                description="Test",
                registered_by="test",
            )

    def test_custom_category_invalid_name_uppercase_raises(self):
        """Uppercase names should raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid lowercase identifier"):
            CustomEventCategory(
                name="SecurityAudit",
                description="Test",
                registered_by="test",
            )

    def test_custom_category_invalid_name_special_chars_raises(self):
        """Names with special characters should raise ValueError."""
        with pytest.raises(ValueError, match="must be a valid lowercase identifier"):
            CustomEventCategory(
                name="security-audit",
                description="Test",
                registered_by="test",
            )

    def test_custom_category_valid_names(self):
        """Valid lowercase identifier names should work."""
        valid_names = ["security_audit", "ml_pipeline", "custom123", "a", "test_category"]
        for name in valid_names:
            category = CustomEventCategory(
                name=name,
                description="Test",
                registered_by="test",
            )
            assert category.name == name


# =============================================================================
# C.3: EventCategoryRegistry Tests
# =============================================================================


class TestEventCategoryRegistry:
    """Tests for EventCategoryRegistry singleton."""

    def test_get_instance_returns_singleton(self):
        """get_instance should return singleton."""
        registry1 = EventCategoryRegistry.get_instance()
        registry2 = EventCategoryRegistry.get_instance()
        assert registry1 is registry2

    def test_register_custom_category(self):
        """Should register custom category."""
        registry = EventCategoryRegistry.get_instance()
        category = registry.register(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )

        assert category.name == "security_audit"
        assert registry.has_category("security_audit")

    def test_register_duplicate_same_source_idempotent(self):
        """Registering same category from same source should be idempotent."""
        registry = EventCategoryRegistry.get_instance()

        cat1 = registry.register(
            name="ml_pipeline",
            description="ML pipeline events",
            registered_by="victor.ml",
        )
        cat2 = registry.register(
            name="ml_pipeline",
            description="ML pipeline events v2",  # Different description
            registered_by="victor.ml",  # Same source
        )

        # Should return existing category
        assert cat1 is cat2
        assert registry.count() == 1

    def test_register_duplicate_different_source_raises(self):
        """Registering same category from different source should raise."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="ml_pipeline",
            description="ML pipeline events",
            registered_by="victor.ml",
        )

        with pytest.raises(ValueError, match="already registered by"):
            registry.register(
                name="ml_pipeline",
                description="Other events",
                registered_by="victor.other",
            )

    def test_register_builtin_name_raises(self):
        """Registering name that conflicts with built-in should raise."""
        registry = EventCategoryRegistry.get_instance()

        # "tool" is a built-in EventCategory
        with pytest.raises(ValueError, match="conflicts with built-in"):
            registry.register(
                name="tool",
                description="Custom tool events",
                registered_by="test",
            )

    def test_has_category_builtin(self):
        """has_category should return True for built-in categories."""
        registry = EventCategoryRegistry.get_instance()

        assert registry.has_category("tool") is True
        assert registry.has_category("state") is True
        assert registry.has_category("lifecycle") is True

    def test_has_category_custom(self):
        """has_category should return True for registered custom categories."""
        registry = EventCategoryRegistry.get_instance()

        assert registry.has_category("security_audit") is False

        registry.register(
            name="security_audit",
            description="Security audit events",
            registered_by="test",
        )

        assert registry.has_category("security_audit") is True

    def test_has_category_unknown(self):
        """has_category should return False for unknown categories."""
        registry = EventCategoryRegistry.get_instance()
        assert registry.has_category("nonexistent_category") is False

    def test_get_category(self):
        """get_category should return CustomEventCategory for custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security audit events",
            registered_by="victor.security",
        )

        category = registry.get_category("security_audit")
        assert category is not None
        assert category.name == "security_audit"

    def test_get_category_returns_none_for_builtin(self):
        """get_category should return None for built-in categories."""
        registry = EventCategoryRegistry.get_instance()
        assert registry.get_category("tool") is None

    def test_get_category_returns_none_for_unknown(self):
        """get_category should return None for unknown categories."""
        registry = EventCategoryRegistry.get_instance()
        assert registry.get_category("nonexistent") is None

    def test_list_custom(self):
        """list_custom should return only custom category names."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security",
            registered_by="test",
        )
        registry.register(
            name="ml_pipeline",
            description="ML",
            registered_by="test",
        )

        custom = registry.list_custom()
        assert custom == {"security_audit", "ml_pipeline"}

    def test_list_all(self):
        """list_all should return both built-in and custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security",
            registered_by="test",
        )

        all_cats = registry.list_all()

        # Should include built-in
        assert "tool" in all_cats
        assert "state" in all_cats
        assert "lifecycle" in all_cats

        # Should include custom
        assert "security_audit" in all_cats

    def test_get_all_custom(self):
        """get_all_custom should return dict of all custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="security_audit",
            description="Security",
            registered_by="test",
        )

        all_custom = registry.get_all_custom()
        assert "security_audit" in all_custom
        assert isinstance(all_custom["security_audit"], CustomEventCategory)

    def test_count(self):
        """count should return number of custom categories."""
        registry = EventCategoryRegistry.get_instance()

        assert registry.count() == 0

        registry.register(name="cat1", description="Test 1", registered_by="test")
        assert registry.count() == 1

        registry.register(name="cat2", description="Test 2", registered_by="test")
        assert registry.count() == 2

    def test_unregister_success(self):
        """unregister should remove category when called by original registrant."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="temp_category",
            description="Temporary",
            registered_by="test_module",
        )
        assert registry.has_category("temp_category") is True

        result = registry.unregister("temp_category", registered_by="test_module")

        assert result is True
        assert registry.has_category("temp_category") is False

    def test_unregister_wrong_source_raises(self):
        """unregister should raise when called by different registrant."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(
            name="protected_category",
            description="Protected",
            registered_by="original_module",
        )

        with pytest.raises(ValueError, match="was registered by"):
            registry.unregister("protected_category", registered_by="other_module")

    def test_unregister_unknown_returns_false(self):
        """unregister should return False for unknown category."""
        registry = EventCategoryRegistry.get_instance()
        result = registry.unregister("nonexistent", registered_by="test")
        assert result is False

    def test_reset_instance(self):
        """reset_instance should clear all custom categories."""
        registry = EventCategoryRegistry.get_instance()

        registry.register(name="cat1", description="Test", registered_by="test")
        registry.register(name="cat2", description="Test", registered_by="test")
        assert registry.count() == 2

        EventCategoryRegistry.reset_instance()

        # Same singleton instance but cleared
        assert registry.count() == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEventSystemIntegration:
    """Integration tests for the event system fixes."""

    def test_factory_with_lifecycle_event(self):
        """get_event_bus and emit_lifecycle_event should work together."""
        bus = get_event_bus()
        events = []

        bus.subscribe(EventCategory.LIFECYCLE, lambda e: events.append(e))

        bus.emit_lifecycle_event("workflow_started", {"id": "test_workflow"})
        bus.emit_lifecycle_event("workflow_completed", {"id": "test_workflow", "status": "success"})

        assert len(events) == 2
        assert events[0].name == "workflow_started"
        assert events[1].name == "workflow_completed"
        assert events[1].data["status"] == "success"

    def test_custom_category_with_event_bus(self):
        # """Custom categories should be usable with EventBus CUSTOM category."""
        bus = get_event_bus()
        registry = EventCategoryRegistry.get_instance()

        # Register custom category
        registry.register(
            name="security_scan",
            description="Security scanning events",
            registered_by="victor.security",
        )

        # Use CUSTOM category with custom_category field
        events = []
        bus.subscribe(EventCategory.CUSTOM, lambda e: events.append(e))

        bus.publish(
            VictorEvent(
                category=EventCategory.CUSTOM,
                name="vulnerability_found",
                data={
                    "custom_category": "security_scan",
                    "severity": "high",
                    "file": "config.py",
                },
            )
        )

        assert len(events) == 1
        assert events[0].data["custom_category"] == "security_scan"
        assert registry.has_category(events[0].data["custom_category"])
