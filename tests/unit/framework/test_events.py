"""Tests for custom event type support in framework events."""

from victor.framework.events import (
    AgentExecutionEvent,
    EventType,
    content_event,
    custom_event,
)


class TestCustomEventType:
    """Tests for CUSTOM event type and custom_event constructor."""

    def test_custom_event_type_exists(self):
        """EventType.CUSTOM should exist with value 'custom'."""
        assert EventType.CUSTOM.value == "custom"

    def test_custom_event_constructor(self):
        """custom_event() should create event with correct type, metadata, content."""
        event = custom_event("my.vertical.event", "payload")
        assert event.type == EventType.CUSTOM
        assert event.content == "payload"
        assert event.metadata["custom_type"] == "my.vertical.event"

    def test_is_custom_event_property(self):
        """is_custom_event should be True for CUSTOM, False for others."""
        custom = custom_event("test.event")
        assert custom.is_custom_event is True

        content = content_event("hello")
        assert content.is_custom_event is False

    def test_custom_event_to_dict(self):
        """Serialization should include custom_type in metadata."""
        event = custom_event("my.event", "data")
        d = event.to_dict()
        assert d["type"] == "custom"
        assert d["metadata"]["custom_type"] == "my.event"
        assert d["content"] == "data"

    def test_custom_event_with_extra_metadata(self):
        """User metadata should be preserved alongside custom_type."""
        event = custom_event(
            "my.event",
            "data",
            metadata={"source": "vertical-x", "priority": 1},
        )
        assert event.metadata["custom_type"] == "my.event"
        assert event.metadata["source"] == "vertical-x"
        assert event.metadata["priority"] == 1

    def test_custom_event_default_content(self):
        """Content should default to empty string."""
        event = custom_event("my.event")
        assert event.content == ""

    def test_custom_event_is_not_other_categories(self):
        """Custom events should not match other event category checks."""
        event = custom_event("test.event")
        assert event.is_tool_event is False
        assert event.is_content_event is False
        assert event.is_error_event is False
        assert event.is_lifecycle_event is False
