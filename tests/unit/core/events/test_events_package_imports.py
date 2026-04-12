"""Tests that victor.core.events package works after taxonomy removal."""


def test_observability_bus_importable():
    from victor.core.events import ObservabilityBus

    assert ObservabilityBus is not None


def test_messaging_event_importable():
    from victor.core.events import MessagingEvent

    assert MessagingEvent is not None


def test_agent_message_bus_importable():
    from victor.core.events import AgentMessageBus

    assert AgentMessageBus is not None


def test_event_backend_importable():
    from victor.core.events import InMemoryEventBackend, create_event_backend

    assert InMemoryEventBackend is not None
    assert create_event_backend is not None


def test_sync_wrappers_importable():
    from victor.core.events import SyncEventWrapper, SyncObservabilityBus

    assert SyncEventWrapper is not None


def test_emit_helper_importable():
    from victor.core.events import emit_event_sync

    assert emit_event_sync is not None


def test_adapter_importable():
    from victor.core.events import EventBusAdapter

    assert EventBusAdapter is not None


def test_taxonomy_not_in_public_api():
    """UnifiedEventType should no longer be in the events package public API."""
    import victor.core.events as events_pkg

    assert "UnifiedEventType" not in events_pkg.__all__
    assert "map_workflow_event" not in events_pkg.__all__
    assert "map_framework_event" not in events_pkg.__all__
    assert "emit_deprecation_warning" not in events_pkg.__all__


def test_framework_events_still_work():
    """Framework EventType (the canonical event system) is unaffected."""
    from victor.framework.events import EventType, AgentExecutionEvent

    assert EventType.CONTENT.value == "content"
    event = AgentExecutionEvent(type=EventType.CONTENT, content="test")
    assert event.content == "test"
