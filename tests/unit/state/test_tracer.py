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

"""Unit tests for StateTracer."""

import pytest

from victor.state.tracer import StateTransition, StateTracer

# TODO: MIGRATION - from victor.observability.event_bus import EventBus, EventCategory  # DELETED


# =============================================================================
# Test StateTransition
# =============================================================================


class TestStateTransition:
    """Test StateTransition dataclass."""

    def test_create_transition(self):
        """Test creating a state transition."""
        transition = StateTransition(
            scope="workflow",
            key="test_key",
            old_value=None,
            new_value="test_value",
        )

        assert transition.scope == "workflow"
        assert transition.key == "test_key"
        assert transition.old_value is None
        assert transition.new_value == "test_value"
        assert transition.metadata == {}
        assert isinstance(transition.timestamp, float)

    def test_transition_with_metadata(self):
        """Test creating transition with metadata."""
        transition = StateTransition(
            scope="conversation",
            key="user_input",
            old_value="",
            new_value="Hello",
            metadata={"source": "user", "timestamp": 123456},
        )

        assert transition.metadata == {"source": "user", "timestamp": 123456}

    def test_transition_to_dict(self):
        """Test converting transition to dictionary."""
        transition = StateTransition(
            scope="team",
            key="coordinator",
            old_value=None,
            new_value="agent-1",
        )

        transition_dict = transition.to_dict()

        assert transition_dict["scope"] == "team"
        assert transition_dict["key"] == "coordinator"
        assert transition_dict["old_value"] is None
        assert transition_dict["new_value"] == "agent-1"
        assert "timestamp" in transition_dict
        assert "metadata" in transition_dict

    def test_transition_truncates_long_values(self):
        """Test that to_dict truncates long values."""
        transition = StateTransition(
            scope="workflow",
            key="long_key",
            old_value="x" * 200,
            new_value="y" * 200,
        )

        transition_dict = transition.to_dict()

        # Values should be truncated to 100 chars
        assert len(transition_dict["old_value"]) == 100
        assert len(transition_dict["new_value"]) == 100


# =============================================================================
# Test StateTracer
# =============================================================================


class TestStateTracer:
    """Test StateTracer class."""

    def test_create_tracer(self):
        """Test creating a state tracer."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        assert tracer.get_transition_count() == 0
        assert tracer._transitions == []

    @pytest.mark.asyncio
    async def test_record_transition(self):
        """Test recording a state transition."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition(
            scope="workflow",
            key="test_key",
            old_value=None,
            new_value="test_value",
        )

        assert tracer.get_transition_count() == 1

        history = tracer.get_history()
        assert len(history) == 1
        assert history[0].scope == "workflow"
        assert history[0].key == "test_key"
        assert history[0].old_value is None
        assert history[0].new_value == "test_value"

    @pytest.mark.asyncio
    async def test_record_multiple_transitions(self):
        """Test recording multiple state transitions."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition("workflow", "key1", None, "value1")
        tracer.record_transition("conversation", "key2", "", "value2")
        tracer.record_transition("team", "key3", None, "value3")

        assert tracer.get_transition_count() == 3

    @pytest.mark.asyncio
    async def test_get_history_with_scope_filter(self):
        """Test get_history() filters by scope."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition("workflow", "key1", None, "value1")
        tracer.record_transition("conversation", "key2", None, "value2")
        tracer.record_transition("workflow", "key3", None, "value3")

        # Get all workflow transitions
        workflow_history = tracer.get_history(scope="workflow")
        assert len(workflow_history) == 2
        assert all(t.scope == "workflow" for t in workflow_history)

        # Get all conversation transitions
        conversation_history = tracer.get_history(scope="conversation")
        assert len(conversation_history) == 1
        assert conversation_history[0].scope == "conversation"

    @pytest.mark.asyncio
    async def test_get_history_with_key_filter(self):
        """Test get_history() filters by key."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition("workflow", "key1", None, "value1")
        tracer.record_transition("workflow", "key2", None, "value2")
        tracer.record_transition("workflow", "key1", "value1", "value1-updated")

        # Get all transitions for key1
        key1_history = tracer.get_history(key="key1")
        assert len(key1_history) == 2
        assert all(t.key == "key1" for t in key1_history)

        # Get all transitions for key2
        key2_history = tracer.get_history(key="key2")
        assert len(key2_history) == 1

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self):
        """Test get_history() respects limit parameter."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        for i in range(10):
            tracer.record_transition("workflow", f"key{i}", None, f"value{i}")

        # Get only 5 most recent
        history = tracer.get_history(limit=5)
        assert len(history) == 5

        # Should be the last 5 entries
        assert history[0].key == "key5"
        assert history[4].key == "key9"

    @pytest.mark.asyncio
    async def test_get_history_with_combined_filters(self):
        """Test get_history() with scope and key filters."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition("workflow", "key1", None, "value1")
        tracer.record_transition("conversation", "key1", None, "value2")
        tracer.record_transition("workflow", "key2", None, "value3")
        tracer.record_transition("workflow", "key1", "value1", "value4")

        # Get workflow transitions for key1 only
        history = tracer.get_history(scope="workflow", key="key1")
        assert len(history) == 2
        assert all(t.scope == "workflow" for t in history)
        assert all(t.key == "key1" for t in history)

    @pytest.mark.asyncio
    async def test_clear_history(self):
        """Test clearing transition history."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition("workflow", "key1", None, "value1")
        tracer.record_transition("workflow", "key2", None, "value2")

        assert tracer.get_transition_count() == 2

        tracer.clear_history()

        assert tracer.get_transition_count() == 0
        assert tracer.get_history() == []

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting transition statistics."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        # Empty statistics
        stats = tracer.get_statistics()
        assert stats["total"] == 0

        # Add some transitions
        tracer.record_transition("workflow", "key1", None, "value1")
        tracer.record_transition("workflow", "key2", None, "value2")
        tracer.record_transition("conversation", "key1", None, "value3")
        tracer.record_transition("team", "key3", None, "value4")

        stats = tracer.get_statistics()

        assert stats["total"] == 4
        assert stats["by_scope"]["workflow"] == 2
        assert stats["by_scope"]["conversation"] == 1
        assert stats["by_scope"]["team"] == 1
        assert stats["by_key"]["key1"] == 2
        assert stats["by_key"]["key2"] == 1
        assert stats["by_key"]["key3"] == 1
        assert "oldest_timestamp" in stats
        assert "newest_timestamp" in stats

    @pytest.mark.asyncio
    async def test_record_transition_with_metadata(self):
        """Test recording transition with metadata."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        tracer.record_transition(
            scope="workflow",
            key="test_key",
            old_value=None,
            new_value="test_value",
            source="user_input",
            user_id="user-123",
        )

        history = tracer.get_history()
        assert len(history) == 1
        assert history[0].metadata == {
            "source": "user_input",
            "user_id": "user-123",
        }


# =============================================================================
# Integration Tests
# =============================================================================


class TestStateTracerIntegration:
    """Integration tests for StateTracer."""

    @pytest.mark.asyncio
    async def test_tracer_event_bus_integration(self):
        """Test tracer integrates with EventBus."""
        event_bus = EventBus()
        tracer = StateTracer(event_bus)

        # Subscribe to STATE events
        received_events = []

        def on_event(event):
            if event.name == "state_transition":
                received_events.append(event.data)

        event_bus.subscribe(EventCategory.STATE, on_event)

        # Record transition
        tracer.record_transition(
            scope="workflow",
            key="test_key",
            old_value=None,
            new_value="test_value",
        )

        # Give event bus time to process
        import asyncio

        await asyncio.sleep(0.01)

        # Verify event was emitted
        assert len(received_events) >= 1
        # Find the state_transition event
        state_events = [e for e in received_events if "scope" in e]
        assert len(state_events) >= 1
        assert state_events[0]["scope"] == "workflow"
        assert state_events[0]["key"] == "test_key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
