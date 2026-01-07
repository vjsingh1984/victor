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

"""Tests for event system adapters.

Run with: pytest tests/unit/core/events/test_adapters.py -v
"""

from datetime import datetime, timezone

import pytest

from victor.core.events import Event
from victor.core.events.adapter import (
    victor_event_to_event,
    event_to_victor_event,
)

# Old event_bus imports removed - migration complete


@pytest.mark.unit
class TestEventConversion:
    """Tests for event format conversion."""

    def test_victor_event_to_event(self):
        """Should convert VictorEvent to Event correctly."""
        victor_event = VictorEvent(
            id="test123",
            timestamp=datetime(2025, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
            category=EventCategory.TOOL,
            name="call",
            data={"tool_name": "read", "args": {"file": "test.txt"}},
            trace_id="trace_abc",
            source="agent_1",
            priority=EventPriority.NORMAL,
        )

        event = victor_event_to_event(victor_event)

        assert event.id == "test123"
        assert event.topic == "tool.call"
        assert event.data["tool_name"] == "read"
        assert event.correlation_id == "trace_abc"
        assert event.source == "agent_1"

    def test_event_to_victor_event(self):
        """Should convert Event to VictorEvent correctly."""
        event = Event(
            id="event456",
            topic="metric.latency",
            data={"value": 42.5, "unit": "ms"},
            timestamp=1704456000.0,  # Fixed timestamp
            source="test",
            correlation_id="corr_xyz",
        )

        victor_event = event_to_victor_event(event)

        assert victor_event.id == "event456"
        assert victor_event.category == EventCategory.METRIC
        assert victor_event.name == "latency"
        assert victor_event.data["value"] == 42.5
        assert victor_event.trace_id == "corr_xyz"

    def test_roundtrip_conversion(self):
        """Roundtrip conversion should preserve key data."""
        original = VictorEvent(
            category=EventCategory.STATE,
            name="transition",
            data={"from": "a", "to": "b"},
            source="workflow",
        )

        # Convert to Event and back
        event = victor_event_to_event(original)
        restored = event_to_victor_event(event)

        assert restored.category == original.category
        assert restored.name == original.name
        assert restored.data == original.data
        assert restored.source == original.source

    def test_unknown_category_maps_to_custom(self):
        """Unknown categories should map to CUSTOM."""
        event = Event(
            topic="unknown_category.something",
            data={},
        )

        victor_event = event_to_victor_event(event)
        assert victor_event.category == EventCategory.CUSTOM

    def test_single_part_topic(self):
        """Single-part topic should use 'custom' category."""
        event = Event(
            topic="simple_topic",
            data={},
        )

        victor_event = event_to_victor_event(event)
        assert victor_event.category == EventCategory.CUSTOM
        assert victor_event.name == "simple_topic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
