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

"""Tests for EventBridge module.

TDD tests for real-time event streaming to VS Code and other clients.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestEventBridgeEventTypes:
    """Tests for EventBridge event type handling."""

    @pytest.mark.asyncio
    async def test_bridge_event_type_enum(self):
        """Test BridgeEventType enum values."""
        from victor.integrations.api.event_bridge import BridgeEventType

        # Tool events
        assert BridgeEventType.TOOL_START.value == "tool.start"
        assert BridgeEventType.TOOL_COMPLETE.value == "tool.complete"
        assert BridgeEventType.TOOL_ERROR.value == "tool.error"

        # File events
        assert BridgeEventType.FILE_CREATED.value == "file.created"
        assert BridgeEventType.FILE_MODIFIED.value == "file.modified"
        assert BridgeEventType.FILE_DELETED.value == "file.deleted"

        # Provider events
        assert BridgeEventType.PROVIDER_SWITCH.value == "provider.switch"
        assert BridgeEventType.PROVIDER_ERROR.value == "provider.error"

    @pytest.mark.asyncio
    async def test_bridge_event_serialization(self):
        """Test BridgeEvent serialization to JSON."""
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType

        event = BridgeEvent(
            type=BridgeEventType.TOOL_START, data={"tool_name": "read_file", "args": {}}
        )

        event_dict = event.to_dict()

        assert event_dict["type"] == "tool.start"
        assert event_dict["data"]["tool_name"] == "read_file"
        assert "id" in event_dict
        assert "timestamp" in event_dict

        # Test JSON serialization
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "tool.start"

    @pytest.mark.asyncio
    async def test_bridge_event_has_unique_ids(self):
        """Test that each BridgeEvent gets a unique ID."""
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType

        event1 = BridgeEvent(type=BridgeEventType.TOOL_START, data={})
        event2 = BridgeEvent(type=BridgeEventType.TOOL_START, data={})

        assert event1.id != event2.id

    @pytest.mark.asyncio
    async def test_bridge_event_has_timestamp(self):
        """Test that BridgeEvent has a timestamp."""
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType

        event = BridgeEvent(type=BridgeEventType.TOOL_START, data={})

        assert event.timestamp > 0


class TestEventBridgeIntegration:
    """Integration tests for EventBridge with EventBus."""

    @pytest.mark.asyncio
    async def test_event_bridge_import(self):
        """Test that EventBridge can be imported."""
        from victor.integrations.api.event_bridge import EventBridge

        assert EventBridge is not None

    @pytest.mark.asyncio
    async def test_event_bridge_creation(self):
        """Test that EventBridge can be created with EventBus."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.observability.event_bus import EventBus

        bus = EventBus()
        bridge = EventBridge(bus)

        assert bridge is not None
        assert bridge._event_bus == bus

    @pytest.mark.asyncio
    async def test_event_bridge_start_stop(self):
        """Test EventBridge start and stop lifecycle."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.observability.event_bus import EventBus

        bus = EventBus()
        bridge = EventBridge(bus)

        bridge.start()
        assert bridge._running is True

        bridge.stop()
        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_event_bridge_broadcaster_exists(self):
        """Test that EventBridge has a broadcaster."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.observability.event_bus import EventBus

        bus = EventBus()
        bridge = EventBridge(bus)

        assert hasattr(bridge, "_broadcaster")


class TestEventBridgeBroadcaster:
    """Tests for EventBridge broadcaster functionality."""

    @pytest.mark.asyncio
    async def test_broadcaster_add_client(self):
        """Test adding a client to the broadcaster."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.observability.event_bus import EventBus

        bus = EventBus()
        bridge = EventBridge(bus)
        bridge.start()

        # Mock send function
        send_func = AsyncMock()
        bridge._broadcaster.add_client("client-1", send_func)

        assert "client-1" in bridge._broadcaster._clients

        bridge.stop()

    @pytest.mark.asyncio
    async def test_broadcaster_remove_client(self):
        """Test removing a client from the broadcaster."""
        from victor.integrations.api.event_bridge import EventBridge
        from victor.observability.event_bus import EventBus

        bus = EventBus()
        bridge = EventBridge(bus)
        bridge.start()

        send_func = AsyncMock()
        bridge._broadcaster.add_client("client-1", send_func)
        bridge._broadcaster.remove_client("client-1")

        assert "client-1" not in bridge._broadcaster._clients

        bridge.stop()


class TestEventBridgeFiltering:
    """Tests for event filtering in EventBridge."""

    @pytest.mark.asyncio
    async def test_event_bridge_all_event_types(self):
        """Test that all expected event types exist."""
        from victor.integrations.api.event_bridge import BridgeEventType

        expected_types = [
            "TOOL_START",
            "TOOL_PROGRESS",
            "TOOL_COMPLETE",
            "TOOL_ERROR",
            "FILE_CREATED",
            "FILE_MODIFIED",
            "FILE_DELETED",
            "PROVIDER_SWITCH",
            "PROVIDER_ERROR",
            "PROVIDER_RECOVERY",
            "SESSION_START",
            "SESSION_END",
            "SESSION_ERROR",
            "METRICS_UPDATE",
            "BUDGET_WARNING",
            "BUDGET_EXHAUSTED",
            "NOTIFICATION",
            "ERROR",
        ]

        for type_name in expected_types:
            assert hasattr(BridgeEventType, type_name), f"Missing event type: {type_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
