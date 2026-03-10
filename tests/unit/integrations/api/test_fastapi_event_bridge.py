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

import asyncio
import json
import time
from unittest.mock import AsyncMock

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
        """Test that EventBridge can be created with ObservabilityBus."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        assert bridge is not None
        assert bridge._event_bus == bus

    @pytest.mark.asyncio
    async def test_event_bridge_start_stop(self):
        """Test EventBridge start and stop lifecycle."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        bridge.start()
        assert bridge._running is True

        bridge.stop()
        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_event_bridge_broadcaster_exists(self):
        """Test that EventBridge has a broadcaster."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        assert hasattr(bridge, "_broadcaster")


class TestEventBusAdapterCompatibility:
    """Compatibility tests for sync and async EventBus APIs.

    M1: Async subscribe is now the primary and only supported path.
    Sync subscribe is no longer supported - use ObservabilityBus or other
    async-compatible event bus.
    """

    @pytest.mark.asyncio
    async def test_event_bus_adapter_supports_async_subscribe(self):
        """Adapter should subscribe and unsubscribe using async handles."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class FakeHandle:
            def __init__(self, pattern: str):
                self.pattern = pattern
                self.unsubscribed = False

            async def unsubscribe(self):
                self.unsubscribed = True

        class FakeAsyncBus:
            def __init__(self):
                self.subscriptions = []
                self.handles = []

            async def subscribe(self, pattern, handler):
                handle = FakeHandle(pattern)
                self.subscriptions.append((pattern, handler))
                self.handles.append(handle)
                return handle

        event_bus = FakeAsyncBus()
        adapter = EventBusAdapter()

        adapter.connect(event_bus)
        await asyncio.sleep(0.01)

        assert len(event_bus.subscriptions) == len(adapter.EVENT_MAPPING)
        assert len(adapter._subscriptions) == len(adapter.EVENT_MAPPING)
        assert all(asyncio.iscoroutinefunction(handler) for _, handler in event_bus.subscriptions)

        adapter.disconnect()
        await asyncio.sleep(0.01)

        assert all(handle.unsubscribed for handle in event_bus.handles)

    @pytest.mark.asyncio
    async def test_event_bus_adapter_rejects_sync_subscribe_m1(self):
        """M1: Adapter should reject event buses with sync subscribe (not supported)."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class FakeSyncBus:
            def __init__(self):
                self.subscribe_calls = []

            def subscribe(self, pattern, handler):
                self.subscribe_calls.append((pattern, handler))
                return None

        event_bus = FakeSyncBus()
        adapter = EventBusAdapter()

        # The sync connect() should fail to schedule the operation
        # and the async connect_async() should raise RuntimeError
        with pytest.raises(RuntimeError, match="async subscribe"):
            await adapter.connect_async(event_bus)


class TestEventBridgeReliability:
    """Reliability checks for bridge event delivery."""

    @pytest.mark.asyncio
    async def test_event_bridge_burst_delivery_has_no_loss_or_reordering(self):
        """A burst of events should be delivered completely and in-order."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        received = []

        async def send_func(message: str):
            received.append(json.loads(message))

        async def wait_for(predicate, timeout: float = 2.0):
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if predicate():
                    return
                await asyncio.sleep(0.01)
            pytest.fail("Timed out waiting for event bridge condition")

        bridge.start()
        bridge._broadcaster.add_client("event-loss-check", send_func)

        await wait_for(lambda: bridge._broadcaster._running)
        await wait_for(
            lambda: backend.get_subscription_count() >= len(bridge._adapter.EVENT_MAPPING)
        )

        total_events = 25
        for idx in range(total_events):
            await bus.emit("tool.start", {"idx": idx})

        await wait_for(lambda: len(received) >= total_events)
        assert len(received) == total_events
        assert [msg["data"]["idx"] for msg in received] == list(range(total_events))

        bridge._broadcaster.remove_client("event-loss-check")
        bridge.stop()
        await wait_for(lambda: not bridge._broadcaster._running)

    def _reset_reliability_state(self, bridge) -> None:
        """Reset singleton broadcaster reliability state between tests."""
        bridge._broadcaster._clients.clear()
        bridge._broadcaster._dispatch_latency_ms_window.clear()
        bridge._broadcaster._client_send_success_count = 0
        bridge._broadcaster._client_send_failure_count = 0
        bridge._broadcaster._events_dispatched_count = 0
        bridge._broadcaster._last_slo_breach_log_ts = 0.0

    @pytest.mark.asyncio
    async def test_reliability_dashboard_defaults_before_delivery(self):
        """Dashboard should expose healthy defaults before any send attempts."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        self._reset_reliability_state(bridge)

        dashboard = bridge.get_reliability_dashboard_data()

        assert dashboard["events_dispatched"] == 0
        assert dashboard["total_send_attempts"] == 0
        assert dashboard["send_successes"] == 0
        assert dashboard["send_failures"] == 0
        assert dashboard["delivery_success_rate"] == 1.0
        assert dashboard["dispatch_latency_p95_ms"] == 0.0
        assert dashboard["slo_status"]["delivery_success_rate"] is True
        assert dashboard["slo_status"]["dispatch_latency_p95_ms"] is True

    @pytest.mark.asyncio
    async def test_reliability_dashboard_tracks_delivery_and_slos(self):
        """Dashboard should track success/failure counts, p95 latency, and SLO status."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import BridgeEvent, BridgeEventType, EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)
        self._reset_reliability_state(bridge)

        async def successful_send(_message: str) -> None:
            return None

        async def failing_send(_message: str) -> None:
            raise RuntimeError("network timeout")

        bridge._broadcaster.add_client("ok-client", successful_send)
        bridge._broadcaster.add_client("bad-client", failing_send)

        event = BridgeEvent(
            type=BridgeEventType.TOOL_START,
            data={"idx": 1},
            timestamp=time.time() - 0.35,
        )
        await bridge._broadcaster._send_to_clients(event)

        dashboard = bridge.get_reliability_dashboard_data()

        assert dashboard["events_dispatched"] == 1
        assert dashboard["total_send_attempts"] == 2
        assert dashboard["send_successes"] == 1
        assert dashboard["send_failures"] == 1
        assert dashboard["delivery_success_rate"] == 0.5
        assert dashboard["dispatch_latency_p95_ms"] > 200.0
        assert dashboard["slo_status"]["delivery_success_rate"] is False
        assert dashboard["slo_status"]["dispatch_latency_p95_ms"] is False

        # Failing clients are evicted after send errors.
        assert "bad-client" not in bridge._broadcaster._clients


class TestEventBridgeBroadcaster:
    """Tests for EventBridge broadcaster functionality."""

    @pytest.mark.asyncio
    async def test_broadcaster_add_client(self):
        """Test adding a client to the broadcaster."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
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
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
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


class TestEventBusAdapterAsyncPath:
    """M1 Tests for async subscribe path in EventBusAdapter."""

    @pytest.mark.asyncio
    async def test_connect_async_with_async_event_bus(self):
        """Test connect_async successfully subscribes to async event bus."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)

        # Verify subscriptions were created
        assert len(adapter._subscriptions) > 0
        assert len(adapter._subscription_handles) == len(adapter._subscriptions)

    @pytest.mark.asyncio
    async def test_connect_async_fails_without_subscribe_method(self):
        """Test connect_async raises error when event bus has no subscribe method."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class InvalidBus:
            pass

        bus = InvalidBus()
        adapter = EventBusAdapter()

        with pytest.raises(RuntimeError, match="no subscribe\\(\\) method"):
            await adapter.connect_async(bus)

    @pytest.mark.asyncio
    async def test_connect_async_fails_with_sync_subscribe(self):
        """Test connect_async raises error when subscribe is not async."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class SyncBus:
            def subscribe(self, pattern, handler):
                return None

        bus = SyncBus()
        adapter = EventBusAdapter()

        with pytest.raises(RuntimeError, match="async subscribe"):
            await adapter.connect_async(bus)

    @pytest.mark.asyncio
    async def test_connect_async_handles_partial_subscription_failures(self):
        """Test connect_async continues when some subscriptions fail."""
        from victor.integrations.api.event_bridge import EventBusAdapter

        class FlakyAsyncBus:
            def __init__(self):
                self.call_count = 0

            async def subscribe(self, pattern, handler):
                self.call_count += 1
                # Fail on first subscription
                if self.call_count == 1:
                    raise RuntimeError("First subscription failed")
                # Return mock handle for others
                from unittest.mock import Mock
                handle = Mock()
                handle.unsubscribe = AsyncMock()
                return handle

        bus = FlakyAsyncBus()
        adapter = EventBusAdapter()

        # Should not raise, should continue with other subscriptions
        await adapter.connect_async(bus)

        # Should have some subscriptions despite failures
        assert len(adapter._subscription_handles) > 0

    @pytest.mark.asyncio
    async def test_disconnect_async_properly_unsubscribes(self):
        """Test disconnect_async properly awaits all unsubscribe calls."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)
        initial_sub_count = len(adapter._subscription_handles)

        await adapter.disconnect_async()

        # All handles should be cleared
        assert len(adapter._subscription_handles) == 0
        assert len(adapter._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_disconnect_async_handles_disconnect_during_subscribe(self):
        """Test disconnect_async handles rapid connect/disconnect cycles."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)
        # Immediately disconnect
        await adapter.disconnect_async()

        # Should cleanly handle the rapid cycle
        assert len(adapter._subscription_handles) == 0

    @pytest.mark.asyncio
    async def test_disconnect_async_idempotent(self):
        """Test disconnect_async can be called multiple times safely."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        await adapter.connect_async(bus)
        await adapter.disconnect_async()
        # Second disconnect should be safe
        await adapter.disconnect_async()

        assert len(adapter._subscription_handles) == 0


class TestEventBridgeAsyncPath:
    """M1 Tests for async path in EventBridge."""

    @pytest.mark.asyncio
    async def test_async_start_stop_lifecycle(self):
        """Test async_start and async_stop lifecycle."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()
        assert bridge._running is True
        assert len(bridge._adapter._subscription_handles) > 0

        await bridge.async_stop()
        assert bridge._running is False
        assert len(bridge._adapter._subscription_handles) == 0

    @pytest.mark.asyncio
    async def test_async_start_idempotent(self):
        """Test async_start can be called multiple times safely."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()
        await bridge.async_start()  # Second call should be no-op

        assert bridge._running is True

        await bridge.async_stop()

    @pytest.mark.asyncio
    async def test_async_stop_idempotent(self):
        """Test async_stop can be called multiple times safely."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()
        await bridge.async_stop()
        await bridge.async_stop()  # Second call should be no-op

        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_async_start_without_event_bus(self):
        """Test async_start works without an event bus (broadcast only)."""
        from victor.integrations.api.event_bridge import EventBridge

        bridge = EventBridge(event_bus=None)

        await bridge.async_start()
        assert bridge._running is True

        await bridge.async_stop()
        assert bridge._running is False


class TestEventBridgeAsyncBackwardCompatibility:
    """M1 Tests for backward compatibility of sync methods."""

    @pytest.mark.asyncio
    async def test_sync_start_still_works(self):
        """Test sync start() still works for backward compatibility."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        # Sync start in async context should schedule the operation
        bridge.start()
        await asyncio.sleep(0.1)  # Let the scheduled operation complete

        # May or may not be running depending on timing
        # but should not raise an error
        await bridge.async_stop()

    @pytest.mark.asyncio
    async def test_sync_stop_still_works(self):
        """Test sync stop() still works for backward compatibility."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBridge

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        bridge = EventBridge(bus)

        await bridge.async_start()

        # Sync stop in async context should schedule the operation
        bridge.stop()
        await asyncio.sleep(0.1)  # Let the scheduled operation complete

    @pytest.mark.asyncio
    async def test_adapter_sync_connect_still_works(self):
        """Test adapter sync connect() still works for backward compatibility."""
        from victor.core.events import InMemoryEventBackend, ObservabilityBus
        from victor.integrations.api.event_bridge import EventBusAdapter

        backend = InMemoryEventBackend()
        bus = ObservabilityBus(backend=backend)
        adapter = EventBusAdapter()

        # Sync connect in async context should schedule the operation
        adapter.connect(bus)
        await asyncio.sleep(0.1)  # Let the scheduled operation complete

        # Clean up
        await adapter.disconnect_async()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
