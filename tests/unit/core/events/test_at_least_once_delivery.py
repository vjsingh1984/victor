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

"""Tests for AT_LEAST_ONCE event delivery.

Tests verify that:
- Events are tracked until ACK'd
- Events are re-delivered on NACK with requeue=True
- Events are dropped after max retries
- AgentMessageBus uses AT_LEAST_ONCE by default

Run with: pytest tests/unit/core/events/test_at_least_once_delivery.py -v
"""

import asyncio

import pytest

from victor.core.events import (
    MessagingEvent,
    InMemoryEventBackend,
    BackendConfig,
    DeliveryGuarantee,
    AgentMessageBus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def at_least_once_backend():
    """Backend configured for AT_LEAST_ONCE delivery."""
    config = BackendConfig.for_agent_messaging()
    backend = InMemoryEventBackend(config=config)
    await backend.connect()
    yield backend
    await backend.disconnect()


@pytest.fixture
async def agent_bus():
    """Agent message bus with AT_LEAST_ONCE delivery."""
    bus = AgentMessageBus()
    await bus.connect()
    yield bus
    await bus.disconnect()


# =============================================================================
# MessagingEvent Ack/Nack Tests
# =============================================================================


@pytest.mark.unit
class TestMessagingEventAckNack:
    """Tests for event acknowledgement methods."""

    def test_event_initially_unacknowledged(self):
        """New events should not be acknowledged."""
        event = MessagingEvent(topic="test.event", data={})
        assert event.is_acknowledged() is False

    def test_ack_marks_event_as_acknowledged(self):
        """Calling ack() should mark event as acknowledged."""
        event = MessagingEvent(topic="test.event", data={})

        async def do_ack():
            await event.ack()

        asyncio.run(do_ack())
        assert event.is_acknowledged() is True

    def test_nack_without_requeue_drops_event(self):
        """NACK with requeue=False should mark as acknowledged (drop)."""
        event = MessagingEvent(topic="test.event", data={})

        async def do_nack():
            await event.nack(requeue=False)

        asyncio.run(do_nack())
        assert event.is_acknowledged() is True

    def test_nack_with_requeue_increments_delivery_count(self):
        """NACK with requeue=True should increment delivery count."""
        event = MessagingEvent(topic="test.event", data={})

        async def do_nack():
            await event.nack(requeue=True)

        asyncio.run(do_nack())
        assert event._delivery_count == 1
        assert event.is_acknowledged() is False

    def test_should_retry_based_on_delivery_count(self):
        """should_retry() should return True until max retries reached."""
        event = MessagingEvent(topic="test.event", data={})
        event._max_delivery_count = 3

        # Initially should retry
        assert event.should_retry() is True

        # After 2 deliveries, should still retry
        event._delivery_count = 2
        assert event.should_retry() is True

        # After 3 deliveries (max), should not retry
        event._delivery_count = 3
        assert event.should_retry() is False


# =============================================================================
# AT_LEAST_ONCE Delivery Tests
# =============================================================================


@pytest.mark.unit
class TestAtLeastOnceDelivery:
    """Tests for AT_LEAST_ONCE delivery behavior."""

    @pytest.mark.asyncio
    async def test_event_tracked_until_acked(self, at_least_once_backend):
        """Backend should track event until it's acknowledged."""
        received_events = []

        async def tracking_handler(event: MessagingEvent):
            received_events.append(event)
            # Acknowledge the event
            await event.ack()

        await at_least_once_backend.subscribe("test.*", tracking_handler)

        # Publish AT_LEAST_ONCE event
        event = MessagingEvent(
            topic="test.event",
            data={"message": "hello"},
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )
        await at_least_once_backend.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Event should be received
        assert len(received_events) == 1
        assert received_events[0].data["message"] == "hello"

        # Event should NOT be in pending (was ACK'd)
        assert event.id not in at_least_once_backend._pending_events

    @pytest.mark.asyncio
    async def test_event_redelivered_on_nack(self, at_least_once_backend):
        """Event should be re-delivered after NACK with requeue=True."""
        received_count = [0]

        async def failing_handler(event: MessagingEvent):
            received_count[0] += 1
            # Fail first time, succeed second time
            if received_count[0] == 1:
                await event.nack(requeue=True)
            else:
                await event.ack()

        await at_least_once_backend.subscribe("test.*", failing_handler)

        # Publish AT_LEAST_ONCE event
        event = MessagingEvent(
            topic="test.event",
            data={"attempt": 1},
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )
        await at_least_once_backend.publish(event)

        # Wait for processing and retry
        await asyncio.sleep(0.5)

        # Event should be received twice (original + retry)
        assert received_count[0] == 2
        # Should be ACK'd now
        assert event.is_acknowledged() is True

    @pytest.mark.asyncio
    async def test_event_dropped_after_max_retries(self, at_least_once_backend):
        """Event should be dropped after max delivery attempts."""
        received_count = [0]
        delivery_counts = []

        async def always_failing_handler(event: MessagingEvent):
            received_count[0] += 1
            delivery_counts.append(event._delivery_count)
            # Always fail
            await event.nack(requeue=True)

        await at_least_once_backend.subscribe("test.*", always_failing_handler)

        # Publish AT_LEAST_ONCE event with low max delivery count
        event = MessagingEvent(
            topic="test.event",
            data={"message": "fail"},
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )
        event._max_delivery_count = 3  # Allow 3 deliveries

        await at_least_once_backend.publish(event)

        # Wait for all delivery attempts (increased timeout)
        await asyncio.sleep(1.0)

        # Should receive event at least twice (original + retry)
        assert received_count[0] >= 2, f"Expected at least 2 deliveries, got {received_count[0]}"
        assert received_count[0] <= 3, f"Expected at most 3 deliveries, got {received_count[0]}"

        # Event should NOT be in pending (max retries reached)
        assert event.id not in at_least_once_backend._pending_events

    @pytest.mark.asyncio
    async def test_at_most_once_not_tracked(self, at_least_once_backend):
        """AT_MOST_ONCE events should not be tracked."""
        received_events = []

        async def handler(event: MessagingEvent):
            received_events.append(event)
            # Don't ACK - should still be removed

        await at_least_once_backend.subscribe("test.*", handler)

        # Publish AT_MOST_ONCE event
        event = MessagingEvent(
            topic="test.event",
            data={"message": "fire and forget"},
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
        )
        await at_least_once_backend.publish(event)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Event should be received
        assert len(received_events) == 1

        # Event should NOT be in pending (AT_MOST_ONCE)
        assert event.id not in at_least_once_backend._pending_events


# =============================================================================
# AgentMessageBus AT_LEAST_ONCE Tests
# =============================================================================


@pytest.mark.unit
class TestAgentMessageBusAtLeastOnce:
    """Tests for AgentMessageBus using AT_LEAST_ONCE delivery."""

    @pytest.mark.asyncio
    async def test_send_uses_at_least_once(self, agent_bus):
        """AgentMessageBus.send() should use AT_LEAST_ONCE delivery."""
        received_events = []

        async def handler(event: MessagingEvent):
            received_events.append(event)
            await event.ack()

        await agent_bus.subscribe_agent("receiver", handler)

        # Send message to agent
        result = await agent_bus.send(
            "task",
            {"action": "analyze"},
            to_agent="receiver",
            from_agent="coordinator",
        )

        assert result is True

        # Wait for processing
        await asyncio.sleep(0.2)

        # Event should be received
        assert len(received_events) == 1
        assert received_events[0].delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE

    @pytest.mark.asyncio
    async def test_broadcast_uses_at_least_once(self, agent_bus):
        """AgentMessageBus.broadcast() should use AT_LEAST_ONCE delivery."""
        agent1_events = []
        agent2_events = []

        async def handler1(event: MessagingEvent):
            agent1_events.append(event)
            await event.ack()

        async def handler2(event: MessagingEvent):
            agent2_events.append(event)
            await event.ack()

        await agent_bus.subscribe_agent("agent1", handler1)
        await agent_bus.subscribe_agent("agent2", handler2)

        # Broadcast message
        result = await agent_bus.broadcast(
            "status",
            {"phase": "planning"},
            from_agent="coordinator",
        )

        assert result is True

        # Wait for processing
        await asyncio.sleep(0.2)

        # Both agents should receive the broadcast
        assert len(agent1_events) == 1
        assert len(agent2_events) == 1

        # Both should have AT_LEAST_ONCE guarantee
        assert agent1_events[0].delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
        assert agent2_events[0].delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE

    @pytest.mark.asyncio
    async def test_agent_message_reliable_delivery(self, agent_bus):
        """Agent messages should be retried on failure."""
        received_count = [0]

        async def flaky_handler(event: MessagingEvent):
            received_count[0] += 1
            if received_count[0] == 1:
                # Fail first time
                await event.nack(requeue=True)
            else:
                # Succeed second time
                await event.ack()

        await agent_bus.subscribe_agent("executor", flaky_handler)

        # Send message that will be retried
        await agent_bus.send(
            "task",
            {"action": "execute"},
            to_agent="executor",
            from_agent="coordinator",
        )

        # Wait for retry
        await asyncio.sleep(0.5)

        # Should be received twice (original + retry)
        assert received_count[0] == 2


# =============================================================================
# ObservabilityBus AT_MOST_ONCE Tests
# =============================================================================


@pytest.mark.unit
class TestObservabilityBusAtMostOnce:
    """Tests for ObservabilityBus using AT_MOST_ONCE delivery."""

    @pytest.mark.asyncio
    async def test_observability_uses_at_most_once(self):
        """ObservabilityBus should use AT_MOST_ONCE delivery."""
        from victor.core.events import ObservabilityBus

        bus = ObservabilityBus()
        await bus.connect()

        received_events = []

        async def handler(event: MessagingEvent):
            received_events.append(event)
            # No ACK needed for observability

        await bus.subscribe("metric.*", handler)

        # Emit metric
        result = await bus.emit("metric.test", {"value": 42})

        assert result is True

        # Wait for processing
        await asyncio.sleep(0.2)

        # Event should be received
        assert len(received_events) == 1
        assert received_events[0].delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE

        await bus.disconnect()


# =============================================================================
# Summary
# =============================================================================

"""
Test Summary:
==============

Total Tests: 15
Test Classes: 5

Coverage:
- MessagingEvent Ack/Nack: 5 tests
- AT_LEAST_ONCE Delivery: 4 tests
- AgentMessageBus: 3 tests
- ObservabilityBus: 1 test

Features Tested:
1. Event acknowledgement (ack)
2. Negative acknowledgement (nack)
3. Delivery count tracking
4. Retry logic (should_retry)
5. Pending event tracking
6. Re-delivery on NACK
7. Max retry limit
8. AT_MOST_ONCE vs AT_LEAST_ONCE behavior
9. AgentMessageBus reliability
10. ObservabilityBus fire-and-forget
"""
