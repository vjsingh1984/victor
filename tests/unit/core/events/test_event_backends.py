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

"""Tests for protocol-based event backends.

These tests verify:
1. Event creation and serialization
2. InMemoryEventBackend pub/sub functionality
3. Pattern matching for subscriptions
4. ObservabilityBus and AgentMessageBus specialized buses
5. Backend factory and registration

Run with: pytest tests/unit/core/events/test_event_backends.py -v
"""

import asyncio
from typing import List

import pytest

from victor.core.events import (
    # Core types
    MessagingEvent,
    SubscriptionHandle,
    DeliveryGuarantee,
    BackendType,
    BackendConfig,
    # Protocols
    IEventBackend,
    # Backends
    InMemoryEventBackend,
    # Specialized buses
    ObservabilityBus,
    AgentMessageBus,
    # Factory
    create_event_backend,
    register_backend_factory,
)


# =============================================================================
# Event Tests
# =============================================================================


@pytest.mark.unit
class TestMessagingEvent:
    """Tests for Event dataclass."""

    def test_event_creation_with_defaults(self):
        """Event should have sensible defaults."""
        event = MessagingEvent(topic="test.topic", data={"key": "value"})

        assert event.topic == "test.topic"
        assert event.data == {"key": "value"}
        assert event.id is not None
        assert event.timestamp > 0
        assert event.source == "victor"
        assert event.delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE

    def test_event_serialization(self):
        """Event should serialize to and from dict."""
        event = MessagingEvent(
            topic="tool.call",
            data={"name": "read", "args": {"file": "test.txt"}},
            source="agent_1",
            correlation_id="task_123",
        )

        # Serialize
        data = event.to_dict()
        assert data["topic"] == "tool.call"
        assert data["source"] == "agent_1"
        assert data["correlation_id"] == "task_123"

        # Deserialize
        restored = MessagingEvent.from_dict(data)
        assert restored.topic == event.topic
        assert restored.data == event.data
        assert restored.source == event.source

    def test_event_pattern_matching_exact(self):
        """Event should match exact topic patterns."""
        event = MessagingEvent(topic="tool.call")

        assert event.matches_pattern("tool.call") is True
        assert event.matches_pattern("tool.result") is False
        assert event.matches_pattern("agent.call") is False

    def test_event_pattern_matching_wildcard(self):
        """Event should match wildcard patterns."""
        event = MessagingEvent(topic="tool.call")

        assert event.matches_pattern("tool.*") is True
        assert event.matches_pattern("*.call") is True
        assert event.matches_pattern("*") is True
        assert event.matches_pattern("agent.*") is False

    def test_event_pattern_matching_nested(self):
        """Event should match nested topic patterns."""
        event = MessagingEvent(topic="agent.researcher.task")

        assert event.matches_pattern("agent.researcher.task") is True
        assert event.matches_pattern("agent.researcher.*") is True
        assert event.matches_pattern("agent.*.*") is True
        assert event.matches_pattern("*.researcher.*") is True


# =============================================================================
# InMemoryEventBackend Tests
# =============================================================================


@pytest.mark.unit
class TestInMemoryEventBackend:
    """Tests for InMemoryEventBackend."""

    @pytest.fixture
    async def backend(self):
        """Create and connect a backend for each test."""
        backend = InMemoryEventBackend()
        await backend.connect()
        yield backend
        await backend.disconnect()

    def test_backend_type(self):
        """Backend should report correct type."""
        backend = InMemoryEventBackend()
        assert backend.backend_type == BackendType.IN_MEMORY

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Backend should connect and disconnect cleanly."""
        backend = InMemoryEventBackend()

        assert backend.is_connected is False

        await backend.connect()
        assert backend.is_connected is True

        await backend.disconnect()
        assert backend.is_connected is False

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Connected backend should pass health check."""
        assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_publish_subscribe_basic(self, backend):
        """Basic pub/sub should work."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        # Subscribe
        handle = await backend.subscribe("test.topic", handler)
        assert handle.is_active is True

        # Publish
        event = MessagingEvent(topic="test.topic", data={"msg": "hello"})
        result = await backend.publish(event)
        assert result is True

        # Wait for delivery
        await asyncio.sleep(0.2)

        # Verify
        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_subscribe_with_pattern(self, backend):
        """Subscriptions should filter by pattern."""
        tool_events: List[MessagingEvent] = []
        agent_events: List[MessagingEvent] = []

        async def tool_handler(event: MessagingEvent):
            tool_events.append(event)

        async def agent_handler(event: MessagingEvent):
            agent_events.append(event)

        # Subscribe to different patterns
        await backend.subscribe("tool.*", tool_handler)
        await backend.subscribe("agent.*", agent_handler)

        # Publish events
        await backend.publish(MessagingEvent(topic="tool.call", data={}))
        await backend.publish(MessagingEvent(topic="agent.message", data={}))
        await backend.publish(MessagingEvent(topic="tool.result", data={}))

        await asyncio.sleep(0.3)

        # Verify filtering
        assert len(tool_events) == 2
        assert len(agent_events) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, backend):
        """Unsubscribe should stop event delivery."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        handle = await backend.subscribe("test.*", handler)

        # First event should be received
        await backend.publish(MessagingEvent(topic="test.1"))
        await asyncio.sleep(0.2)
        assert len(received) == 1

        # Unsubscribe
        await handle.unsubscribe()
        assert handle.is_active is False

        # Second event should NOT be received
        await backend.publish(MessagingEvent(topic="test.2"))
        await asyncio.sleep(0.2)
        assert len(received) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, backend):
        """Multiple subscribers should all receive events."""
        handler1_events: List[MessagingEvent] = []
        handler2_events: List[MessagingEvent] = []

        async def handler1(event: MessagingEvent):
            handler1_events.append(event)

        async def handler2(event: MessagingEvent):
            handler2_events.append(event)

        await backend.subscribe("test.*", handler1)
        await backend.subscribe("test.*", handler2)

        await backend.publish(MessagingEvent(topic="test.event"))
        await asyncio.sleep(0.2)

        assert len(handler1_events) == 1
        assert len(handler2_events) == 1

    @pytest.mark.asyncio
    async def test_publish_batch(self, backend):
        """Batch publish should deliver all events."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        await backend.subscribe("batch.*", handler)

        events = [
            MessagingEvent(topic="batch.1", data={"i": 1}),
            MessagingEvent(topic="batch.2", data={"i": 2}),
            MessagingEvent(topic="batch.3", data={"i": 3}),
        ]

        count = await backend.publish_batch(events)
        assert count == 3

        await asyncio.sleep(0.3)
        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_subscription_count(self, backend):
        """Backend should track subscription count."""
        assert backend.get_subscription_count() == 0

        h1 = await backend.subscribe("a.*", lambda e: None)
        assert backend.get_subscription_count() == 1

        h2 = await backend.subscribe("b.*", lambda e: None)
        assert backend.get_subscription_count() == 2

        await h1.unsubscribe()
        assert backend.get_subscription_count() == 1


# =============================================================================
# ObservabilityBus Tests
# =============================================================================


@pytest.mark.unit
class TestObservabilityBus:
    """Tests for ObservabilityBus specialized bus."""

    @pytest.fixture
    async def bus(self):
        """Create and connect bus for each test."""
        bus = ObservabilityBus()
        await bus.connect()
        yield bus
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_emit_event(self, bus):
        """Emit should publish events."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        await bus.subscribe("metric.*", handler)

        result = await bus.emit(
            "metric.latency",
            {"value": 42.5, "unit": "ms"},
            source="test",
        )
        assert result is True

        await asyncio.sleep(0.2)
        assert len(received) == 1
        assert received[0].data["value"] == 42.5

    @pytest.mark.asyncio
    async def test_emit_with_correlation(self, bus):
        """Emit should support correlation IDs."""
        received: List[MessagingEvent] = []

        async def handler(event: MessagingEvent):
            received.append(event)

        await bus.subscribe("trace.*", handler)

        await bus.emit(
            "trace.span",
            {"name": "test_span"},
            correlation_id="trace_123",
        )

        await asyncio.sleep(0.2)
        assert received[0].correlation_id == "trace_123"


# =============================================================================
# AgentMessageBus Tests
# =============================================================================


@pytest.mark.unit
class TestAgentMessageBus:
    """Tests for AgentMessageBus specialized bus."""

    @pytest.fixture
    async def bus(self):
        """Create and connect bus for each test."""
        bus = AgentMessageBus()
        await bus.connect()
        yield bus
        await bus.disconnect()

    @pytest.mark.asyncio
    async def test_send_to_agent(self, bus):
        """Send should deliver to specific agent."""
        researcher_msgs: List[MessagingEvent] = []
        executor_msgs: List[MessagingEvent] = []

        async def researcher_handler(event: MessagingEvent):
            researcher_msgs.append(event)

        async def executor_handler(event: MessagingEvent):
            executor_msgs.append(event)

        await bus.subscribe_agent("researcher", researcher_handler)
        await bus.subscribe_agent("executor", executor_handler)

        # Send to researcher only
        await bus.send(
            "task",
            {"action": "analyze"},
            to_agent="researcher",
            from_agent="coordinator",
        )

        await asyncio.sleep(0.2)

        assert len(researcher_msgs) == 1
        assert researcher_msgs[0].data["action"] == "analyze"
        # Executor should not receive directed message
        # (only broadcasts)

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, bus):
        """Broadcast should reach all agents."""
        agent1_msgs: List[MessagingEvent] = []
        agent2_msgs: List[MessagingEvent] = []

        async def agent1_handler(event: MessagingEvent):
            agent1_msgs.append(event)

        async def agent2_handler(event: MessagingEvent):
            agent2_msgs.append(event)

        await bus.subscribe_agent("agent1", agent1_handler)
        await bus.subscribe_agent("agent2", agent2_handler)

        # Broadcast
        await bus.broadcast(
            "status",
            {"phase": "planning"},
            from_agent="coordinator",
        )

        await asyncio.sleep(0.2)

        # Both agents should receive broadcast
        assert len(agent1_msgs) == 1
        assert len(agent2_msgs) == 1
        assert agent1_msgs[0].data["phase"] == "planning"


# =============================================================================
# Factory Tests
# =============================================================================


@pytest.mark.unit
class TestBackendFactory:
    """Tests for backend factory functions."""

    def test_create_default_backend(self):
        """Factory should create InMemory backend by default."""
        backend = create_event_backend()
        assert backend.backend_type == BackendType.IN_MEMORY
        assert isinstance(backend, InMemoryEventBackend)

    def test_create_with_config(self):
        """Factory should respect configuration."""
        config = BackendConfig(
            backend_type=BackendType.IN_MEMORY,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )
        backend = create_event_backend(config)
        assert backend.backend_type == BackendType.IN_MEMORY

    def test_create_with_type_override(self):
        """Factory should allow type override."""
        backend = create_event_backend(backend_type=BackendType.IN_MEMORY)
        assert backend.backend_type == BackendType.IN_MEMORY

    def test_fallback_for_unregistered_type(self):
        """Factory should fallback to InMemory for unregistered types."""
        config = BackendConfig(backend_type=BackendType.KAFKA)
        backend = create_event_backend(config)
        # Should fallback to InMemory since Kafka isn't registered
        assert backend.backend_type == BackendType.IN_MEMORY

    def test_register_custom_backend(self):
        """Custom backends can be registered."""

        class MockKafkaBackend:
            """Mock Kafka backend for testing."""

            @property
            def backend_type(self):
                return BackendType.KAFKA

        # Register
        register_backend_factory(
            BackendType.KAFKA,
            lambda config: MockKafkaBackend(),
        )

        # Create
        backend = create_event_backend(backend_type=BackendType.KAFKA)
        assert backend.backend_type == BackendType.KAFKA


# =============================================================================
# BackendConfig Tests
# =============================================================================


@pytest.mark.unit
class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_observability_config(self):
        """for_observability should return optimized config."""
        config = BackendConfig.for_observability()

        assert config.delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE
        assert config.max_batch_size == 100

    def test_agent_messaging_config(self):
        """for_agent_messaging should return reliable config."""
        config = BackendConfig.for_agent_messaging()

        assert config.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
        assert config.max_retries == 5


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


@pytest.mark.unit
class TestProtocolCompliance:
    """Tests verifying protocol compliance."""

    def test_inmemory_implements_ieventbackend(self):
        """InMemoryEventBackend should implement IEventBackend."""
        backend = InMemoryEventBackend()

        # Check protocol methods exist
        assert hasattr(backend, "connect")
        assert hasattr(backend, "disconnect")
        assert hasattr(backend, "publish")
        assert hasattr(backend, "subscribe")
        assert hasattr(backend, "health_check")
        assert hasattr(backend, "backend_type")
        assert hasattr(backend, "is_connected")

    def test_protocol_is_runtime_checkable(self):
        """IEventBackend should be runtime checkable."""
        backend = InMemoryEventBackend()
        assert isinstance(backend, IEventBackend)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
