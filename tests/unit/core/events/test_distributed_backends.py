"""Tests for distributed event backends (Redis, Kafka).

Tests the RedisEventBackend and KafkaEventBackend implementations
for distributed event messaging.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    DeliveryGuarantee,
    MessagingEvent,
)


# =============================================================================
# Redis Backend Tests
# =============================================================================


class TestRedisEventBackend:
    """Tests for RedisEventBackend."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock()
        mock.close = AsyncMock()
        mock.xadd = AsyncMock(return_value="1234567890-0")
        mock.xreadgroup = AsyncMock(return_value=[])
        mock.xgroup_create = AsyncMock()
        mock.xack = AsyncMock()
        return mock

    @pytest.fixture
    def config(self):
        """Create test config."""
        return BackendConfig(
            backend_type=BackendType.REDIS,
            extra={
                "redis_url": "redis://localhost:6379/0",
                "stream_prefix": "test:events",
                "consumer_group": "test-consumers",
            },
        )

    @pytest.mark.asyncio
    async def test_connect(self, config, mock_redis):
        """Test connecting to Redis."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            assert backend.is_connected
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, config, mock_redis):
        """Test disconnecting from Redis."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()
            await backend.disconnect()

            assert not backend.is_connected
            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish(self, config, mock_redis):
        """Test publishing an event to Redis Stream."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            event = MessagingEvent(
                topic="tool.call",
                data={"name": "read_file"},
                source="test",
            )

            result = await backend.publish(event)

            assert result is True
            mock_redis.xadd.assert_called_once()
            call_args = mock_redis.xadd.call_args
            assert call_args[0][0] == "test:events:tool.call"

    @pytest.mark.asyncio
    async def test_publish_not_connected(self, config):
        """Test publishing when not connected raises error."""
        with patch("victor.core.events.redis_backend.aioredis"):
            from victor.core.events.redis_backend import RedisEventBackend
            from victor.core.events.protocols import EventPublishError

            backend = RedisEventBackend(config)

            event = MessagingEvent(topic="test", data={})

            with pytest.raises(EventPublishError):
                await backend.publish(event)

    @pytest.mark.asyncio
    async def test_subscribe(self, config, mock_redis):
        """Test subscribing to events."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)
            mock_aioredis.ResponseError = Exception

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            handler = AsyncMock()
            handle = await backend.subscribe("tool.*", handler)

            assert handle.is_active
            assert handle.pattern == "tool.*"
            assert len(backend._subscriptions) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, config, mock_redis):
        """Test unsubscribing from events."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)
            mock_aioredis.ResponseError = Exception

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            handler = AsyncMock()
            handle = await backend.subscribe("tool.*", handler)

            result = await backend.unsubscribe(handle)

            assert result is True
            assert not handle.is_active
            assert len(backend._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_health_check(self, config, mock_redis):
        """Test health check."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            result = await backend.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, config):
        """Test health check when not connected."""
        with patch("victor.core.events.redis_backend.aioredis"):
            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)

            result = await backend.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_backend_type(self, config):
        """Test backend type property."""
        with patch("victor.core.events.redis_backend.aioredis"):
            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)

            assert backend.backend_type == BackendType.REDIS

    @pytest.mark.asyncio
    async def test_get_stats(self, config, mock_redis):
        """Test getting backend statistics."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            stats = backend.get_stats()

            assert stats["backend_type"] == "redis"
            assert stats["is_connected"] is True
            assert "published_count" in stats
            assert "consumed_count" in stats

    @pytest.mark.asyncio
    async def test_publish_batch(self, config, mock_redis):
        """Test publishing multiple events."""
        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend(config)
            await backend.connect()

            events = [MessagingEvent(topic="tool.call", data={"index": i}) for i in range(5)]

            count = await backend.publish_batch(events)

            assert count == 5
            assert mock_redis.xadd.call_count == 5

    def test_matches_pattern(self):
        """Test pattern matching using shared utility."""
        from victor.core.events.pattern_matcher import matches_topic_pattern

        # Test single-segment wildcard
        assert matches_topic_pattern("tool.call", "tool.*") is True
        assert matches_topic_pattern("tool.result", "tool.*") is True
        assert matches_topic_pattern("agent.message", "tool.*") is False

        # Test universal wildcard
        assert matches_topic_pattern("anything", "*") is True

        # Test multi-segment wildcard (trailing **)
        assert matches_topic_pattern("tool.call.sub", "tool.**") is True
        assert matches_topic_pattern("tool.call.sub.deep", "tool.**") is True
        assert matches_topic_pattern("agent.call", "tool.**") is False


# =============================================================================
# Kafka Backend Tests
# =============================================================================


class TestKafkaEventBackend:
    """Tests for KafkaEventBackend."""

    @pytest.fixture
    def mock_producer(self):
        """Create mock Kafka producer."""
        mock = AsyncMock()
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        mock.send_and_wait = AsyncMock()
        mock.send = AsyncMock()
        mock.client = MagicMock()
        mock.client.fetch_all_metadata = AsyncMock(return_value=MagicMock())
        return mock

    @pytest.fixture
    def mock_consumer(self):
        """Create mock Kafka consumer."""
        import asyncio

        async def mock_getone():
            """Mock getone that times out quickly."""
            raise asyncio.TimeoutError()

        mock = AsyncMock()
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        mock.getone = mock_getone
        return mock

    @pytest.fixture
    def config(self):
        """Create test config."""
        return BackendConfig(
            backend_type=BackendType.KAFKA,
            extra={
                "bootstrap_servers": "localhost:9092",
                "topic_prefix": "test.events",
                "consumer_group": "test-consumers",
            },
        )

    @pytest.mark.asyncio
    async def test_connect(self, config, mock_producer):
        """Test connecting to Kafka."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            MockProducer.return_value = mock_producer

            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)
            await backend.connect()

            assert backend.is_connected
            mock_producer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, config, mock_producer):
        """Test disconnecting from Kafka."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            MockProducer.return_value = mock_producer

            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)
            await backend.connect()
            await backend.disconnect()

            assert not backend.is_connected
            mock_producer.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish(self, config, mock_producer):
        """Test publishing an event to Kafka."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            MockProducer.return_value = mock_producer

            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)
            await backend.connect()

            event = MessagingEvent(
                topic="tool.call",
                data={"name": "read_file"},
                source="test",
            )

            result = await backend.publish(event)

            assert result is True
            mock_producer.send_and_wait.assert_called_once()
            call_args = mock_producer.send_and_wait.call_args
            assert call_args[0][0] == "test.events.tool_call"

    @pytest.mark.asyncio
    async def test_publish_with_partition_key(self, config, mock_producer):
        """Test publishing an event with partition key."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            MockProducer.return_value = mock_producer

            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)
            await backend.connect()

            event = MessagingEvent(
                topic="tool.call",
                data={"name": "read_file"},
                partition_key="user123",
            )

            await backend.publish(event)

            call_args = mock_producer.send_and_wait.call_args
            assert call_args[1]["key"] == b"user123"

    @pytest.mark.asyncio
    async def test_publish_not_connected(self, config):
        """Test publishing when not connected raises error."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer"):
            from victor.core.events.kafka_backend import KafkaEventBackend
            from victor.core.events.protocols import EventPublishError

            backend = KafkaEventBackend(config)

            event = MessagingEvent(topic="test", data={})

            with pytest.raises(EventPublishError):
                await backend.publish(event)

    @pytest.mark.asyncio
    async def test_subscribe(self, config, mock_producer, mock_consumer):
        """Test subscribing to events."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            with patch("victor.core.events.kafka_backend.AIOKafkaConsumer") as MockConsumer:
                MockProducer.return_value = mock_producer
                MockConsumer.return_value = mock_consumer

                from victor.core.events.kafka_backend import KafkaEventBackend

                backend = KafkaEventBackend(config)
                await backend.connect()

                handler = AsyncMock()
                handle = await backend.subscribe("tool.*", handler)

                assert handle.is_active
                assert handle.pattern == "tool.*"
                assert len(backend._subscriptions) == 1

                # Clean up: disconnect to stop the background consume loop
                await backend.disconnect()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, config, mock_producer, mock_consumer):
        """Test unsubscribing from events."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            with patch("victor.core.events.kafka_backend.AIOKafkaConsumer") as MockConsumer:
                MockProducer.return_value = mock_producer
                MockConsumer.return_value = mock_consumer

                from victor.core.events.kafka_backend import KafkaEventBackend

                backend = KafkaEventBackend(config)
                await backend.connect()

                handler = AsyncMock()
                handle = await backend.subscribe("tool.*", handler)

                result = await backend.unsubscribe(handle)

                assert result is True
                assert not handle.is_active
                assert len(backend._subscriptions) == 0

                # Clean up: disconnect to stop the background consume loop
                await backend.disconnect()

    @pytest.mark.asyncio
    async def test_health_check(self, config, mock_producer):
        """Test health check."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            MockProducer.return_value = mock_producer

            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)
            await backend.connect()

            result = await backend.health_check()

            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self, config):
        """Test health check when not connected."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer"):
            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)

            result = await backend.health_check()

            assert result is False

    @pytest.mark.asyncio
    async def test_backend_type(self, config):
        """Test backend type property."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer"):
            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)

            assert backend.backend_type == BackendType.KAFKA

    @pytest.mark.asyncio
    async def test_get_stats(self, config, mock_producer):
        """Test getting backend statistics."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            MockProducer.return_value = mock_producer

            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)
            await backend.connect()

            stats = backend.get_stats()

            assert stats["backend_type"] == "kafka"
            assert stats["is_connected"] is True
            assert "published_count" in stats
            assert "consumed_count" in stats
            assert stats["bootstrap_servers"] == "localhost:9092"

    @pytest.mark.asyncio
    async def test_topic_name_conversion(self, config):
        """Test topic name conversion."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer"):
            from victor.core.events.kafka_backend import KafkaEventBackend

            backend = KafkaEventBackend(config)

            # Event topic to Kafka topic
            kafka_topic = backend._get_topic_name("tool.call")
            assert kafka_topic == "test.events.tool_call"

            # Kafka topic back to event topic
            event_topic = backend._extract_event_topic("test.events.tool_call")
            assert event_topic == "tool.call"

    def test_matches_pattern(self):
        """Test pattern matching using shared utility."""
        from victor.core.events.pattern_matcher import matches_topic_pattern

        # Test single-segment wildcard
        assert matches_topic_pattern("tool.call", "tool.*") is True
        assert matches_topic_pattern("tool.result", "tool.*") is True
        assert matches_topic_pattern("agent.message", "tool.*") is False

        # Test universal wildcard
        assert matches_topic_pattern("anything", "*") is True

        # Test multi-segment wildcard (trailing **)
        assert matches_topic_pattern("tool.call.sub", "tool.**") is True
        assert matches_topic_pattern("tool.call.sub.deep", "tool.**") is True
        assert matches_topic_pattern("agent.call", "tool.**") is False


# =============================================================================
# Backend Factory Tests
# =============================================================================


class TestBackendFactoryRegistration:
    """Tests for backend factory registration."""

    @pytest.mark.asyncio
    async def test_redis_backend_registered(self):
        """Test that Redis backend is registered when available."""
        with patch("victor.core.events.redis_backend.aioredis"):
            # Re-import to trigger registration
            from victor.core.events.backends import (
                create_event_backend,
                _backend_factories,
            )

            # Check if Redis is registered (may not be if redis not installed)
            if BackendType.REDIS in _backend_factories:
                config = BackendConfig(
                    backend_type=BackendType.REDIS,
                    extra={"redis_url": "redis://localhost:6379/0"},
                )
                backend = create_event_backend(config)
                assert backend.backend_type == BackendType.REDIS

    @pytest.mark.asyncio
    async def test_kafka_backend_registered(self):
        """Test that Kafka backend is registered when available."""
        with patch("victor.core.events.kafka_backend.AIOKafkaProducer"):
            from victor.core.events.backends import (
                create_event_backend,
                _backend_factories,
            )

            # Check if Kafka is registered (may not be if aiokafka not installed)
            if BackendType.KAFKA in _backend_factories:
                config = BackendConfig(
                    backend_type=BackendType.KAFKA,
                    extra={"bootstrap_servers": "localhost:9092"},
                )
                backend = create_event_backend(config)
                assert backend.backend_type == BackendType.KAFKA

    @pytest.mark.asyncio
    async def test_fallback_to_inmemory(self):
        """Test fallback to in-memory when backend not registered."""
        from victor.core.events.backends import create_event_backend

        # Use a type that might not be registered
        config = BackendConfig(backend_type=BackendType.SQS)
        backend = create_event_backend(config)

        # Should fall back to in-memory
        assert backend.backend_type == BackendType.IN_MEMORY


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestDistributedBackendIntegration:
    """Integration tests for distributed backends with mocks."""

    @pytest.mark.asyncio
    async def test_redis_publish_subscribe_flow(self):
        """Test complete publish-subscribe flow with Redis."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.close = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value="1234567890-0")
        mock_redis.xreadgroup = AsyncMock(return_value=[])
        mock_redis.xgroup_create = AsyncMock()

        with patch("victor.core.events.redis_backend.aioredis") as mock_aioredis:
            mock_aioredis.from_url = AsyncMock(return_value=mock_redis)
            mock_aioredis.ResponseError = Exception

            from victor.core.events.redis_backend import RedisEventBackend

            backend = RedisEventBackend()
            await backend.connect()

            # Subscribe
            received_events = []

            async def handler(event):
                received_events.append(event)

            await backend.subscribe("tool.*", handler)

            # Publish
            event = MessagingEvent(topic="tool.call", data={"test": True})
            await backend.publish(event)

            # Verify publish was called
            assert mock_redis.xadd.called

            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_kafka_publish_subscribe_flow(self):
        """Test complete publish-subscribe flow with Kafka."""
        mock_producer = AsyncMock()
        mock_producer.start = AsyncMock()
        mock_producer.stop = AsyncMock()
        mock_producer.send_and_wait = AsyncMock()
        mock_producer.client = MagicMock()

        mock_consumer = AsyncMock()
        mock_consumer.start = AsyncMock()
        mock_consumer.stop = AsyncMock()

        with patch("victor.core.events.kafka_backend.AIOKafkaProducer") as MockProducer:
            with patch("victor.core.events.kafka_backend.AIOKafkaConsumer") as MockConsumer:
                MockProducer.return_value = mock_producer
                MockConsumer.return_value = mock_consumer

                from victor.core.events.kafka_backend import KafkaEventBackend

                backend = KafkaEventBackend()
                await backend.connect()

                # Subscribe
                received_events = []

                async def handler(event):
                    received_events.append(event)

                await backend.subscribe("tool.*", handler)

                # Publish
                event = MessagingEvent(topic="tool.call", data={"test": True})
                await backend.publish(event)

                # Verify publish was called
                assert mock_producer.send_and_wait.called

                # Clean up: disconnect to stop the background consume loop
                await backend.disconnect()
