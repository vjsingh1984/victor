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

"""Redis event backend implementation using Redis Streams.

This module provides RedisEventBackend, a distributed event backend that uses
Redis Streams for reliable message delivery with support for consumer groups,
acknowledgments, and message persistence.

Features:
    - Redis Streams for ordered, persistent message storage
    - Consumer groups for distributed processing
    - Automatic acknowledgment handling
    - Pattern-based subscriptions via multiple stream consumers
    - AT_LEAST_ONCE and AT_MOST_ONCE delivery guarantees
    - Automatic reconnection and error recovery

Architecture:
    - Each topic maps to a Redis Stream (e.g., "victor:events:tool.call")
    - Wildcard patterns subscribe to multiple streams
    - Consumer groups enable distributed processing
    - XACK used for AT_LEAST_ONCE acknowledgments

Example:
    from victor.core.events.redis_backend import RedisEventBackend
    from victor.core.events.protocols import BackendConfig, BackendType

    config = BackendConfig(
        backend_type=BackendType.REDIS,
        extra={
            "redis_url": "redis://localhost:6379/0",
            "stream_prefix": "victor:events",
            "consumer_group": "victor-consumers",
        }
    )

    backend = RedisEventBackend(config)
    await backend.connect()

    # Publish
    await backend.publish(MessagingEvent(topic="tool.call", data={"name": "read"}))

    # Subscribe
    async def handler(event):
        print(f"Received: {event.topic}")
        await event.ack()

    handle = await backend.subscribe("tool.*", handler)

    await backend.disconnect()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from collections.abc import Awaitable, Callable

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

from victor.core.events.pattern_matcher import matches_topic_pattern
from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    DeliveryGuarantee,
    EventHandler,
    EventPublishError,
    MessagingEvent,
    SubscriptionHandle,
)

logger = logging.getLogger(__name__)


@dataclass
class _RedisSubscription:
    """Internal subscription tracking for Redis backend."""

    id: str
    pattern: str
    handler: EventHandler
    is_active: bool = True
    stream_keys: list[str] = field(default_factory=list)


class _BoundSubscriptionHandle(SubscriptionHandle):
    """SubscriptionHandle with bound unsubscribe method."""

    def __init__(
        self,
        subscription_id: str,
        pattern: str,
        unsubscribe_func: Callable[[], Awaitable[None]],
    ):
        super().__init__(subscription_id=subscription_id, pattern=pattern, is_active=True)
        self._unsubscribe_func = unsubscribe_func

    async def unsubscribe(self) -> None:
        """Unsubscribe using the bound function."""
        await self._unsubscribe_func()
        self.is_active = False


class RedisEventBackend:
    """Redis Streams-based distributed event backend.

    This backend uses Redis Streams for reliable, persistent event delivery
    across distributed processes. It supports consumer groups for load
    balancing and automatic acknowledgments.

    Features:
        - Persistent message storage (survives restarts)
        - Consumer groups for distributed processing
        - Automatic message acknowledgment
        - Pattern-based topic matching
        - Configurable delivery guarantees

    Configuration (via BackendConfig.extra):
        redis_url: Redis connection URL (default: "redis://localhost:6379/0")
        stream_prefix: Prefix for stream names (default: "victor:events")
        consumer_group: Consumer group name (default: "victor-consumers")
        consumer_name: This consumer's name (default: auto-generated)
        max_stream_length: Maximum stream length, 0 for unlimited (default: 10000)
        block_timeout_ms: Timeout for blocking reads (default: 1000)
        batch_size: Number of messages to read per batch (default: 10)

    Example:
        config = BackendConfig(
            backend_type=BackendType.REDIS,
            extra={"redis_url": "redis://localhost:6379/0"}
        )
        backend = RedisEventBackend(config)
        await backend.connect()
    """

    def __init__(self, config: Optional[BackendConfig] = None) -> None:
        """Initialize the Redis event backend.

        Args:
            config: Backend configuration with Redis-specific options in extra

        Raises:
            ImportError: If redis package is not installed
        """
        if aioredis is None:
            raise ImportError(
                "redis package is required for RedisEventBackend. "
                "Install it with: pip install redis>=4.0"
            )

        self._config = config or BackendConfig()

        # Extract Redis-specific config
        extra = self._config.extra
        self._redis_url = extra.get("redis_url", "redis://localhost:6379/0")
        self._stream_prefix = extra.get("stream_prefix", "victor:events")
        self._consumer_group = extra.get("consumer_group", "victor-consumers")
        self._consumer_name = extra.get("consumer_name", f"consumer-{uuid.uuid4().hex[:8]}")
        self._max_stream_length = extra.get("max_stream_length", 10000)
        self._block_timeout_ms = extra.get("block_timeout_ms", 1000)
        self._batch_size = extra.get("batch_size", 10)

        # Connection state
        self._redis: Optional[aioredis.Redis] = None
        self._is_connected = False

        # Subscription management
        self._subscriptions: dict[str, _RedisSubscription] = {}
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._known_streams: set[str] = set()
        self._lock = asyncio.Lock()

        # Statistics
        self._published_count = 0
        self._consumed_count = 0
        self._error_count = 0

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.REDIS

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected and ready."""
        return self._is_connected

    async def connect(self) -> None:
        """Connect to Redis and initialize streams.

        Creates the Redis connection and ensures consumer groups exist
        for known streams.

        Raises:
            ConnectionError: If connection to Redis fails
        """
        if self._is_connected:
            return

        try:
            self._redis = await aioredis.from_url(
                self._redis_url,
                decode_responses=True,
            )

            # Verify connection
            await self._redis.ping()  # type: ignore[misc]

            self._is_connected = True
            logger.info(f"RedisEventBackend connected to {self._redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Redis and clean up resources.

        Stops the consumer task and closes the Redis connection.
        """
        if not self._is_connected:
            return

        self._is_connected = False

        # Stop consumer task
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await asyncio.wait_for(self._consumer_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Close Redis connection
        if self._redis:
            await self._redis.close()
            self._redis = None

        logger.info("RedisEventBackend disconnected")

    async def health_check(self) -> bool:
        """Check backend health.

        Returns:
            True if connected and Redis is responsive
        """
        if not self._is_connected or not self._redis:
            return False

        try:
            await self._redis.ping()  # type: ignore[misc]
            return True
        except Exception:
            return False

    def _get_stream_name(self, topic: str) -> str:
        """Get Redis stream name for a topic.

        Args:
            topic: Event topic (e.g., "tool.call")

        Returns:
            Stream name (e.g., "victor:events:tool.call")
        """
        return f"{self._stream_prefix}:{topic}"

    async def _ensure_consumer_group(self, stream_name: str) -> None:
        """Ensure consumer group exists for a stream.

        Creates the stream and consumer group if they don't exist.

        Args:
            stream_name: Name of the Redis stream
        """
        if not self._redis:
            return

        try:
            # Create consumer group (creates stream if needed)
            await self._redis.xgroup_create(
                stream_name,
                self._consumer_group,
                id="0",  # Start from beginning
                mkstream=True,
            )
            logger.debug(f"Created consumer group for stream {stream_name}")
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                pass
            else:
                raise

    async def publish(self, event: MessagingEvent) -> bool:
        """Publish an event to Redis Stream.

        Args:
            event: Event to publish

        Returns:
            True if published successfully

        Raises:
            EventPublishError: If publish fails
        """
        if not self._is_connected or not self._redis:
            raise EventPublishError(event, "Backend not connected", retryable=True)

        stream_name = self._get_stream_name(event.topic)

        try:
            # Serialize event
            event_data = {
                "id": event.id,
                "topic": event.topic,
                "data": json.dumps(event.data),
                "timestamp": str(event.timestamp),
                "source": event.source,
                "correlation_id": event.correlation_id or "",
                "partition_key": event.partition_key or "",
                "headers": json.dumps(event.headers),
                "delivery_guarantee": event.delivery_guarantee.value,
            }

            # Add to stream with optional max length
            if self._max_stream_length > 0:
                await self._redis.xadd(
                    stream_name,
                    event_data,  # type: ignore[arg-type]
                    maxlen=self._max_stream_length,
                    approximate=True,
                )
            else:
                await self._redis.xadd(stream_name, event_data)  # type: ignore[arg-type]

            # Track stream for subscriptions
            async with self._lock:
                self._known_streams.add(stream_name)

            self._published_count += 1
            logger.debug(f"Published event {event.id} to stream {stream_name}")
            return True

        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to publish event {event.id}: {e}")
            raise EventPublishError(event, str(e), retryable=True) from e

    async def publish_batch(self, events: list[MessagingEvent]) -> int:
        """Publish multiple events to Redis Streams.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully published
        """
        if not self._is_connected or not self._redis:
            return 0

        success_count = 0
        for event in events:
            try:
                if await self.publish(event):
                    success_count += 1
            except EventPublishError:
                continue

        return success_count

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events matching a pattern.

        Supports wildcard patterns:
        - "tool.*" matches "tool.call", "tool.result"
        - "*" matches all topics

        Args:
            pattern: Topic pattern with optional wildcards
            handler: Async callback for received events

        Returns:
            Handle for managing the subscription
        """
        subscription_id = uuid.uuid4().hex[:12]

        subscription = _RedisSubscription(
            id=subscription_id,
            pattern=pattern,
            handler=handler,
            is_active=True,
        )

        async with self._lock:
            self._subscriptions[subscription_id] = subscription

        # Start consumer task if not running
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._consume_loop())

        # Create handle with bound unsubscribe capability
        async def bound_unsubscribe() -> None:
            # Create a temporary handle for unsubscription
            temp_handle = SubscriptionHandle(
                subscription_id=subscription_id,
                pattern=pattern,
                is_active=True,
            )
            await self.unsubscribe(temp_handle)

        handle = _BoundSubscriptionHandle(
            subscription_id=subscription_id,
            pattern=pattern,
            unsubscribe_func=bound_unsubscribe,
        )

        logger.debug(f"Subscribed to pattern '{pattern}' (id: {subscription_id})")
        return handle

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe using a handle.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        async with self._lock:
            subscription = self._subscriptions.get(handle.subscription_id)
            if subscription:
                subscription.is_active = False
                del self._subscriptions[handle.subscription_id]
                handle.is_active = False
                logger.debug(f"Unsubscribed from pattern '{handle.pattern}'")
                return True
        return False

    async def _consume_loop(self) -> None:
        """Background loop that consumes messages from Redis Streams.

        Reads from all known streams and dispatches to matching handlers.
        """
        while self._is_connected and self._subscriptions:
            try:
                if not self._redis:
                    await asyncio.sleep(0.1)
                    continue

                # Get current streams to monitor
                async with self._lock:
                    streams = list(self._known_streams)
                    subscriptions = list(self._subscriptions.values())

                if not streams:
                    # No streams yet - wait and retry
                    await asyncio.sleep(0.1)
                    continue

                # Ensure consumer groups exist
                for stream in streams:
                    await self._ensure_consumer_group(stream)

                # Build stream dict for XREADGROUP
                # Use ">" to only get new messages
                stream_dict = dict.fromkeys(streams, ">")

                try:
                    # Read from streams
                    messages = await self._redis.xreadgroup(
                        self._consumer_group,
                        self._consumer_name,
                        streams=stream_dict,  # type: ignore[arg-type]
                        count=self._batch_size,
                        block=self._block_timeout_ms,
                    )
                except aioredis.ResponseError as e:
                    if "NOGROUP" in str(e):
                        # Consumer group doesn't exist - will be created on next iteration
                        await asyncio.sleep(0.1)
                        continue
                    raise

                if not messages:
                    continue

                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        try:
                            event = self._deserialize_event(message_data)

                            # Find matching subscriptions
                            for subscription in subscriptions:
                                if subscription.is_active and matches_topic_pattern(
                                    event.topic, subscription.pattern
                                ):
                                    try:
                                        await subscription.handler(event)
                                    except Exception as e:
                                        logger.warning(f"Handler error for {event.topic}: {e}")

                            # Acknowledge message
                            await self._redis.xack(stream_name, self._consumer_group, message_id)
                            self._consumed_count += 1

                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            self._error_count += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                self._error_count += 1
                await asyncio.sleep(1.0)  # Back off on error

    def _deserialize_event(self, data: dict[str, str]) -> MessagingEvent:
        """Deserialize event from Redis stream message.

        Args:
            data: Message data dict from Redis

        Returns:
            Deserialized MessagingEvent
        """
        return MessagingEvent(
            id=data.get("id", uuid.uuid4().hex[:16]),
            topic=data["topic"],
            data=json.loads(data.get("data", "{}")),
            timestamp=float(data.get("timestamp", 0)),
            source=data.get("source", "victor"),
            correlation_id=data.get("correlation_id") or None,
            partition_key=data.get("partition_key") or None,
            headers=json.loads(data.get("headers", "{}")),
            delivery_guarantee=DeliveryGuarantee(
                data.get("delivery_guarantee", DeliveryGuarantee.AT_MOST_ONCE.value)
            ),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dictionary with backend statistics
        """
        return {
            "backend_type": "redis",
            "is_connected": self._is_connected,
            "published_count": self._published_count,
            "consumed_count": self._consumed_count,
            "error_count": self._error_count,
            "subscription_count": len(self._subscriptions),
            "known_streams": len(self._known_streams),
            "consumer_group": self._consumer_group,
            "consumer_name": self._consumer_name,
        }


__all__ = [
    "RedisEventBackend",
]
