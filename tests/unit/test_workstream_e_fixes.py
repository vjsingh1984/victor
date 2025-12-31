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

"""Tests for Workstream E fixes - OpenAI Codex feedback.

This module tests three fixes:
1. Event Bus Silent Drops - DROP_OLDEST should log at WARNING level (rate-limited)
2. Tool Pipeline Rate Limiter - Token bucket rate limiting for tool execution
3. LRU Tool Cache - Replace FIFO with LRU eviction for idempotent tool cache
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fix 1: Event Bus Silent Drops Tests
# =============================================================================


class TestEventBusDropOldestWarning:
    """Tests for Event Bus DROP_OLDEST warning logging."""

    @pytest.fixture
    def small_queue_event_bus(self):
        """Create an event bus with a small queue for testing."""
        import asyncio
        from victor.observability.event_bus import (
            BackpressureStrategy,
            EventBus,
        )

        # Reset singleton for clean test
        EventBus.reset_instance()

        # Get the singleton instance
        bus = EventBus.get_instance()

        # Configure it for testing with small queue and DROP_OLDEST
        bus.configure_backpressure(
            strategy=BackpressureStrategy.DROP_OLDEST,
            queue_maxsize=2,
        )

        # Reset internal state for testing
        bus._last_drop_warning = 0
        bus._drop_warning_interval = 60

        yield bus

        # Clean up
        EventBus.reset_instance()

    @pytest.mark.asyncio
    async def test_drop_oldest_logs_warning_on_first_drop(self, small_queue_event_bus, caplog):
        """Test that DROP_OLDEST logs a WARNING on first event drop."""
        from victor.observability.event_bus import EventCategory, VictorEvent

        bus = small_queue_event_bus

        # Fill the queue
        event1 = VictorEvent(category=EventCategory.TOOL, name="event1")
        event2 = VictorEvent(category=EventCategory.TOOL, name="event2")
        await bus.queue_event_async(event1)
        await bus.queue_event_async(event2)

        # This should trigger DROP_OLDEST
        with caplog.at_level(logging.WARNING):
            event3 = VictorEvent(category=EventCategory.TOOL, name="event3")
            await bus.queue_event_async(event3)

        # Should have a WARNING log about dropping events
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "dropping" in r.message.lower() or "drop" in r.message.lower() for r in warning_logs
        ), "Expected WARNING log about dropping events"

    @pytest.mark.asyncio
    async def test_drop_oldest_warning_is_rate_limited(self, small_queue_event_bus, caplog):
        """Test that DROP_OLDEST warnings are rate-limited (max 1 per minute)."""
        from victor.observability.event_bus import EventCategory, VictorEvent

        bus = small_queue_event_bus

        # Fill the queue
        event1 = VictorEvent(category=EventCategory.TOOL, name="event1")
        event2 = VictorEvent(category=EventCategory.TOOL, name="event2")
        await bus.queue_event_async(event1)
        await bus.queue_event_async(event2)

        with caplog.at_level(logging.WARNING):
            # Trigger multiple drops rapidly
            for i in range(5):
                event = VictorEvent(category=EventCategory.TOOL, name=f"overflow_{i}")
                await bus.queue_event_async(event)

        # Should only have 1 WARNING (rate-limited)
        warning_logs = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and ("dropping" in r.message.lower() or "drop" in r.message.lower())
        ]
        assert len(warning_logs) <= 1, "Expected rate-limited warnings (max 1)"

    @pytest.mark.asyncio
    async def test_drop_oldest_warning_interval_respected(self, small_queue_event_bus, caplog):
        """Test that warnings are emitted again after interval passes."""
        from victor.observability.event_bus import EventCategory, VictorEvent

        bus = small_queue_event_bus
        # Set a very short interval for testing
        bus._drop_warning_interval = 0.1  # 100ms

        # Fill the queue
        event1 = VictorEvent(category=EventCategory.TOOL, name="event1")
        event2 = VictorEvent(category=EventCategory.TOOL, name="event2")
        await bus.queue_event_async(event1)
        await bus.queue_event_async(event2)

        with caplog.at_level(logging.WARNING):
            # First drop
            event3 = VictorEvent(category=EventCategory.TOOL, name="event3")
            await bus.queue_event_async(event3)

            # Wait for interval
            await asyncio.sleep(0.15)

            # Second drop (should trigger another warning)
            event4 = VictorEvent(category=EventCategory.TOOL, name="event4")
            await bus.queue_event_async(event4)

        # Should have 2 warnings (one before interval, one after)
        warning_logs = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and ("dropping" in r.message.lower() or "drop" in r.message.lower())
        ]
        assert len(warning_logs) == 2, "Expected 2 warnings after interval passed"


# =============================================================================
# Fix 2: Tool Pipeline Rate Limiter Tests
# =============================================================================


class TestToolRateLimiter:
    """Tests for Tool Pipeline Rate Limiter."""

    def test_rate_limiter_initialization(self):
        """Test that ToolRateLimiter initializes with correct defaults."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter()
        assert limiter.rate == 10.0
        assert limiter.burst == 5
        assert limiter.tokens == 5

    def test_rate_limiter_custom_params(self):
        """Test ToolRateLimiter with custom rate and burst."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=20.0, burst=10)
        assert limiter.rate == 20.0
        assert limiter.burst == 10
        assert limiter.tokens == 10

    def test_rate_limiter_acquire_uses_token(self):
        """Test that acquire() uses a token."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=10.0, burst=5)
        initial_tokens = limiter.tokens

        result = limiter.acquire()

        assert result is True
        assert limiter.tokens < initial_tokens

    def test_rate_limiter_acquire_fails_when_empty(self):
        """Test that acquire() returns False when no tokens available."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=10.0, burst=2)

        # Use all tokens
        limiter.acquire()
        limiter.acquire()

        # Should fail now
        result = limiter.acquire()
        assert result is False

    def test_rate_limiter_tokens_replenish_over_time(self):
        """Test that tokens replenish over time."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=100.0, burst=5)  # High rate for fast test

        # Use all tokens
        for _ in range(5):
            limiter.acquire()

        assert limiter.tokens < 1

        # Wait for replenishment (100/sec = 1 token per 10ms)
        time.sleep(0.05)  # 50ms = ~5 tokens at 100/sec

        # Should be able to acquire again
        result = limiter.acquire()
        assert result is True

    def test_rate_limiter_tokens_dont_exceed_burst(self):
        """Test that tokens don't exceed burst limit."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=1000.0, burst=3)

        # Wait for potential over-accumulation
        time.sleep(0.1)

        # Tokens should still be at burst limit
        assert limiter.tokens <= limiter.burst

    def test_rate_limiter_is_thread_safe(self):
        """Test that ToolRateLimiter is thread-safe."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=1.0, burst=10)
        results = []

        def acquire_tokens():
            for _ in range(20):
                results.append(limiter.acquire())

        threads = [threading.Thread(target=acquire_tokens) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 10 True results (burst limit)
        # plus any that replenished during execution
        true_count = sum(1 for r in results if r)
        # With 1 token/sec and ~instant execution, we expect ~10-15 successes
        assert 10 <= true_count <= 20

    @pytest.mark.asyncio
    async def test_rate_limiter_wait(self):
        """Test that wait() blocks until a token is available."""
        from victor.agent.tool_pipeline import ToolRateLimiter

        limiter = ToolRateLimiter(rate=100.0, burst=1)

        # Use the token
        limiter.acquire()

        # Start timing
        start = time.monotonic()

        # Wait for a token
        await limiter.wait()

        elapsed = time.monotonic() - start

        # Should have waited at least 1/rate seconds (10ms for rate=100)
        assert elapsed >= 0.01


# =============================================================================
# Fix 3: LRU Tool Cache Tests
# =============================================================================


class TestLRUToolCache:
    """Tests for LRU Tool Cache."""

    def test_lru_cache_initialization(self):
        """Test LRUToolCache initializes correctly."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=100)
        assert cache._max_size == 100
        assert len(cache._cache) == 0

    def test_lru_cache_set_and_get(self):
        """Test basic set and get operations."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=10)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_lru_cache_returns_none_for_missing_key(self):
        """Test that get returns None for missing keys."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_cache_evicts_oldest_when_full(self):
        """Test that LRU cache evicts oldest item when full."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Cache is full, add one more
        cache.set("key4", "value4")

        # Oldest (key1) should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_cache_access_updates_order(self):
        """Test that accessing an item moves it to most recently used."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it most recently used
        _ = cache.get("key1")

        # Add a new item, should evict key2 (now oldest)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_cache_set_existing_updates_order(self):
        """Test that setting an existing key updates its order."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Update key1 with new value (moves to end)
        cache.set("key1", "new_value1")

        # Add new item, should evict key2 (now oldest)
        cache.set("key4", "value4")

        assert cache.get("key1") == "new_value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_lru_cache_respects_max_size(self):
        """Test that cache never exceeds max_size."""
        from victor.agent.tool_pipeline import LRUToolCache

        cache = LRUToolCache(max_size=5)

        for i in range(20):
            cache.set(f"key{i}", f"value{i}")

        assert len(cache._cache) == 5


class TestToolPipelineIdempotentCacheLRU:
    """Tests for idempotent tool cache using LRU eviction."""

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        registry = MagicMock()
        registry.is_tool_enabled = MagicMock(return_value=True)
        return registry

    @pytest.fixture
    def mock_tool_executor(self):
        """Create a mock tool executor."""
        from victor.agent.tool_executor import ToolExecutionResult

        executor = MagicMock()
        executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="read",
                success=True,
                result={"content": "file content"},
                error=None,
            )
        )
        return executor

    @pytest.fixture
    def pipeline(self, mock_tool_registry, mock_tool_executor):
        """Create a tool pipeline for testing."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        return ToolPipeline(
            tool_registry=mock_tool_registry,
            tool_executor=mock_tool_executor,
            config=ToolPipelineConfig(
                tool_budget=50,
                enable_idempotent_caching=True,
                idempotent_cache_max_size=3,  # Small for testing
            ),
        )

    @pytest.mark.asyncio
    async def test_idempotent_cache_uses_lru_eviction(self, pipeline, mock_tool_executor):
        """Test that idempotent cache uses LRU eviction instead of FIFO."""
        from victor.agent.tool_executor import ToolExecutionResult

        # Set up executor to return different results for different files
        call_count = 0

        async def mock_execute(tool_name, arguments, context):
            nonlocal call_count
            call_count += 1
            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result={"content": f"content_{call_count}"},
                error=None,
            )

        mock_tool_executor.execute = AsyncMock(side_effect=mock_execute)

        # Read 3 files (fills cache of size 3)
        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "file1.py"}}], {})
        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "file2.py"}}], {})
        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "file3.py"}}], {})

        # Access file1 to make it most recently used
        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "file1.py"}}], {})

        # Read a 4th file (should evict file2, not file1)
        await pipeline.execute_tool_calls([{"name": "read", "arguments": {"path": "file4.py"}}], {})

        # Verify LRU cache state directly:
        # file1 should be in cache (most recently accessed before file4)
        # file2 should NOT be in cache (least recently used, evicted)
        # file3 should be in cache
        # file4 should be in cache (just added)
        # Note: We need to use normalized args because the cache uses normalized signatures
        norm_args1, _ = pipeline._normalize_arguments("read", {"path": "file1.py"})
        norm_args2, _ = pipeline._normalize_arguments("read", {"path": "file2.py"})
        norm_args3, _ = pipeline._normalize_arguments("read", {"path": "file3.py"})
        norm_args4, _ = pipeline._normalize_arguments("read", {"path": "file4.py"})

        sig_file1 = pipeline._get_call_signature("read", norm_args1)
        sig_file2 = pipeline._get_call_signature("read", norm_args2)
        sig_file3 = pipeline._get_call_signature("read", norm_args3)
        sig_file4 = pipeline._get_call_signature("read", norm_args4)

        # file1, file3, file4 should be in cache
        assert pipeline._idempotent_cache.get(sig_file1) is not None, "file1 should be in LRU cache"
        assert pipeline._idempotent_cache.get(sig_file3) is not None, "file3 should be in LRU cache"
        assert pipeline._idempotent_cache.get(sig_file4) is not None, "file4 should be in LRU cache"

        # file2 should have been evicted (LRU victim)
        assert (
            pipeline._idempotent_cache.get(sig_file2) is None
        ), "file2 should have been evicted from LRU cache"
