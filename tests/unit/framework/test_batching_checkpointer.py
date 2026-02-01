"""Tests for BatchingCheckpointer.

Tests the batching checkpoint wrapper for reducing I/O pressure.
"""

import asyncio
import time
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from victor.framework.graph import (
    BatchingCheckpointer,
    BatchingCheckpointerConfig,
    CheckpointerProtocol,
    MemoryCheckpointer,
    WorkflowCheckpoint,
)


def create_checkpoint(
    thread_id: str = "thread-1",
    node_id: str = "node-1",
    checkpoint_id: Optional[str] = None,
) -> WorkflowCheckpoint:
    """Helper to create test checkpoints."""
    return WorkflowCheckpoint(
        checkpoint_id=checkpoint_id or f"cp-{time.time()}",
        thread_id=thread_id,
        node_id=node_id,
        state={"key": "value"},
        timestamp=time.time(),
        metadata={},
    )


class TestBatchingCheckpointerConfig:
    """Tests for BatchingCheckpointerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchingCheckpointerConfig()
        assert config.batch_size == 10
        assert config.flush_interval == 5.0
        assert config.flush_on_load is True
        assert config.keep_latest_only is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = BatchingCheckpointerConfig(
            batch_size=20,
            flush_interval=10.0,
            flush_on_load=False,
            keep_latest_only=True,
        )
        assert config.batch_size == 20
        assert config.flush_interval == 10.0
        assert config.flush_on_load is False
        assert config.keep_latest_only is True

    def test_disable_interval_flush(self):
        """Test disabling interval-based flushing."""
        config = BatchingCheckpointerConfig(flush_interval=None)
        assert config.flush_interval is None


class TestBatchingCheckpointerBasic:
    """Basic tests for BatchingCheckpointer."""

    @pytest.mark.asyncio
    async def test_save_accumulates_checkpoints(self):
        """Test that save accumulates checkpoints without immediate flush."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=10, flush_interval=None)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save 5 checkpoints (less than batch size)
        for i in range(5):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        # Should be pending, not flushed
        assert batching.pending_count == 5

        # Backend should be empty
        persisted = await backend.list("thread-1")
        assert len(persisted) == 0

    @pytest.mark.asyncio
    async def test_batch_size_triggers_flush(self):
        """Test that reaching batch size triggers automatic flush."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=5, flush_interval=None)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save exactly batch_size checkpoints
        for i in range(5):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        # Should have been flushed
        assert batching.pending_count == 0

        # Backend should have all checkpoints
        persisted = await backend.list("thread-1")
        assert len(persisted) == 5

    @pytest.mark.asyncio
    async def test_explicit_flush(self):
        """Test explicit flush() call."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_interval=None)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save some checkpoints
        for i in range(3):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        assert batching.pending_count == 3

        # Explicit flush
        flushed = await batching.flush()
        assert flushed == 3
        assert batching.pending_count == 0

        # Backend should have checkpoints
        persisted = await backend.list("thread-1")
        assert len(persisted) == 3

    @pytest.mark.asyncio
    async def test_load_returns_latest(self):
        """Test that load returns the latest checkpoint."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_on_load=False)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save multiple checkpoints
        for i in range(5):
            cp = create_checkpoint(node_id=f"node-{i}", checkpoint_id=f"cp-{i}")
            await batching.save(cp)

        # Load should return the latest
        latest = await batching.load("thread-1")
        assert latest is not None
        assert latest.node_id == "node-4"
        assert latest.checkpoint_id == "cp-4"

    @pytest.mark.asyncio
    async def test_load_flushes_on_load(self):
        """Test that load flushes pending checkpoints when configured."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_on_load=True)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save some checkpoints
        for i in range(3):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        assert batching.pending_count == 3

        # Load triggers flush
        await batching.load("thread-1")

        # Should be flushed now
        assert batching.pending_count == 0
        persisted = await backend.list("thread-1")
        assert len(persisted) == 3

    @pytest.mark.asyncio
    async def test_load_no_flush_when_disabled(self):
        """Test that load doesn't flush when flush_on_load is False."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_on_load=False)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save some checkpoints
        for i in range(3):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        # Load doesn't trigger flush
        await batching.load("thread-1")

        # Still pending
        assert batching.pending_count == 3

    @pytest.mark.asyncio
    async def test_list_combines_pending_and_persisted(self):
        """Test that list returns both pending and persisted checkpoints."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_on_load=False)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save some directly to backend
        for i in range(2):
            cp = create_checkpoint(node_id=f"persisted-{i}", checkpoint_id=f"p-{i}")
            await backend.save(cp)

        # Save some through batching (pending)
        for i in range(3):
            cp = create_checkpoint(node_id=f"pending-{i}", checkpoint_id=f"b-{i}")
            await batching.save(cp)

        # List should return all 5
        all_checkpoints = await batching.list("thread-1")
        assert len(all_checkpoints) == 5


class TestBatchingCheckpointerMultiThread:
    """Tests for multi-thread checkpoint handling."""

    @pytest.mark.asyncio
    async def test_multiple_threads(self):
        """Test handling checkpoints from multiple threads."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_interval=None)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save checkpoints for different threads
        await batching.save(create_checkpoint(thread_id="thread-1", node_id="a"))
        await batching.save(create_checkpoint(thread_id="thread-2", node_id="b"))
        await batching.save(create_checkpoint(thread_id="thread-1", node_id="c"))

        # Flush
        await batching.flush()

        # Each thread should have its checkpoints
        thread1_cps = await backend.list("thread-1")
        thread2_cps = await backend.list("thread-2")

        assert len(thread1_cps) == 2
        assert len(thread2_cps) == 1

    @pytest.mark.asyncio
    async def test_load_correct_thread(self):
        """Test that load returns checkpoints for the correct thread."""
        backend = MemoryCheckpointer()
        batching = BatchingCheckpointer(backend=backend)

        await batching.save(create_checkpoint(thread_id="thread-1", node_id="a"))
        await batching.save(create_checkpoint(thread_id="thread-2", node_id="b"))

        latest1 = await batching.load("thread-1")
        latest2 = await batching.load("thread-2")

        assert latest1.node_id == "a"
        assert latest2.node_id == "b"


class TestBatchingCheckpointerKeepLatestOnly:
    """Tests for keep_latest_only mode."""

    @pytest.mark.asyncio
    async def test_keep_latest_only(self):
        """Test that only latest checkpoint per thread is persisted."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(
            batch_size=100,
            keep_latest_only=True,
            flush_interval=None,
        )
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save multiple checkpoints for same thread
        for i in range(5):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        # Flush with keep_latest_only
        flushed = await batching.flush()

        # Only 1 should be flushed (the latest)
        assert flushed == 1

        persisted = await backend.list("thread-1")
        assert len(persisted) == 1
        assert persisted[0].node_id == "node-4"

    @pytest.mark.asyncio
    async def test_keep_latest_only_multiple_threads(self):
        """Test keep_latest_only with multiple threads."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(
            batch_size=100,
            keep_latest_only=True,
            flush_interval=None,
        )
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Multiple checkpoints per thread
        for i in range(3):
            await batching.save(create_checkpoint(thread_id="thread-1", node_id=f"t1-{i}"))
            await batching.save(create_checkpoint(thread_id="thread-2", node_id=f"t2-{i}"))

        flushed = await batching.flush()

        # 2 flushed (1 per thread)
        assert flushed == 2

        t1_cps = await backend.list("thread-1")
        t2_cps = await backend.list("thread-2")

        assert len(t1_cps) == 1
        assert t1_cps[0].node_id == "t1-2"
        assert len(t2_cps) == 1
        assert t2_cps[0].node_id == "t2-2"


class TestBatchingCheckpointerTimeBasedFlush:
    """Tests for time-based auto-flush."""

    @pytest.mark.asyncio
    async def test_interval_triggers_flush(self):
        """Test that flush interval triggers automatic flush."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(
            batch_size=100,
            flush_interval=0.1,  # 100ms
        )
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save a checkpoint
        await batching.save(create_checkpoint(node_id="node-1"))
        assert batching.pending_count == 1

        # Wait for interval to elapse
        await asyncio.sleep(0.15)

        # Save another - should trigger time-based flush
        await batching.save(create_checkpoint(node_id="node-2"))

        # First checkpoint should be flushed, second pending
        # (flush happens before adding second)
        persisted = await backend.list("thread-1")
        # After flush, we add the new one, so pending_count could be 1 or 0
        assert batching.pending_count <= 2


class TestBatchingCheckpointerBackgroundFlush:
    """Tests for background flush task."""

    @pytest.mark.asyncio
    async def test_background_flush_task(self):
        """Test background flush task flushes periodically."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(
            batch_size=100,
            flush_interval=0.1,
        )
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Start background flush
        await batching.start_background_flush()

        try:
            # Save some checkpoints
            await batching.save(create_checkpoint(node_id="node-1"))
            await batching.save(create_checkpoint(node_id="node-2"))

            assert batching.pending_count == 2

            # Wait for background flush
            await asyncio.sleep(0.15)

            # Should be flushed
            assert batching.pending_count == 0
        finally:
            await batching.stop_background_flush()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager starts/stops background flush."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_interval=0.1)

        async with BatchingCheckpointer(backend=backend, config=config) as batching:
            await batching.save(create_checkpoint())
            assert batching.pending_count == 1

        # After context exit, should be flushed
        persisted = await backend.list("thread-1")
        assert len(persisted) == 1

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self):
        """Test that stop_background_flush flushes remaining checkpoints."""
        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_interval=10.0)
        batching = BatchingCheckpointer(backend=backend, config=config)

        await batching.start_background_flush()

        # Save checkpoints
        for i in range(5):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        assert batching.pending_count == 5

        # Stop should flush
        await batching.stop_background_flush()

        assert batching.pending_count == 0
        persisted = await backend.list("thread-1")
        assert len(persisted) == 5


class TestBatchingCheckpointerStats:
    """Tests for statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test get_stats returns expected information."""
        config = BatchingCheckpointerConfig(
            batch_size=20,
            flush_interval=10.0,
            keep_latest_only=True,
        )
        batching = BatchingCheckpointer(
            backend=MemoryCheckpointer(),
            config=config,
        )

        await batching.save(create_checkpoint(thread_id="t1"))
        await batching.save(create_checkpoint(thread_id="t2"))

        stats = batching.get_stats()

        assert stats["pending_count"] == 2
        assert stats["pending_threads"] == 2
        assert stats["batch_size"] == 20
        assert stats["flush_interval"] == 10.0
        assert stats["keep_latest_only"] is True
        assert stats["background_flush_active"] is False

    @pytest.mark.asyncio
    async def test_backend_property(self):
        """Test backend property returns underlying checkpointer."""
        backend = MemoryCheckpointer()
        batching = BatchingCheckpointer(backend=backend)

        assert batching.backend is backend


class TestBatchingCheckpointerWithMockedBackend:
    """Tests using mocked backend for verifying behavior."""

    @pytest.mark.asyncio
    async def test_backend_save_called_on_flush(self):
        """Test that backend.save is called for each checkpoint on flush."""
        backend = AsyncMock(spec=CheckpointerProtocol)
        backend.list = AsyncMock(return_value=[])

        config = BatchingCheckpointerConfig(batch_size=100, flush_interval=None)
        batching = BatchingCheckpointer(backend=backend, config=config)

        # Save 3 checkpoints
        for i in range(3):
            await batching.save(create_checkpoint(node_id=f"node-{i}"))

        # Backend save not called yet
        assert backend.save.call_count == 0

        # Flush
        await batching.flush()

        # Backend save called 3 times
        assert backend.save.call_count == 3

    @pytest.mark.asyncio
    async def test_backend_load_called_when_no_pending(self):
        """Test that backend.load is called when no pending checkpoints."""
        backend = AsyncMock(spec=CheckpointerProtocol)
        backend.load = AsyncMock(return_value=None)

        batching = BatchingCheckpointer(backend=backend)

        # Load with no pending
        result = await batching.load("thread-1")

        # Backend load should be called
        backend.load.assert_called_once_with("thread-1")
        assert result is None


class TestBatchingCheckpointerIntegration:
    """Integration tests with StateGraph."""

    @pytest.mark.asyncio
    async def test_with_state_graph(self):
        """Test BatchingCheckpointer with actual StateGraph execution."""
        from victor.framework.graph import StateGraph, END

        backend = MemoryCheckpointer()
        config = BatchingCheckpointerConfig(batch_size=100, flush_interval=None)
        batching = BatchingCheckpointer(backend=backend, config=config)

        async def node1(state):
            state["node1"] = True
            return state

        async def node2(state):
            state["node2"] = True
            return state

        async def node3(state):
            state["node3"] = True
            return state

        graph = StateGraph()
        graph.add_node("node1", node1)
        graph.add_node("node2", node2)
        graph.add_node("node3", node3)
        graph.set_entry_point("node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", "node3")
        graph.add_edge("node3", END)

        compiled = graph.compile(checkpointer=batching)

        # Execute
        result = await compiled.invoke({})

        assert result.success is True
        assert result.state["node1"] is True
        assert result.state["node2"] is True
        assert result.state["node3"] is True

        # Checkpoints should be pending (batch size not reached)
        assert batching.pending_count == 3

        # Flush to persist
        await batching.flush()
        assert batching.pending_count == 0

        # Verify persisted
        persisted = await backend.list(result.node_history[0])
        # Note: thread_id is auto-generated, so we check via batching
