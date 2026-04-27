"""Tests for ToolPipeline, ToolCallResult, ToolPipelineConfig."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.tool_pipeline import (
    ToolCallResult,
    ToolPipeline,
    ToolPipelineConfig,
    PipelineExecutionResult,
    LRUToolCache,
)
from victor.agent.tool_executor import ToolExecutionResult


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.is_tool_enabled.return_value = True
    return registry


@pytest.fixture
def mock_tool_executor():
    from victor.agent.tool_executor import ToolExecutionResult

    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read",
            success=True,
            result="file contents",
            execution_time=0.01,
        )
    )
    return executor


@pytest.fixture
def pipeline(mock_tool_registry, mock_tool_executor):
    config = ToolPipelineConfig(
        tool_budget=5,
        enable_caching=False,
        enable_output_aggregation=False,
        enable_synthesis_checkpoints=False,
        enable_semantic_caching=False,
    )
    return ToolPipeline(
        tool_registry=mock_tool_registry,
        tool_executor=mock_tool_executor,
        config=config,
    )


class TestToolPipelineConfig:
    def test_default_budget(self):
        config = ToolPipelineConfig()
        assert config.tool_budget > 0

    def test_custom_budget(self):
        config = ToolPipelineConfig(tool_budget=10)
        assert config.tool_budget == 10


class TestToolPipelineInit:
    def test_initial_state(self, pipeline):
        assert pipeline.calls_used == 0
        assert pipeline.calls_remaining == 5
        assert pipeline.executed_tools == []

    def test_is_valid_tool_name(self, pipeline):
        assert pipeline.is_valid_tool_name("read") is True
        assert pipeline.is_valid_tool_name("read_file") is True
        assert pipeline.is_valid_tool_name("") is False
        assert pipeline.is_valid_tool_name("Read") is False  # must be lowercase
        assert pipeline.is_valid_tool_name("123abc") is False

    def test_reset(self, pipeline):
        pipeline._calls_used = 3
        pipeline._executed_tools.append("read")
        pipeline.reset()
        assert pipeline.calls_used == 0
        assert pipeline.executed_tools == []


class TestExecuteToolCalls:
    async def test_single_successful_call(self, pipeline):
        tool_calls = [{"name": "read", "arguments": {"path": "/tmp/f.py"}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert isinstance(result, PipelineExecutionResult)
        assert result.total_calls == 1
        assert result.budget_exhausted is False

    async def test_invalid_tool_call_structure(self, pipeline):
        tool_calls = ["not_a_dict"]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].success is False
        assert result.results[0].skipped is True

    async def test_missing_tool_name(self, pipeline):
        tool_calls = [{"arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].skipped is True
        assert "missing name" in result.results[0].skip_reason.lower()

    async def test_budget_enforcement(self, pipeline):
        pipeline.config.tool_budget = 2
        pipeline._calls_used = 2
        tool_calls = [{"name": "read", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].skipped is True
        assert "budget" in result.results[0].skip_reason.lower()
        assert result.results[0].outcome_kind == "budget_exhausted"
        assert result.results[0].block_source == "tool_budget"
        assert result.results[0].retryable is False

    async def test_unknown_tool_skipped(self, pipeline):
        pipeline.tools.is_tool_enabled.return_value = False
        tool_calls = [{"name": "nonexistent", "arguments": {}}]
        result = await pipeline.execute_tool_calls(tool_calls)
        assert result.results[0].skipped is True

    async def test_callbacks_invoked(self, pipeline):
        on_start = MagicMock()
        on_complete = MagicMock()
        pipeline.on_tool_start = on_start
        pipeline.on_tool_complete = on_complete
        tool_calls = [{"name": "read", "arguments": {"path": "/tmp/f.py"}}]
        await pipeline.execute_tool_calls(tool_calls)
        on_start.assert_called()
        on_complete.assert_called()

    async def test_legacy_read_file_uses_exact_dedup_key(self, pipeline):
        await pipeline.execute_tool_calls(
            [{"name": "read_file", "arguments": {"path": "/tmp/f.py"}}]
        )
        assert "/tmp/f.py:None:None" in pipeline._read_file_timestamps

    async def test_write_file_invalidates_read_dedup_key(self, pipeline, mock_tool_executor):
        await pipeline.execute_tool_calls(
            [{"name": "read_file", "arguments": {"path": "/tmp/f.py"}}]
        )
        assert "/tmp/f.py:None:None" in pipeline._read_file_timestamps

        mock_tool_executor.execute.return_value = ToolExecutionResult(
            tool_name="write",
            success=True,
            result="updated",
            execution_time=0.01,
        )
        await pipeline.execute_tool_calls(
            [{"name": "write_file", "arguments": {"path": "/tmp/f.py", "content": "x"}}]
        )

        assert "/tmp/f.py:None:None" not in pipeline._read_file_timestamps

    async def test_duplicate_read_sets_structured_skip_metadata(self, pipeline):
        with patch.object(pipeline, "_is_duplicate_read", return_value=True):
            result = await pipeline.execute_tool_calls(
                [{"name": "read", "arguments": {"path": "/tmp/f.py"}}]
            )

        call_result = result.results[0]
        assert call_result.skipped is True
        assert call_result.success is True
        assert call_result.outcome_kind == "duplicate_read"
        assert call_result.block_source == "session_read_dedup"
        assert call_result.retryable is True
        assert "already read with the same offset/limit" in call_result.user_message


class TestLRUToolCache:
    def test_set_and_get(self):
        cache = LRUToolCache(max_size=10, ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_eviction_on_max_size(self):
        cache = LRUToolCache(max_size=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("c") == 3

    def test_clear(self):
        cache = LRUToolCache(max_size=10, ttl_seconds=60)
        cache.set("k", "v")
        cache.clear()
        assert len(cache) == 0
