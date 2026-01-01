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

"""Tests for StreamingToolAdapter.

Tests the streaming adapter that wraps ToolPipeline for unified
streaming tool execution with real-time chunk emission.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.protocols import StreamingToolChunk
from victor.agent.streaming_tool_adapter import (
    StreamingToolAdapter,
    create_streaming_tool_adapter,
)


@dataclass
class MockToolCallResult:
    """Mock ToolCallResult for testing."""

    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None
    cached: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    normalization_applied: Optional[str] = None


@dataclass
class MockPipelineResult:
    """Mock PipelineExecutionResult for testing."""

    results: List[MockToolCallResult]
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    budget_exhausted: bool = False


@pytest.fixture
def mock_pipeline():
    """Create a mock ToolPipeline."""
    pipeline = MagicMock()
    pipeline.calls_used = 0
    pipeline.calls_remaining = 100
    pipeline.config = MagicMock()
    pipeline.config.tool_budget = 100
    pipeline.execute_tool_calls = AsyncMock()
    pipeline.reset = MagicMock()
    return pipeline


class TestStreamingToolAdapter:
    """Test suite for StreamingToolAdapter."""

    @pytest.mark.asyncio
    async def test_execute_streaming_single_success(self, mock_pipeline):
        """Test streaming a single successful tool call."""
        # Setup mock result
        mock_result = MockToolCallResult(
            tool_name="test_tool",
            arguments={"arg1": "value1"},
            success=True,
            result="Tool output",
        )
        mock_pipeline.execute_tool_calls.return_value = MockPipelineResult(
            results=[mock_result], successful_calls=1
        )

        adapter = StreamingToolAdapter(mock_pipeline)

        chunks = []
        async for chunk in adapter.execute_streaming_single(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
        ):
            chunks.append(chunk)

        # Should have 2 chunks: start and result
        assert len(chunks) == 2
        assert chunks[0].chunk_type == "start"
        assert chunks[0].tool_name == "test_tool"
        assert chunks[0].is_final is False
        assert chunks[1].chunk_type == "result"
        assert chunks[1].is_final is True
        assert chunks[1].content.success is True

    @pytest.mark.asyncio
    async def test_execute_streaming_single_error(self, mock_pipeline):
        """Test streaming a single failed tool call."""
        mock_result = MockToolCallResult(
            tool_name="failing_tool",
            arguments={},
            success=False,
            error="Tool failed",
        )
        mock_pipeline.execute_tool_calls.return_value = MockPipelineResult(
            results=[mock_result], failed_calls=1
        )

        adapter = StreamingToolAdapter(mock_pipeline)

        chunks = []
        async for chunk in adapter.execute_streaming_single(
            tool_name="failing_tool",
            tool_args={},
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].chunk_type == "start"
        assert chunks[1].chunk_type == "error"
        assert chunks[1].is_final is True
        assert "Tool failed" in chunks[1].content

    @pytest.mark.asyncio
    async def test_execute_streaming_single_cache_hit(self, mock_pipeline):
        """Test streaming with cache hit."""
        mock_result = MockToolCallResult(
            tool_name="cached_tool",
            arguments={},
            success=True,
            result="Cached result",
            cached=True,
        )
        mock_pipeline.execute_tool_calls.return_value = MockPipelineResult(
            results=[mock_result], successful_calls=1
        )

        adapter = StreamingToolAdapter(mock_pipeline)

        chunks = []
        async for chunk in adapter.execute_streaming_single(
            tool_name="cached_tool",
            tool_args={},
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].chunk_type == "start"
        assert chunks[1].chunk_type == "cache_hit"
        assert chunks[1].is_final is True
        assert chunks[1].metadata.get("cached") is True

    @pytest.mark.asyncio
    async def test_execute_streaming_single_skipped(self, mock_pipeline):
        """Test streaming with skipped tool call."""
        mock_result = MockToolCallResult(
            tool_name="skipped_tool",
            arguments={},
            success=False,
            skipped=True,
            skip_reason="Budget exhausted",
        )
        mock_pipeline.execute_tool_calls.return_value = MockPipelineResult(
            results=[mock_result], skipped_calls=1
        )

        adapter = StreamingToolAdapter(mock_pipeline)

        chunks = []
        async for chunk in adapter.execute_streaming_single(
            tool_name="skipped_tool",
            tool_args={},
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].chunk_type == "start"
        assert chunks[1].chunk_type == "error"
        assert chunks[1].is_final is True
        assert chunks[1].metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_execute_streaming_multiple_tools(self, mock_pipeline):
        """Test streaming multiple tool calls."""
        mock_results = [
            MockToolCallResult(tool_name="tool1", arguments={}, success=True, result="Result 1"),
            MockToolCallResult(tool_name="tool2", arguments={}, success=True, result="Result 2"),
        ]

        # Mock to return results one at a time (called twice)
        mock_pipeline.execute_tool_calls.side_effect = [
            MockPipelineResult(results=[mock_results[0]], successful_calls=1),
            MockPipelineResult(results=[mock_results[1]], successful_calls=1),
        ]

        adapter = StreamingToolAdapter(mock_pipeline)

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]

        chunks = []
        async for chunk in adapter.execute_streaming(tool_calls):
            chunks.append(chunk)

        # Should have 4 chunks: 2 starts and 2 results
        assert len(chunks) == 4
        assert chunks[0].chunk_type == "start"
        assert chunks[0].tool_name == "tool1"
        assert chunks[1].chunk_type == "result"
        assert chunks[2].chunk_type == "start"
        assert chunks[2].tool_name == "tool2"
        assert chunks[3].chunk_type == "result"

    @pytest.mark.asyncio
    async def test_execute_streaming_budget_exhausted(self, mock_pipeline):
        """Test that streaming stops when budget is exhausted."""
        mock_result = MockToolCallResult(
            tool_name="tool1", arguments={}, success=True, result="Result"
        )
        mock_pipeline.execute_tool_calls.return_value = MockPipelineResult(
            results=[mock_result], successful_calls=1
        )

        # Set budget to exhausted after first call
        mock_pipeline.calls_used = 100
        mock_pipeline.config.tool_budget = 100

        adapter = StreamingToolAdapter(mock_pipeline)

        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},  # Should not execute
        ]

        chunks = []
        async for chunk in adapter.execute_streaming(tool_calls):
            chunks.append(chunk)

        # Should have 3 chunks: start, result, budget_exhausted
        # Tool2 should not execute
        assert len(chunks) == 3
        assert chunks[0].chunk_type == "start"
        assert chunks[0].tool_name == "tool1"
        assert chunks[1].chunk_type == "result"
        assert chunks[2].chunk_type == "error"
        assert chunks[2].metadata.get("budget_exhausted") is True

    @pytest.mark.asyncio
    async def test_execute_streaming_exception_handling(self, mock_pipeline):
        """Test exception handling during execution."""
        mock_pipeline.execute_tool_calls.side_effect = RuntimeError("Unexpected error")

        adapter = StreamingToolAdapter(mock_pipeline)

        chunks = []
        async for chunk in adapter.execute_streaming_single(
            tool_name="failing_tool",
            tool_args={},
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].chunk_type == "start"
        assert chunks[1].chunk_type == "error"
        assert chunks[1].is_final is True
        assert "Unexpected error" in chunks[1].content
        assert chunks[1].metadata.get("exception") == "RuntimeError"

    @pytest.mark.asyncio
    async def test_on_chunk_callback(self, mock_pipeline):
        """Test that on_chunk callback is called for each chunk."""
        mock_result = MockToolCallResult(
            tool_name="test_tool", arguments={}, success=True, result="Result"
        )
        mock_pipeline.execute_tool_calls.return_value = MockPipelineResult(
            results=[mock_result], successful_calls=1
        )

        callback_chunks = []

        def on_chunk(chunk):
            callback_chunks.append(chunk)

        adapter = StreamingToolAdapter(mock_pipeline, on_chunk=on_chunk)

        async for _ in adapter.execute_streaming_single("test_tool", {}):
            pass

        assert len(callback_chunks) == 2
        assert callback_chunks[0].chunk_type == "start"
        assert callback_chunks[1].chunk_type == "result"

    def test_property_delegation(self, mock_pipeline):
        """Test that properties delegate to pipeline."""
        mock_pipeline.calls_used = 5
        mock_pipeline.calls_remaining = 95
        mock_pipeline.config.tool_budget = 100

        adapter = StreamingToolAdapter(mock_pipeline)

        assert adapter.calls_used == 5
        assert adapter.calls_remaining == 95
        assert adapter.budget == 100

    def test_is_budget_exhausted(self, mock_pipeline):
        """Test budget exhaustion check."""
        mock_pipeline.calls_used = 99
        mock_pipeline.config.tool_budget = 100

        adapter = StreamingToolAdapter(mock_pipeline)
        assert adapter.is_budget_exhausted() is False

        mock_pipeline.calls_used = 100
        assert adapter.is_budget_exhausted() is True

    def test_reset(self, mock_pipeline):
        """Test reset delegates to pipeline."""
        adapter = StreamingToolAdapter(mock_pipeline)
        adapter.reset()
        mock_pipeline.reset.assert_called_once()

    def test_factory_function(self, mock_pipeline):
        """Test create_streaming_tool_adapter factory."""
        adapter = create_streaming_tool_adapter(mock_pipeline)
        assert isinstance(adapter, StreamingToolAdapter)
        assert adapter._pipeline is mock_pipeline


class TestStreamingToolChunk:
    """Test StreamingToolChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a StreamingToolChunk."""
        chunk = StreamingToolChunk(
            tool_name="test_tool",
            tool_call_id="abc123",
            chunk_type="start",
            content={"arg": "value"},
        )

        assert chunk.tool_name == "test_tool"
        assert chunk.tool_call_id == "abc123"
        assert chunk.chunk_type == "start"
        assert chunk.content == {"arg": "value"}
        assert chunk.is_final is False
        assert chunk.metadata == {}

    def test_chunk_with_metadata(self):
        """Test chunk with metadata."""
        chunk = StreamingToolChunk(
            tool_name="test_tool",
            tool_call_id="abc123",
            chunk_type="result",
            content="Result",
            is_final=True,
            metadata={"execution_time_ms": 100.5},
        )

        assert chunk.is_final is True
        assert chunk.metadata["execution_time_ms"] == 100.5
