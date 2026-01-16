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

"""Tests for ResponseCoordinator.

Tests the response processing and sanitization coordination functionality.
"""

import pytest
from unittest.mock import MagicMock, Mock

from victor.agent.coordinators.response_coordinator import (
    ResponseCoordinator,
    IResponseCoordinator,
    ResponseCoordinatorConfig,
    ProcessedResponse,
    ChunkProcessResult,
    ToolCallValidationResult,
)


class MockResponseSanitizer:
    """Mock ResponseSanitizer for testing."""

    def __init__(self, garbage_content=False):
        self._garbage_content = garbage_content
        self.sanitize_calls = []
        self.garbage_calls = []

    def sanitize(self, content: str) -> str:
        """Mock sanitize that tracks calls."""
        self.sanitize_calls.append(content)
        return f"sanitized:{content}"

    def is_garbage_content(self, content: str) -> bool:
        """Mock garbage detection."""
        self.garbage_calls.append(content)
        return self._garbage_content or content.startswith("garbage")

    def is_valid_tool_name(self, name: str) -> bool:
        """Mock tool name validation."""
        return bool(name and not name.startswith("_"))


class MockStreamChunk:
    """Mock stream chunk for testing."""

    def __init__(self, content: str):
        self.content = content


class TestProcessedResponse:
    """Tests for ProcessedResponse dataclass."""

    def test_default_values(self):
        """Test default values for ProcessedResponse."""
        response = ProcessedResponse(content="test")

        assert response.content == "test"
        assert response.tool_calls is None
        assert response.tokens_used == 0.0
        assert response.garbage_detected is False
        assert response.is_final is False

    def test_has_tool_calls(self):
        """Test has_tool_calls method."""
        # No tool calls
        response = ProcessedResponse(content="test")
        assert response.has_tool_calls() is False

        # With tool calls
        response = ProcessedResponse(
            content="test", tool_calls=[{"name": "read_file"}]
        )
        assert response.has_tool_calls() is True

    def test_is_empty(self):
        """Test is_empty method."""
        # Empty content, no tool calls
        response = ProcessedResponse(content="")
        assert response.is_empty() is True

        # Has content
        response = ProcessedResponse(content="test")
        assert response.is_empty() is False

        # Has tool calls
        response = ProcessedResponse(
            content="", tool_calls=[{"name": "read_file"}]
        )
        assert response.is_empty() is False


class TestChunkProcessResult:
    """Tests for ChunkProcessResult dataclass."""

    def test_default_values(self):
        """Test default values for ChunkProcessResult."""
        result = ChunkProcessResult(
            chunk=Mock(),
            consecutive_garbage_count=1,
            garbage_detected=True,
        )

        assert result.consecutive_garbage_count == 1
        assert result.garbage_detected is True
        assert result.should_stop is False


class TestToolCallValidationResult:
    """Tests for ToolCallValidationResult dataclass."""

    def test_default_values(self):
        """Test default values for ToolCallValidationResult."""
        result = ToolCallValidationResult(
            tool_calls=None,
            remaining_content="content",
        )

        assert result.tool_calls is None
        assert result.remaining_content == "content"
        assert result.filtered_count == 0


class TestResponseCoordinatorConfig:
    """Tests for ResponseCoordinatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ResponseCoordinatorConfig()

        assert config.max_garbage_chunks == 3
        assert config.enable_tool_call_extraction is True
        assert config.enable_content_sanitization is True
        assert config.min_content_length == 20

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ResponseCoordinatorConfig(
            max_garbage_chunks=5,
            enable_tool_call_extraction=False,
            enable_content_sanitization=False,
            min_content_length=50,
        )

        assert config.max_garbage_chunks == 5
        assert config.enable_tool_call_extraction is False
        assert config.enable_content_sanitization is False
        assert config.min_content_length == 50


class TestResponseCoordinator:
    """Tests for ResponseCoordinator."""

    @pytest.fixture
    def mock_sanitizer(self):
        """Create mock sanitizer."""
        return MockResponseSanitizer()

    @pytest.fixture
    def mock_tool_adapter(self):
        """Create mock tool adapter."""
        adapter = MagicMock()
        adapter.parse_tool_calls.return_value = MagicMock(
            tool_calls=[],
            remaining_content="content",
            parse_method="mock"
        )
        return adapter

    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = MagicMock()
        registry.list_tools.return_value = ["read_file", "write_file"]
        return registry

    @pytest.fixture
    def coordinator(self, mock_sanitizer, mock_tool_adapter, mock_tool_registry):
        """Create coordinator with mocks."""
        return ResponseCoordinator(
            sanitizer=mock_sanitizer,
            tool_adapter=mock_tool_adapter,
            tool_registry=mock_tool_registry,
        )

    def test_init(self, mock_sanitizer):
        """Test initialization."""
        config = ResponseCoordinatorConfig(enable_content_sanitization=False)
        coordinator = ResponseCoordinator(
            sanitizer=mock_sanitizer,
            config=config,
        )

        assert coordinator._sanitizer == mock_sanitizer
        assert coordinator._config.enable_content_sanitization is False

    def test_sanitize_response(self, coordinator):
        """Test response sanitization."""
        content = "test content"
        result = coordinator.sanitize_response(content)

        assert result == "sanitized:test content"
        assert content in coordinator._sanitizer.sanitize_calls

    def test_sanitize_response_disabled(self, mock_sanitizer):
        """Test sanitization when disabled."""
        config = ResponseCoordinatorConfig(enable_content_sanitization=False)
        coordinator = ResponseCoordinator(
            sanitizer=mock_sanitizer,
            config=config,
        )

        content = "test content"
        result = coordinator.sanitize_response(content)

        # Should return content unchanged
        assert result == content
        assert len(coordinator._sanitizer.sanitize_calls) == 0

    def test_is_garbage_content(self, coordinator):
        """Test garbage content detection."""
        # Normal content
        assert coordinator.is_garbage_content("normal content") is False

        # Garbage content
        assert coordinator.is_garbage_content("garbage: test") is True

    def test_is_valid_tool_name(self, coordinator):
        """Test tool name validation."""
        # Valid tool name
        assert coordinator.is_valid_tool_name("read_file") is True

        # Invalid tool name
        assert coordinator.is_valid_tool_name("_private") is False

    def test_parse_and_validate_tool_calls_no_calls(self, coordinator):
        """Test parsing when no tool calls."""
        result = coordinator.parse_and_validate_tool_calls(
            tool_calls=None,
            content="No tool calls here",
        )

        assert result.tool_calls is None
        assert result.filtered_count == 0

    def test_parse_and_validate_tool_calls_with_filtering(
        self, coordinator, mock_tool_adapter
    ):
        """Test tool call filtering by enabled tools."""
        mock_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "write_file", "arguments": {}},
            {"name": "secret_tool", "arguments": {}},
        ]

        result = coordinator.parse_and_validate_tool_calls(
            tool_calls=mock_calls,
            content="response",
            enabled_tools={"read_file", "write_file"},
        )

        assert len(result.tool_calls) == 2
        assert result.filtered_count == 1

    def test_normalize_tool_call_arguments(self, coordinator):
        """Test argument normalization."""
        # JSON string arguments
        tool_calls = [
            {"name": "test", "arguments": '{"key": "value"}'},
        ]

        result = coordinator.normalize_tool_call_arguments(tool_calls)

        assert result[0]["arguments"] == {"key": "value"}

    def test_normalize_tool_call_arguments_literal_eval(self, coordinator):
        """Test argument normalization with literal_eval."""
        # Python dict string
        tool_calls = [
            {"name": "test", "arguments": "{'key': 'value'}"},
        ]

        result = coordinator.normalize_tool_call_arguments(tool_calls)

        assert result[0]["arguments"] == {"key": "value"}

    def test_normalize_tool_call_arguments_none(self, coordinator):
        """Test argument normalization when None."""
        tool_calls = [
            {"name": "test", "arguments": None},
        ]

        result = coordinator.normalize_tool_call_arguments(tool_calls)

        assert result[0]["arguments"] == {}

    def test_process_stream_chunk_valid_content(self, coordinator):
        """Test processing valid stream chunk."""
        chunk = MockStreamChunk(content="valid content")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=0,
            max_garbage_chunks=3,
        )

        assert result.chunk == chunk
        assert result.consecutive_garbage_count == 0
        assert result.garbage_detected is False
        assert result.should_stop is False

    def test_process_stream_chunk_garbage_content(self, coordinator):
        """Test processing garbage stream chunk."""
        chunk = MockStreamChunk(content="garbage: detection test")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=0,
            max_garbage_chunks=3,
        )

        assert result.chunk == chunk
        assert result.consecutive_garbage_count == 1
        assert result.garbage_detected is False
        assert result.should_stop is False

    def test_process_stream_chunk_max_garbage_reached(self, coordinator):
        """Test stopping when max garbage reached."""
        chunk = MockStreamChunk(content="garbage: max test")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=2,  # Will become 3
            max_garbage_chunks=3,
            garbage_detected=False,
        )

        assert result.chunk is None
        assert result.consecutive_garbage_count == 3
        assert result.garbage_detected is True
        assert result.should_stop is True

    def test_process_stream_chunk_reset_on_valid(self, coordinator):
        """Test that valid content resets garbage counter."""
        chunk = MockStreamChunk(content="valid content")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=2,
            max_garbage_chunks=3,
        )

        assert result.consecutive_garbage_count == 0

    def test_aggregate_chunks(self, coordinator):
        """Test chunk aggregation."""
        chunks = ["chunk1", "chunk2", None, "chunk3"]

        result = coordinator.aggregate_chunks(chunks)

        assert result == "chunk1chunk2chunk3"

    def test_aggregate_chunks_empty(self, coordinator):
        """Test aggregating empty chunks."""
        result = coordinator.aggregate_chunks([])

        assert result == ""

    def test_aggregate_stream_chunks(self, coordinator):
        """Test aggregating stream chunk objects."""
        chunks = [
            MockStreamChunk("content1"),
            MockStreamChunk("content2"),
            MockStreamChunk(""),
        ]

        result = coordinator.aggregate_stream_chunks(chunks)

        assert result == "content1content2"

    def test_is_content_meaningful(self, coordinator):
        """Test content meaningfulness check."""
        # Meaningful content
        assert coordinator.is_content_meaningful("This is meaningful content") is True

        # Too short
        assert coordinator.is_content_meaningful("short") is False

        # Empty
        assert coordinator.is_content_meaningful("") is False

        # Custom threshold
        assert coordinator.is_content_meaningful("short", min_length=5) is True

    def test_format_response_for_display(self, coordinator):
        """Test response formatting."""
        content = "Response content"
        tool_calls = [
            {"name": "read_file"},
            {"name": "write_file"},
        ]

        # Without tool calls
        result = coordinator.format_response_for_display(content)
        assert result == content

        # With tool calls
        result = coordinator.format_response_for_display(
            content, tool_calls=tool_calls, show_tool_calls=True
        )
        assert "Response content" in result
        assert "Tool calls:" in result
        assert "read_file" in result
        assert "write_file" in result


class TestResponseCoordinatorIntegration:
    """Integration tests for ResponseCoordinator."""

    def test_full_response_processing_workflow(self):
        """Test complete workflow of processing a response."""
        # Create real-like mocks
        sanitizer = MockResponseSanitizer()
        adapter = MagicMock()

        # Setup tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.to_dict.return_value = {
            "name": "read_file",
            "arguments": {"path": "/test/file"}
        }

        adapter.parse_tool_calls.return_value = MagicMock(
            tool_calls=[mock_tool_call],
            remaining_content="remaining content",
            parse_method="mock"
        )

        coordinator = ResponseCoordinator(
            sanitizer=sanitizer,
            tool_adapter=adapter,
        )

        # Process response
        tool_calls = [{"name": "read_file", "arguments": "{}"}]
        validation = coordinator.parse_and_validate_tool_calls(
            tool_calls=tool_calls,
            content="Full response content",
            enabled_tools={"read_file", "write_file"},
        )

        assert validation.tool_calls is not None
        assert len(validation.tool_calls) == 1
        assert validation.tool_calls[0]["name"] == "read_file"
