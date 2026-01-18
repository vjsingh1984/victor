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


@pytest.fixture
def mock_sanitizer():
    """Create mock sanitizer."""
    return MockResponseSanitizer()


@pytest.fixture
def mock_tool_adapter():
    """Create mock tool adapter."""
    adapter = MagicMock()
    adapter.parse_tool_calls.return_value = MagicMock(
        tool_calls=[],
        remaining_content="content",
        parse_method="mock"
    )
    return adapter


@pytest.fixture
def mock_tool_registry():
    """Create mock tool registry."""
    registry = MagicMock()
    registry.list_tools.return_value = ["read_file", "write_file"]
    return registry


@pytest.fixture
def coordinator(mock_sanitizer, mock_tool_adapter, mock_tool_registry):
    """Create coordinator with mocks."""
    return ResponseCoordinator(
        sanitizer=mock_sanitizer,
        tool_adapter=mock_tool_adapter,
        tool_registry=mock_tool_registry,
    )


class TestResponseCoordinator:
    """Tests for ResponseCoordinator."""

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


class TestResponseCoordinatorExtendedMethods:
    """Tests for extended methods in ResponseCoordinator."""

    @pytest.fixture
    def coordinator(self, mock_sanitizer):
        """Create coordinator with mocks."""
        return ResponseCoordinator(
            sanitizer=mock_sanitizer,
        )

    def test_aggregate_stream_chunks_default_attr(self, coordinator):
        """Test aggregating stream chunks with default attribute."""
        chunks = [
            MockStreamChunk("content1"),
            MockStreamChunk("content2"),
            None,  # Should be skipped
        ]

        result = coordinator.aggregate_stream_chunks(chunks)
        assert result == "content1content2"

    def test_aggregate_stream_chunks_custom_attr(self, coordinator):
        """Test aggregating stream chunks with custom attribute."""
        class CustomChunk:
            def __init__(self, text):
                self.text = text

        chunks = [CustomChunk("text1"), CustomChunk("text2")]

        result = coordinator.aggregate_stream_chunks(chunks, content_attr="text")
        assert result == "text1text2"

    def test_aggregate_stream_chunks_empty(self, coordinator):
        """Test aggregating empty stream chunks."""
        result = coordinator.aggregate_stream_chunks([])
        assert result == ""

    def test_aggregate_stream_chunks_no_content_attr(self, coordinator):
        """Test aggregating chunks without content attribute."""
        class BadChunk:
            pass

        chunks = [BadChunk(), BadChunk()]
        result = coordinator.aggregate_stream_chunks(chunks)
        assert result == ""

    def test_extract_remaining_content_no_tool_calls(self, coordinator):
        """Test extracting remaining content when no tool calls."""
        content = "This is just regular content"
        result = coordinator.extract_remaining_content(content, tool_calls=None)
        assert result == "This is just regular content"

    def test_extract_remaining_content_with_tool_calls(self, coordinator):
        """Test extracting remaining content with tool call removal."""
        content = 'Some text {"name": "read_file", "arguments": {}} more text'
        tool_calls = [{"name": "read_file", "arguments": {}}]

        result = coordinator.extract_remaining_content(content, tool_calls)
        # Tool call JSON should be removed
        assert '{"name": "read_file"' not in result
        assert "Some text" in result
        assert "more text" in result

    def test_extract_remaining_content_empty(self, coordinator):
        """Test extracting from empty content."""
        result = coordinator.extract_remaining_content("", tool_calls=None)
        assert result == ""

    def test_normalize_tool_call_arguments_invalid_string(self, coordinator):
        """Test argument normalization with invalid string."""
        tool_calls = [
            {"name": "test", "arguments": "not valid json or dict"},
        ]

        result = coordinator.normalize_tool_call_arguments(tool_calls)
        # Should wrap invalid string in dict
        assert result[0]["arguments"] == {"value": "not valid json or dict"}

    def test_parse_and_validate_tool_calls_no_adapter(self):
        """Test parsing when no adapter is configured."""
        sanitizer = MockResponseSanitizer()
        coordinator = ResponseCoordinator(
            sanitizer=sanitizer,
            tool_adapter=None,  # No adapter
        )

        # Should not crash, just return empty tool calls
        result = coordinator.parse_and_validate_tool_calls(
            tool_calls=None,
            content="Some content",
        )

        assert result.tool_calls is None
        assert result.remaining_content == "Some content"

    def test_parse_and_validate_tool_calls_non_dict_calls(self, coordinator):
        """Test parsing with non-dict tool calls."""
        # Mix of dict and non-dict
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            "not a dict",
            {"name": "write_file", "arguments": {}},
            None,
        ]

        result = coordinator.parse_and_validate_tool_calls(
            tool_calls=tool_calls,
            content="content",
        )

        # Should filter out non-dict calls
        assert len(result.tool_calls) == 2
        assert result.filtered_count >= 2

    def test_parse_and_validate_tool_calls_empty_enabled_tools(self, coordinator):
        """Test filtering with empty enabled tools set."""
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "write_file", "arguments": {}},
        ]

        result = coordinator.parse_and_validate_tool_calls(
            tool_calls=tool_calls,
            content="content",
            enabled_tools=set(),  # Empty set
        )

        # All tools should be filtered out
        assert result.tool_calls is None
        assert result.filtered_count == 2

    def test_process_stream_chunk_no_content_attr(self, coordinator):
        """Test processing chunk without content attribute."""
        class BadChunk:
            pass

        chunk = BadChunk()
        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=0,
            max_garbage_chunks=3,
        )

        # Should return chunk as-is without garbage detection
        assert result.chunk == chunk
        assert result.consecutive_garbage_count == 0
        assert result.should_stop is False

    def test_process_stream_chunk_empty_content(self, coordinator):
        """Test processing chunk with empty content."""
        chunk = MockStreamChunk(content="")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=0,
            max_garbage_chunks=3,
        )

        # Empty content should not trigger garbage detection
        assert result.chunk == chunk
        assert result.consecutive_garbage_count == 0

    def test_try_extract_tool_calls_disabled(self, coordinator):
        """Test tool call extraction when disabled."""
        config = ResponseCoordinatorConfig(enable_tool_call_extraction=False)
        coordinator = ResponseCoordinator(
            sanitizer=coordinator._sanitizer,
            config=config,
        )

        result = coordinator.try_extract_tool_calls_from_text(
            "Some content with tool calls",
            valid_tool_names={"read_file"}
        )

        # Should return None when disabled
        assert result is None

    def test_is_garbage_content_empty(self, coordinator):
        """Test garbage detection with empty content."""
        assert coordinator.is_garbage_content("") is False

    def test_sanitize_response_empty(self, coordinator):
        """Test sanitizing empty content."""
        result = coordinator.sanitize_response("")
        assert result == ""

    def test_format_response_for_display_no_tool_calls(self, coordinator):
        """Test formatting without tool calls but flag set."""
        content = "Just content"

        result = coordinator.format_response_for_display(
            content,
            tool_calls=None,
            show_tool_calls=True
        )

        # Should just return content
        assert result == "Just content"

    def test_format_response_for_display_tool_calls_no_name(self, coordinator):
        """Test formatting tool calls with missing name."""
        content = "Content"
        tool_calls = [{"arguments": {}}, {"name": "read_file"}]

        result = coordinator.format_response_for_display(
            content,
            tool_calls=tool_calls,
            show_tool_calls=True
        )

        assert "Content" in result
        assert "Tool calls:" in result
        assert "unknown" in result or "read_file" in result


class TestResponseCoordinatorWithResolution:
    """Tests for parse_and_validate_tool_calls_with_resolution method."""

    @pytest.fixture
    def mock_shell_resolver(self):
        """Create mock shell variant resolver."""
        resolver = MagicMock()
        resolver.resolve_shell_variant.return_value = "shell_readonly"
        return resolver

    @pytest.fixture
    def mock_tool_enabled_checker(self):
        """Create mock tool enabled checker."""
        checker = MagicMock()
        checker.is_tool_enabled.return_value = True
        return checker

    @pytest.fixture
    def coordinator_with_resolvers(
        self, mock_sanitizer, mock_shell_resolver, mock_tool_enabled_checker
    ):
        """Create coordinator with resolver dependencies."""
        return ResponseCoordinator(
            sanitizer=mock_sanitizer,
            shell_variant_resolver=mock_shell_resolver,
            tool_enabled_checker=mock_tool_enabled_checker,
        )

    def test_resolve_shell_variant_internal(self, coordinator_with_resolvers):
        """Test internal shell variant resolution."""
        # Should call resolver for shell aliases
        result = coordinator_with_resolvers._resolve_shell_variant_internal("run")
        assert result == "shell_readonly"
        coordinator_with_resolvers._shell_variant_resolver.resolve_shell_variant.assert_called_once_with("run")

    def test_resolve_shell_variant_internal_not_shell_alias(self, coordinator_with_resolvers):
        """Test that non-shell tools are not resolved."""
        result = coordinator_with_resolvers._resolve_shell_variant_internal("read_file")
        assert result == "read_file"
        coordinator_with_resolvers._shell_variant_resolver.resolve_shell_variant.assert_not_called()

    def test_resolve_shell_variant_internal_no_resolver(self, mock_sanitizer):
        """Test shell resolution without resolver."""
        coordinator = ResponseCoordinator(sanitizer=mock_sanitizer)

        result = coordinator._resolve_shell_variant_internal("run")
        # Should return original name
        assert result == "run"

    def test_is_tool_enabled_internal_with_set(self, coordinator_with_resolvers):
        """Test tool enabled check with provided set."""
        enabled_tools = {"read_file", "write_file"}

        result = coordinator_with_resolvers._is_tool_enabled_internal(
            "read_file",
            enabled_tools=enabled_tools
        )

        assert result is True
        coordinator_with_resolvers._tool_enabled_checker.is_tool_enabled.assert_not_called()

    def test_is_tool_enabled_internal_with_checker(self, coordinator_with_resolvers):
        """Test tool enabled check with checker."""
        result = coordinator_with_resolvers._is_tool_enabled_internal("read_file")

        assert result is True
        coordinator_with_resolvers._tool_enabled_checker.is_tool_enabled.assert_called_once_with("read_file")

    def test_is_tool_enabled_internal_with_registry(self, mock_sanitizer, mock_tool_registry):
        """Test tool enabled check with registry."""
        coordinator = ResponseCoordinator(
            sanitizer=mock_sanitizer,
            tool_registry=mock_tool_registry,
        )

        result = coordinator._is_tool_enabled_internal("read_file")

        # Registry should be consulted
        assert result is True
        mock_tool_registry.get_tool.assert_called_once_with("read_file")

    def test_is_tool_enabled_internal_registry_error(self, mock_sanitizer):
        """Test tool enabled check with registry error."""
        registry = MagicMock()
        registry.get_tool.side_effect = Exception("Tool not found")

        coordinator = ResponseCoordinator(
            sanitizer=mock_sanitizer,
            tool_registry=registry,
        )

        # Should default to True on error
        result = coordinator._is_tool_enabled_internal("unknown_tool")
        assert result is True

    def test_is_tool_enabled_internal_no_fallbacks(self, mock_sanitizer):
        """Test tool enabled check with no fallbacks."""
        coordinator = ResponseCoordinator(sanitizer=mock_sanitizer)

        # Should default to True
        result = coordinator._is_tool_enabled_internal("any_tool")
        assert result is True

    def test_parse_and_validate_with_resolution_shell_alias(self, coordinator_with_resolvers):
        """Test parsing with shell alias resolution."""
        tool_calls = [
            {"name": "run", "arguments": {"command": "ls"}}
        ]

        result_tool_calls, result_content = coordinator_with_resolvers.parse_and_validate_tool_calls_with_resolution(
            tool_calls=tool_calls,
            full_content="Content",
            enabled_tools={"shell_readonly"}
        )

        # Should resolve "run" to "shell_readonly"
        assert result_tool_calls is not None
        assert result_tool_calls[0]["name"] == "shell_readonly"

    def test_parse_and_validate_with_resolution_filters_disabled(self, coordinator_with_resolvers):
        """Test that disabled tools are filtered."""
        tool_calls = [
            {"name": "read_file", "arguments": {}},
            {"name": "disabled_tool", "arguments": {}},
        ]

        coordinator_with_resolvers._tool_enabled_checker.is_tool_enabled.side_effect = lambda name: name == "read_file"

        result_tool_calls, result_content = coordinator_with_resolvers.parse_and_validate_tool_calls_with_resolution(
            tool_calls=tool_calls,
            full_content="Content",
        )

        # Should filter out disabled tool
        assert len(result_tool_calls) == 1
        assert result_tool_calls[0]["name"] == "read_file"

    def test_parse_and_validate_with_resolution_no_tool_calls(self, coordinator_with_resolvers):
        """Test parsing when no tool calls provided."""
        result_tool_calls, result_content = coordinator_with_resolvers.parse_and_validate_tool_calls_with_resolution(
            tool_calls=None,
            full_content="Just content",
        )

        assert result_tool_calls is None
        assert result_content == "Just content"

    def test_parse_and_validate_with_resolution_normalizes_arguments(self, coordinator_with_resolvers):
        """Test that arguments are normalized."""
        tool_calls = [
            {"name": "read_file", "arguments": '{"path": "/test"}'}
        ]

        result_tool_calls, result_content = coordinator_with_resolvers.parse_and_validate_tool_calls_with_resolution(
            tool_calls=tool_calls,
            full_content="Content",
        )

        # Arguments should be converted from JSON string to dict
        assert isinstance(result_tool_calls[0]["arguments"], dict)
        assert result_tool_calls[0]["arguments"]["path"] == "/test"


class TestResponseCoordinatorEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def coordinator(self, mock_sanitizer):
        """Create coordinator with mocks."""
        return ResponseCoordinator(
            sanitizer=mock_sanitizer,
        )

    def test_is_content_meaningful_whitespace_only(self, coordinator):
        """Test content meaningfulness with whitespace."""
        assert coordinator.is_content_meaningful("   \n\t  ") is False

    def test_is_content_meaningful_custom_threshold(self, coordinator):
        """Test with custom length threshold."""
        short_content = "short"
        assert coordinator.is_content_meaningful(short_content, min_length=10) is False
        assert coordinator.is_content_meaningful(short_content, min_length=5) is True

    def test_aggregate_chunks_all_none(self, coordinator):
        """Test aggregating all None chunks."""
        result = coordinator.aggregate_chunks([None, None, None])
        assert result == ""

    def test_aggregate_chunks_with_empty_strings(self, coordinator):
        """Test aggregating with empty strings."""
        result = coordinator.aggregate_chunks(["a", "", "b", "", "c"])
        assert result == "abc"

    def test_parse_and_validate_tool_calls_all_filtered(self, coordinator, mock_tool_adapter):
        """Test when all tool calls are filtered out."""
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]

        result = coordinator.parse_and_validate_tool_calls(
            tool_calls=tool_calls,
            content="content",
            enabled_tools=set(),  # Filter all
        )

        assert result.tool_calls is None
        assert result.filtered_count == 2

    def test_process_stream_chunk_already_garbage_detected(self, coordinator):
        """Test processing when garbage was already detected."""
        chunk = MockStreamChunk(content="garbage: more")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=2,
            max_garbage_chunks=3,
            garbage_detected=True,  # Already detected
        )

        # Should still stop
        assert result.should_stop is True
        assert result.garbage_detected is True
        assert result.chunk is None

    def test_process_stream_chunk_exactly_max_garbage(self, coordinator):
        """Test when consecutive count exactly equals max."""
        chunk = MockStreamChunk(content="garbage: test")

        result = coordinator.process_stream_chunk(
            chunk=chunk,
            consecutive_garbage_count=3,  # Exactly at max
            max_garbage_chunks=3,
            garbage_detected=False,
        )

        # Should trigger stop
        assert result.should_stop is True
        assert result.garbage_detected is True
        assert result.chunk is None
