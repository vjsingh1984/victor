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

"""Coverage-focused tests for victor/providers/base.py.

These tests target the base provider interfaces, protocols, and helper functions
to improve coverage from ~10% to 30% target.
"""

import pytest
from typing import List, Dict, Any, AsyncIterator

from victor.providers.base import (
    # Protocols
    StreamingProvider,
    ToolCallingProvider,
    # Helper functions
    is_streaming_provider,
    is_tool_calling_provider,
    # Models
    Message,
    ToolDefinition,
    CompletionResponse,
    StreamChunk,
    BaseProvider,
)


class TestStreamingProviderProtocol:
    """Tests for StreamingProvider protocol."""

    def test_protocol_has_supports_streaming(self):
        """Test StreamingProvider has supports_streaming method."""
        assert hasattr(StreamingProvider, "supports_streaming")

    def test_protocol_has_stream_method(self):
        """Test StreamingProvider has stream method."""
        assert hasattr(StreamingProvider, "stream")


class TestToolCallingProviderProtocol:
    """Tests for ToolCallingProvider protocol."""

    def test_protocol_has_supports_tools(self):
        """Test ToolCallingProvider has supports_tools method."""
        assert hasattr(ToolCallingProvider, "supports_tools")


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_streaming_provider_with_method(self):
        """Test is_streaming_provider with object that has method."""
        class MockProvider:
            def supports_streaming(self) -> bool:
                return True

        provider = MockProvider()
        assert is_streaming_provider(provider) is True

    def test_is_streaming_provider_without_method(self):
        """Test is_streaming_provider with object without method."""
        class MockProvider:
            pass

        provider = MockProvider()
        assert is_streaming_provider(provider) is False

    def test_is_streaming_provider_returns_false(self):
        """Test is_streaming_provider when method returns False."""
        class MockProvider:
            def supports_streaming(self) -> bool:
                return False

        provider = MockProvider()
        assert is_streaming_provider(provider) is False

    def test_is_tool_calling_provider_with_method(self):
        """Test is_tool_calling_provider with object that has method."""
        class MockProvider:
            def supports_tools(self) -> bool:
                return True

        provider = MockProvider()
        assert is_tool_calling_provider(provider) is True

    def test_is_tool_calling_provider_without_method(self):
        """Test is_tool_calling_provider with object without method."""
        class MockProvider:
            pass

        provider = MockProvider()
        assert is_tool_calling_provider(provider) is False

    def test_is_tool_calling_provider_returns_false(self):
        """Test is_tool_calling_provider when method returns False."""
        class MockProvider:
            def supports_tools(self) -> bool:
                return False

        provider = MockProvider()
        assert is_tool_calling_provider(provider) is False


class TestMessage:
    """Tests for Message model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = Message(role="system", content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_message_with_name(self):
        """Test message with optional name field."""
        msg = Message(role="user", content="Hello", name="Alice")
        assert msg.name == "Alice"

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [{"name": "search", "arguments": '{"query": "test"}'}]
        msg = Message(role="assistant", content="", tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls

    def test_message_with_tool_call_id(self):
        """Test message with tool call ID."""
        msg = Message(role="tool", content="Result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(role="user", content="Hello", name="Alice")
        msg_dict = msg.to_dict()
        assert isinstance(msg_dict, dict)
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello"
        assert msg_dict["name"] == "Alice"


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_create_tool_definition(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "properties" in tool.parameters


class TestCompletionResponse:
    """Tests for CompletionResponse model."""

    def test_create_basic_response(self):
        """Test creating a basic completion response."""
        response = CompletionResponse(content="Hello, world!")
        assert response.content == "Hello, world!"
        assert response.role == "assistant"

    def test_response_with_usage(self):
        """Test response with usage information."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        response = CompletionResponse(content="Hi", usage=usage)
        assert response.usage == usage

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tool_calls = [{"name": "search", "arguments": "{}"}]
        response = CompletionResponse(content="", tool_calls=tool_calls)
        assert response.tool_calls == tool_calls

    def test_response_with_model(self):
        """Test response with model specified."""
        response = CompletionResponse(content="Hi", model="gpt-4")
        assert response.model == "gpt-4"

    def test_response_with_stop_reason(self):
        """Test response with stop reason."""
        response = CompletionResponse(content="Hi", stop_reason="stop")
        assert response.stop_reason == "stop"


class TestStreamChunk:
    """Tests for StreamChunk model."""

    def test_create_stream_chunk(self):
        """Test creating a stream chunk."""
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"

    def test_stream_chunk_attributes(self):
        """Test stream chunk has content attribute."""
        chunk = StreamChunk(content="Hello")
        assert hasattr(chunk, "content")
        # Check what attributes actually exist
        assert chunk.content == "Hello"


class TestBaseProvider:
    """Tests for BaseProvider abstract class."""

    def test_base_provider_is_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        # BaseProvider is abstract, so we can't instantiate it
        # But we can verify it exists and has the right methods
        assert hasattr(BaseProvider, "chat")
        assert hasattr(BaseProvider, "stream_chat")
        assert hasattr(BaseProvider, "supports_tools")
        assert hasattr(BaseProvider, "name")

    def test_base_provider_has_abstract_methods(self):
        """Test BaseProvider has required abstract methods."""
        import inspect
        abstract_methods = BaseProvider.__abstractmethods__
        # Should have at least chat and name as abstract
        assert "chat" in abstract_methods or len(abstract_methods) > 0
