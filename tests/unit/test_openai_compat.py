"""Tests for OpenAI-compatible utilities."""

from victor.providers.openai_compat import (
    convert_tools_to_openai_format,
    convert_tools_to_anthropic_format,
    convert_messages_to_openai_format,
    parse_openai_tool_calls,
    parse_openai_stream_chunk,
)
from victor.providers.base import Message, ToolDefinition


class TestConvertToolsToOpenAIFormat:
    """Tests for convert_tools_to_openai_format."""

    def test_basic_tool_conversion(self):
        """Test converting a basic tool."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = convert_tools_to_openai_format(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"
        assert result[0]["function"]["description"] == "A test tool"

    def test_multiple_tools_conversion(self):
        """Test converting multiple tools."""
        tools = [
            ToolDefinition(name="tool1", description="Tool 1", parameters={}),
            ToolDefinition(name="tool2", description="Tool 2", parameters={}),
        ]
        result = convert_tools_to_openai_format(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    def test_empty_tools_list(self):
        """Test converting empty tools list."""
        result = convert_tools_to_openai_format([])
        assert result == []


class TestConvertToolsToAnthropicFormat:
    """Tests for convert_tools_to_anthropic_format."""

    def test_basic_tool_conversion(self):
        """Test converting a basic tool to Anthropic format."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = convert_tools_to_anthropic_format(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert result[0]["description"] == "A test tool"
        assert "input_schema" in result[0]


class TestConvertMessagesToOpenAIFormat:
    """Tests for convert_messages_to_openai_format."""

    def test_basic_message(self):
        """Test converting a basic message."""
        messages = [Message(role="user", content="Hello")]
        result = convert_messages_to_openai_format(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_message_with_tool_calls(self):
        """Test converting message with tool calls."""
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_123", "name": "test_func", "arguments": {"arg1": "val1"}}],
            )
        ]
        result = convert_messages_to_openai_format(messages)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["id"] == "call_123"
        assert result[0]["tool_calls"][0]["function"]["name"] == "test_func"

    def test_tool_response_message(self):
        """Test converting tool response message."""
        msg = Message(role="tool", content="Tool result")
        msg.tool_call_id = "call_123"
        msg.name = "test_func"
        messages = [msg]
        result = convert_messages_to_openai_format(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["name"] == "test_func"

    def test_multiple_messages(self):
        """Test converting multiple messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = convert_messages_to_openai_format(messages)
        assert len(result) == 2

    def test_empty_content(self):
        """Test message with empty content."""
        messages = [Message(role="assistant", content="")]
        result = convert_messages_to_openai_format(messages)
        assert result[0]["content"] == ""


class TestParseOpenAIToolCalls:
    """Tests for parse_openai_tool_calls."""

    def test_parse_valid_tool_calls(self):
        """Test parsing valid tool calls."""
        data = [
            {
                "id": "call_123",
                "function": {"name": "test_func", "arguments": '{"arg": "value"}'},
            }
        ]
        result = parse_openai_tool_calls(data)
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["name"] == "test_func"
        assert result[0]["arguments"] == {"arg": "value"}

    def test_parse_invalid_json_arguments(self):
        """Test parsing tool calls with invalid JSON arguments."""
        data = [{"id": "call_123", "function": {"name": "test", "arguments": "invalid json"}}]
        result = parse_openai_tool_calls(data)
        assert result is not None
        assert result[0]["arguments"] == {"raw": "invalid json"}

    def test_parse_none_input(self):
        """Test parsing None input."""
        result = parse_openai_tool_calls(None)
        assert result is None

    def test_parse_empty_list(self):
        """Test parsing empty list."""
        result = parse_openai_tool_calls([])
        assert result is None

    def test_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments (not string)."""
        data = [
            {
                "id": "call_123",
                "function": {"name": "test", "arguments": {"already": "dict"}},
            }
        ]
        result = parse_openai_tool_calls(data)
        assert result[0]["arguments"] == {"already": "dict"}


class TestParseOpenAIStreamChunk:
    """Tests for parse_openai_stream_chunk."""

    def test_parse_content_chunk(self):
        """Test parsing chunk with content."""
        chunk = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        result = parse_openai_stream_chunk(chunk)
        assert result["content"] == "Hello"
        assert result["finish_reason"] is None

    def test_parse_tool_calls_chunk(self):
        """Test parsing chunk with tool calls."""
        chunk = {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"name": "test"}}]},
                    "finish_reason": None,
                }
            ]
        }
        result = parse_openai_stream_chunk(chunk)
        assert result["tool_calls"] is not None

    def test_parse_finish_chunk(self):
        """Test parsing chunk with finish reason."""
        chunk = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        result = parse_openai_stream_chunk(chunk)
        assert result["finish_reason"] == "stop"

    def test_parse_empty_choices(self):
        """Test parsing chunk with no choices."""
        chunk = {"choices": []}
        result = parse_openai_stream_chunk(chunk)
        assert result["content"] is None
        assert result["tool_calls"] is None
        assert result["finish_reason"] is None

    def test_parse_missing_choices(self):
        """Test parsing chunk without choices key."""
        chunk = {}
        result = parse_openai_stream_chunk(chunk)
        assert result["content"] is None
