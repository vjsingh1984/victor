"""Tests for OpenAI-compatible utilities."""

from victor.providers.openai_compat import (
    convert_tools_to_openai_format,
    convert_tools_to_anthropic_format,
    convert_messages_to_openai_format,
    parse_openai_tool_calls,
    parse_openai_stream_chunk,
    build_openai_messages,
    fix_orphaned_tool_messages,
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
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "test_func",
                        "arguments": {"arg1": "val1"},
                    }
                ],
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
        data = [
            {
                "id": "call_123",
                "function": {"name": "test", "arguments": "invalid json"},
            }
        ]
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


class TestBuildOpenAIMessages:
    """Tests for build_openai_messages."""

    def test_tool_message_with_name_and_tool_call_id(self):
        """Test that tool messages preserve both tool_call_id and name fields when paired."""
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [
            {"id": "call_123", "name": "test_tool", "arguments": {"arg": "value"}}
        ]
        messages.append(assistant_msg)
        # Create a tool message with both tool_call_id and name
        tool_msg = Message(role="tool", content="Result")
        tool_msg.tool_call_id = "call_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Find the tool message in result
        tool_result = [m for m in result if m["role"] == "tool"][0]
        assert tool_result["tool_call_id"] == "call_123"
        assert tool_result["name"] == "test_tool"

    def test_orphaned_tool_message_is_removed(self):
        """Test that orphaned tool messages (no matching tool_call) are removed."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        # Create an orphaned tool message (no assistant message with tool_calls)
        tool_msg = Message(role="tool", content="Orphaned result")
        tool_msg.tool_call_id = "orphaned_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # The orphaned tool message should be removed by fix_orphaned_tool_messages
        assert not any(m["role"] == "tool" for m in result)

    def test_paired_tool_call_and_response(self):
        """Test that properly paired tool_calls and tool responses are preserved."""
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [
            {"id": "call_abc", "name": "test_tool", "arguments": {"arg": "value"}}
        ]
        messages.append(assistant_msg)

        # Create tool response
        tool_msg = Message(role="tool", content="Success")
        tool_msg.tool_call_id = "call_abc"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Both should be present
        assert len(result) == 3
        assistant_result = [m for m in result if m["role"] == "assistant"][0]
        tool_result = [m for m in result if m["role"] == "tool"][0]

        assert "tool_calls" in assistant_result
        assert assistant_result["tool_calls"][0]["id"] == "call_abc"
        assert tool_result["tool_call_id"] == "call_abc"
        assert tool_result["name"] == "test_tool"

    def test_no_fallback_tool_call_id_generated(self):
        """Test that no fallback tool_call_id is generated for missing IDs."""
        messages = [
            Message(role="user", content="Hello"),
        ]
        # Create a tool message without tool_call_id
        tool_msg = Message(role="tool", content="Result")
        # tool_call_id is None
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Tool message should be removed (orphaned) since no tool_call_id
        assert not any(m["role"] == "tool" for m in result)

    def test_empty_tool_content_gets_placeholder(self):
        """Test that tool messages with empty content get a placeholder value.

        DeepSeek and some other providers reject tool messages with empty content.
        This test verifies that empty content is replaced with a placeholder.
        """
        messages = [
            Message(role="user", content="Use the tool"),
        ]
        # Create assistant message with tool_calls
        assistant_msg = Message(role="assistant", content="")
        assistant_msg.tool_calls = [{"id": "call_123", "name": "test_tool", "arguments": {}}]
        messages.append(assistant_msg)

        # Create tool message with empty content
        tool_msg = Message(role="tool", content="")
        tool_msg.tool_call_id = "call_123"
        tool_msg.name = "test_tool"
        messages.append(tool_msg)

        result = build_openai_messages(messages)

        # Find the tool message in result
        tool_result = [m for m in result if m["role"] == "tool"][0]
        # Empty content should be replaced with placeholder
        assert tool_result["content"] == "(no output)"
        assert tool_result["tool_call_id"] == "call_123"


class TestFixOrphanedToolMessages:
    """Tests for fix_orphaned_tool_messages."""

    def test_removes_orphaned_tool_response(self):
        """Test that tool responses without matching tool_calls are removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "orphan_123", "content": "Result"},
        ]

        result = fix_orphaned_tool_messages(messages)

        # Orphaned tool message should be removed
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_removes_tool_calls_without_responses(self):
        """Test that tool_calls without matching responses are stripped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            },
        ]

        result = fix_orphaned_tool_messages(messages)

        # tool_calls should be stripped from assistant message
        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" not in result[1]
        assert result[1]["content"] == ""

    def test_preserves_valid_tool_call_pairs(self):
        """Test that valid tool_call/response pairs are preserved."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Result"},
        ]

        result = fix_orphaned_tool_messages(messages)

        # Both should be preserved
        assert len(result) == 3
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_123"
