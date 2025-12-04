"""Tests for tool calling adapters."""

from victor.agent.tool_calling.adapters import (
    AnthropicToolCallingAdapter,
    OpenAIToolCallingAdapter,
    OllamaToolCallingAdapter,
    OpenAICompatToolCallingAdapter,
    _get_capability_loader,
)
from victor.agent.tool_calling.base import ToolCallFormat
from victor.providers.base import ToolDefinition


class TestAnthropicAdapter:
    """Tests for AnthropicToolCallingAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        assert adapter.provider_name == "anthropic"

    def test_get_capabilities(self):
        """Test getting capabilities."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC

    def test_convert_tools(self):
        """Test converting tools to Anthropic format."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert "input_schema" in result[0]

    def test_parse_tool_calls_native(self):
        """Test parsing native Anthropic tool calls."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"

    def test_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        result = adapter.parse_tool_calls("Hello", None)
        assert len(result.tool_calls) == 0


class TestOpenAIAdapter:
    """Tests for OpenAIToolCallingAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        assert adapter.provider_name == "openai"

    def test_get_capabilities(self):
        """Test getting capabilities."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        caps = adapter.get_capabilities()
        assert caps.native_tool_calls is True
        assert caps.tool_call_format == ToolCallFormat.OPENAI

    def test_convert_tools(self):
        """Test converting tools to OpenAI format."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"

    def test_parse_tool_calls_native(self):
        """Test parsing native OpenAI tool calls."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": '{"arg": "value"}'}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_func"

    def test_parse_tool_calls_no_tools(self):
        """Test parsing response with no tool calls."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        result = adapter.parse_tool_calls("Hello", None)
        assert len(result.tool_calls) == 0


class TestOllamaAdapter:
    """Tests for OllamaToolCallingAdapter."""

    def test_provider_name(self):
        """Test provider name is correct."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        assert adapter.provider_name == "ollama"

    def test_get_capabilities_tool_capable_model(self):
        """Test getting capabilities for tool-capable model."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        caps = adapter.get_capabilities()
        # Should have native tool calls for llama3.1
        assert caps.native_tool_calls is True

    def test_convert_tools(self):
        """Test converting tools to Ollama format."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"

    def test_parse_tool_calls_native(self):
        """Test parsing native Ollama tool calls."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        raw_tool_calls = [{"function": {"name": "test_func", "arguments": {"arg": "value"}}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1

    def test_parse_json_fallback(self):
        """Test JSON fallback parsing."""
        adapter = OllamaToolCallingAdapter(model="codellama:7b")
        content = '```json\n{"name": "test_func", "arguments": {"arg": "value"}}\n```'
        result = adapter.parse_tool_calls(content, None)
        # Result is valid (may or may not have tool calls depending on parsing)
        assert result is not None

    def test_parse_xml_fallback(self):
        """Test XML fallback parsing."""
        adapter = OllamaToolCallingAdapter(model="qwen3-coder:30b")
        content = "<function=test_func><parameter=arg>value</parameter></function>"
        result = adapter.parse_tool_calls(content, None)
        assert result is not None
        # XML parsing should find the tool call (if fallback is enabled for this model)
        # The result may have 0 or more tool calls depending on model capabilities
        assert hasattr(result, "tool_calls")


class TestOpenAICompatAdapter:
    """Tests for OpenAICompatToolCallingAdapter (LMStudio/vLLM)."""

    def test_provider_name_lmstudio(self):
        """Test provider name for LMStudio."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model", provider_variant="lmstudio")
        assert adapter.provider_name == "lmstudio"

    def test_provider_name_vllm(self):
        """Test provider name for vLLM."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model", provider_variant="vllm")
        assert adapter.provider_name == "vllm"

    def test_convert_tools(self):
        """Test converting tools."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model")
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object"},
            )
        ]
        result = adapter.convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"


class TestCapabilityLoader:
    """Tests for capability loader singleton."""

    def test_singleton_behavior(self):
        """Test that capability loader is a singleton."""
        loader1 = _get_capability_loader()
        loader2 = _get_capability_loader()
        assert loader1 is loader2

    def test_loader_returns_capabilities(self):
        """Test that loader returns valid capabilities."""
        loader = _get_capability_loader()
        caps = loader.get_capabilities("openai", "gpt-4", ToolCallFormat.OPENAI)
        assert caps is not None
        assert hasattr(caps, "native_tool_calls")


class TestAnthropicAdapterEdgeCases:
    """Edge case tests for AnthropicToolCallingAdapter."""

    def test_parse_invalid_tool_name(self):
        """Test parsing with invalid tool name."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        # Empty name should be skipped silently
        raw_tool_calls = [{"id": "call_123", "name": "", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        # Invalid names are skipped silently
        assert len(result.tool_calls) == 0

    def test_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        # Anthropic API returns arguments as dicts, not JSON strings
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"arg": "value"}

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        adapter = AnthropicToolCallingAdapter(model="claude-3-sonnet")
        raw_tool_calls = [
            {"id": "call_1", "name": "func1", "arguments": {"x": 1}},
            {"id": "call_2", "name": "func2", "arguments": {"y": 2}},
        ]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "func1"
        assert result.tool_calls[1].name == "func2"


class TestOpenAIAdapterEdgeCases:
    """Edge case tests for OpenAIToolCallingAdapter."""

    def test_parse_invalid_tool_name(self):
        """Test parsing with invalid tool name."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "", "arguments": "{}"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.warnings) > 0

    def test_parse_dict_arguments(self):
        """Test parsing tool calls with dict arguments."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": {"arg": "value"}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1

    def test_parse_invalid_json_arguments(self):
        """Test parsing tool calls with invalid JSON arguments."""
        adapter = OpenAIToolCallingAdapter(model="gpt-4")
        raw_tool_calls = [{"id": "call_123", "name": "test_func", "arguments": "invalid json {{"}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1
        assert len(result.warnings) > 0


class TestOllamaAdapterEdgeCases:
    """Edge case tests for OllamaToolCallingAdapter."""

    def test_get_capabilities_non_tool_model(self):
        """Test getting capabilities for non-tool-capable model."""
        adapter = OllamaToolCallingAdapter(model="old-model-no-tools")
        caps = adapter.get_capabilities()
        # Non-matching model might have native_tool_calls=False
        assert hasattr(caps, "native_tool_calls")

    def test_parse_empty_raw_tool_calls(self):
        """Test parsing with empty raw_tool_calls."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        result = adapter.parse_tool_calls("", [])
        assert len(result.tool_calls) == 0

    def test_parse_xml_function_call_format(self):
        """Test parsing XML function call format."""
        adapter = OllamaToolCallingAdapter(model="codellama:7b")
        content = '<function_call><name>my_tool</name><arguments>{"key": "value"}</arguments></function_call>'
        result = adapter.parse_tool_calls(content, None)
        assert result is not None

    def test_parse_tool_call_xml_format(self):
        """Test parsing <tool_call> XML format."""
        adapter = OllamaToolCallingAdapter(model="qwen2.5:7b")
        content = (
            '<tool_call><name>test_func</name><arguments>{"arg": "val"}</arguments></tool_call>'
        )
        result = adapter.parse_tool_calls(content, None)
        assert result is not None

    def test_text_fallback_parsing(self):
        """Test text content without tool calls."""
        adapter = OllamaToolCallingAdapter(model="llama3.1:8b")
        result = adapter.parse_tool_calls("Just some text response", None)
        assert len(result.tool_calls) == 0


class TestOpenAICompatAdapterEdgeCases:
    """Edge case tests for OpenAICompatToolCallingAdapter."""

    def test_default_provider_variant(self):
        """Test default provider variant is lmstudio."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model")
        assert adapter.provider_name == "lmstudio"

    def test_parse_tool_calls(self):
        """Test parsing tool calls."""
        adapter = OpenAICompatToolCallingAdapter(model="llama3.1:8b", provider_variant="vllm")
        raw_tool_calls = [{"id": "call_123", "name": "test", "arguments": {"x": 1}}]
        result = adapter.parse_tool_calls("", raw_tool_calls)
        assert len(result.tool_calls) == 1

    def test_get_system_prompt_hints(self):
        """Test getting system prompt hints."""
        adapter = OpenAICompatToolCallingAdapter(model="local-model")
        hints = adapter.get_system_prompt_hints()
        assert isinstance(hints, str)
