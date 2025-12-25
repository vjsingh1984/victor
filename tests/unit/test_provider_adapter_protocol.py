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

"""Tests for provider adapter protocol and implementations."""

import pytest
from unittest.mock import MagicMock

from victor.protocols.provider_adapter import (
    ToolCallFormat,
    ProviderCapabilities,
    ToolCall,
    ContinuationContext,
    IProviderAdapter,
    BaseProviderAdapter,
    get_provider_adapter,
    register_provider_adapter,
)


# =============================================================================
# TOOL CALL FORMAT TESTS
# =============================================================================


class TestToolCallFormat:
    """Tests for ToolCallFormat enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert ToolCallFormat.OPENAI.value == "openai"
        assert ToolCallFormat.ANTHROPIC.value == "anthropic"
        assert ToolCallFormat.NATIVE.value == "native"
        assert ToolCallFormat.FALLBACK.value == "fallback"


# =============================================================================
# PROVIDER CAPABILITIES TESTS
# =============================================================================


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = ProviderCapabilities()
        assert caps.quality_threshold == 0.80
        assert caps.supports_thinking_tags is False
        assert caps.thinking_tag_format == ""
        assert caps.continuation_markers == []
        assert caps.max_continuation_attempts == 5
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.output_deduplication is False
        assert caps.streaming_chunk_size == 1024
        assert caps.supports_parallel_tools is True
        assert caps.grounding_required is True
        assert caps.grounding_strictness == 0.8
        assert caps.continuation_patience == 3
        assert caps.thinking_tokens_budget == 500
        assert caps.requires_thinking_time is False

    def test_custom_values(self):
        """Test custom capability values."""
        caps = ProviderCapabilities(
            quality_threshold=0.70,
            supports_thinking_tags=True,
            thinking_tag_format="<think>...</think>",
            continuation_markers=["...", "---"],
            max_continuation_attempts=10,
        )
        assert caps.quality_threshold == 0.70
        assert caps.supports_thinking_tags is True
        assert len(caps.continuation_markers) == 2


# =============================================================================
# TOOL CALL TESTS
# =============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation(self):
        """Test ToolCall creation."""
        call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "test.py"},
        )
        assert call.id == "call_123"
        assert call.name == "read_file"
        assert call.arguments == {"path": "test.py"}
        assert call.raw is None

    def test_with_raw(self):
        """Test ToolCall with raw data."""
        raw_data = {"original": "data"}
        call = ToolCall(
            id="call_456",
            name="write_file",
            arguments={"path": "out.py", "content": "test"},
            raw=raw_data,
        )
        assert call.raw == raw_data


# =============================================================================
# CONTINUATION CONTEXT TESTS
# =============================================================================


class TestContinuationContext:
    """Tests for ContinuationContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = ContinuationContext()
        assert ctx.attempt == 0
        assert ctx.reason == ""
        assert ctx.partial_response == ""
        assert ctx.tool_calls_made == 0

    def test_custom_values(self):
        """Test custom context values."""
        ctx = ContinuationContext(
            attempt=3,
            reason="incomplete response",
            partial_response="Started analyzing...",
            tool_calls_made=5,
        )
        assert ctx.attempt == 3
        assert ctx.reason == "incomplete response"
        assert ctx.partial_response == "Started analyzing..."
        assert ctx.tool_calls_made == 5


# =============================================================================
# BASE PROVIDER ADAPTER TESTS
# =============================================================================


class TestBaseProviderAdapter:
    """Tests for BaseProviderAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a base adapter for testing."""
        return BaseProviderAdapter()

    def test_name_property(self, adapter):
        """Test name property returns 'base'."""
        assert adapter.name == "base"

    def test_capabilities_property(self, adapter):
        """Test capabilities returns default ProviderCapabilities."""
        caps = adapter.capabilities
        assert isinstance(caps, ProviderCapabilities)
        assert caps.quality_threshold == 0.80

    def test_detect_continuation_empty_response(self, adapter):
        """Test continuation detection for empty response."""
        assert adapter.detect_continuation_needed("") is True
        assert adapter.detect_continuation_needed("   ") is True

    def test_detect_continuation_complete_response(self, adapter):
        """Test continuation detection for complete response."""
        assert adapter.detect_continuation_needed("This is complete.") is False

    def test_detect_continuation_with_markers(self):
        """Test continuation detection with custom markers."""

        class CustomAdapter(BaseProviderAdapter):
            @property
            def capabilities(self):
                return ProviderCapabilities(
                    continuation_markers=["...", "---"]
                )

        adapter = CustomAdapter()
        assert adapter.detect_continuation_needed("Some text...") is True
        assert adapter.detect_continuation_needed("Some text---") is True
        assert adapter.detect_continuation_needed("Some text.") is False

    def test_extract_thinking_no_tags(self, adapter):
        """Test thinking extraction when not supported."""
        thinking, content = adapter.extract_thinking_content("Hello world")
        assert thinking == ""
        assert content == "Hello world"

    def test_normalize_tool_calls_dict_format(self, adapter):
        """Test normalizing OpenAI dict format tool calls."""
        raw_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "test.py"},
                },
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].id == "call_1"
        assert normalized[0].name == "read_file"
        assert normalized[0].arguments == {"path": "test.py"}

    def test_normalize_tool_calls_object_format(self, adapter):
        """Test normalizing OpenAI object format tool calls."""

        class MockFunction:
            name = "write_file"
            arguments = {"path": "out.py"}

        class MockCall:
            id = "call_2"
            function = MockFunction()

        raw_calls = [MockCall()]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "write_file"

    def test_normalize_tool_calls_generates_id(self, adapter):
        """Test that missing IDs are generated."""
        raw_calls = [
            {"function": {"name": "test_tool", "arguments": {}}}
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert normalized[0].id == "call_0"

    def test_should_retry_rate_limit(self, adapter):
        """Test retry logic for rate limit errors."""
        error = Exception("Rate limit exceeded")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 60.0

    def test_should_retry_server_error(self, adapter):
        """Test retry logic for server errors."""
        error = Exception("503 Service Unavailable")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 5.0

    def test_should_retry_timeout(self, adapter):
        """Test retry logic for timeout errors."""
        error = Exception("Connection timeout")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 2.0

    def test_should_retry_unknown_error(self, adapter):
        """Test retry logic for unknown errors."""
        error = Exception("Unknown error")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is False
        assert backoff == 0.0


# =============================================================================
# PROVIDER ADAPTER REGISTRY TESTS
# =============================================================================


class TestProviderAdapterRegistry:
    """Tests for provider adapter registry functions."""

    def test_get_provider_adapter_openai(self):
        """Test getting OpenAI adapter."""
        adapter = get_provider_adapter("openai")
        assert adapter is not None
        assert adapter.name == "openai"

    def test_get_provider_adapter_anthropic(self):
        """Test getting Anthropic adapter."""
        adapter = get_provider_adapter("anthropic")
        assert adapter is not None
        assert adapter.name == "anthropic"

    def test_get_provider_adapter_deepseek(self):
        """Test getting DeepSeek adapter."""
        adapter = get_provider_adapter("deepseek")
        assert adapter is not None
        assert adapter.name == "deepseek"

    def test_get_provider_adapter_groq(self):
        """Test getting Groq adapter."""
        adapter = get_provider_adapter("groq")
        assert adapter is not None
        assert adapter.name in ["groq", "groqcloud"]

    def test_get_provider_adapter_ollama(self):
        """Test getting Ollama adapter."""
        adapter = get_provider_adapter("ollama")
        assert adapter is not None
        assert adapter.name == "ollama"

    def test_get_provider_adapter_unknown(self):
        """Test getting unknown provider returns base adapter."""
        adapter = get_provider_adapter("unknown_provider")
        assert adapter is not None
        # Should return base or default adapter

    def test_get_provider_adapter_case_insensitive(self):
        """Test provider name is case-insensitive."""
        adapter1 = get_provider_adapter("OpenAI")
        adapter2 = get_provider_adapter("openai")
        assert adapter1.name == adapter2.name

    def test_get_provider_adapter_aliases(self):
        """Test provider aliases work correctly."""
        # gemini should map to google
        adapter = get_provider_adapter("gemini")
        assert adapter.name in ["google", "gemini"]

    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""

        class CustomAdapter(BaseProviderAdapter):
            @property
            def name(self):
                return "custom_test"

            @property
            def capabilities(self):
                return ProviderCapabilities(quality_threshold=0.99)

        register_provider_adapter("custom_test", CustomAdapter)
        adapter = get_provider_adapter("custom_test")
        assert adapter.name == "custom_test"
        assert adapter.capabilities.quality_threshold == 0.99


# =============================================================================
# SPECIFIC PROVIDER ADAPTER TESTS
# =============================================================================


class TestDeepSeekAdapter:
    """Tests for DeepSeek adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get DeepSeek adapter."""
        return get_provider_adapter("deepseek")

    def test_supports_thinking_tags(self, adapter):
        """Test DeepSeek supports thinking tags."""
        assert adapter.capabilities.supports_thinking_tags is True

    def test_thinking_tag_format(self, adapter):
        """Test DeepSeek thinking tag format."""
        assert "<think>" in adapter.capabilities.thinking_tag_format.lower()

    def test_quality_threshold(self, adapter):
        """Test DeepSeek quality threshold is appropriate."""
        assert adapter.capabilities.quality_threshold <= 0.80


class TestAnthropicAdapter:
    """Tests for Anthropic adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get Anthropic adapter."""
        return get_provider_adapter("anthropic")

    def test_high_quality_threshold(self, adapter):
        """Test Anthropic has high quality threshold."""
        assert adapter.capabilities.quality_threshold >= 0.80

    def test_tool_call_format(self, adapter):
        """Test Anthropic uses Anthropic tool format."""
        assert adapter.capabilities.tool_call_format == ToolCallFormat.ANTHROPIC


class TestOllamaAdapter:
    """Tests for Ollama adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get Ollama adapter."""
        return get_provider_adapter("ollama")

    def test_fallback_tool_format(self, adapter):
        """Test Ollama uses fallback tool format."""
        assert adapter.capabilities.tool_call_format == ToolCallFormat.FALLBACK

    def test_lower_quality_threshold(self, adapter):
        """Test Ollama has appropriate quality threshold."""
        # Local models may have lower quality
        assert adapter.capabilities.quality_threshold <= 0.80


class TestXAIAdapter:
    """Tests for xAI/Grok adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get xAI adapter."""
        return get_provider_adapter("xai")

    def test_output_deduplication(self, adapter):
        """Test xAI enables output deduplication."""
        assert adapter.capabilities.output_deduplication is True

    def test_quality_threshold(self, adapter):
        """Test xAI quality threshold."""
        assert adapter.capabilities.quality_threshold >= 0.75


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Tests for IProviderAdapter protocol compliance."""

    def test_base_adapter_is_protocol_compliant(self):
        """Test BaseProviderAdapter implements IProviderAdapter."""
        adapter = BaseProviderAdapter()
        assert isinstance(adapter, IProviderAdapter)

    def test_custom_adapter_protocol_check(self):
        """Test custom adapter can implement protocol."""

        class MinimalAdapter:
            @property
            def name(self) -> str:
                return "minimal"

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities()

            def detect_continuation_needed(self, response: str) -> bool:
                return False

            def extract_thinking_content(self, response: str):
                return ("", response)

            def normalize_tool_calls(self, raw_calls):
                return []

            def should_retry(self, error):
                return (False, 0.0)

        adapter = MinimalAdapter()
        assert isinstance(adapter, IProviderAdapter)
