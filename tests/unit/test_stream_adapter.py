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

"""Tests for UnifiedStreamAdapter and SDK-specific adapters."""

import pytest
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional
from unittest.mock import MagicMock

from victor.providers.stream_adapter import (
    StreamMetrics,
    SDKType,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    OllamaAdapter,
    LMStudioAdapter,
    VLLMAdapter,
    SDKAdapterRegistry,
    UnifiedStreamAdapter,
    wrap_provider,
    detect_provider_sdk,
)
from victor.providers.base import StreamChunk, Message, BaseProvider


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockOpenAIChoice:
    """Mock OpenAI choice object."""

    delta: Any
    finish_reason: Optional[str] = None


@dataclass
class MockOpenAIDelta:
    """Mock OpenAI delta object."""

    content: Optional[str] = None
    tool_calls: Optional[list] = None


@dataclass
class MockOpenAIChunk:
    """Mock OpenAI stream chunk."""

    choices: List[MockOpenAIChoice]


@dataclass
class MockAnthropicEvent:
    """Mock Anthropic stream event."""

    type: str
    delta: Optional[Any] = None
    name: Optional[str] = None
    input: Optional[dict] = None


@dataclass
class MockAnthropicDelta:
    """Mock Anthropic delta."""

    text: str = ""


@dataclass
class MockGeminiCandidate:
    """Mock Gemini candidate."""

    finish_reason: Optional[str] = None
    content: Optional[Any] = None


@dataclass
class MockGeminiChunk:
    """Mock Gemini stream chunk."""

    text: str
    candidates: Optional[List[MockGeminiCandidate]] = None


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name
        self._stream_chunks: List[StreamChunk] = []

    @property
    def name(self) -> str:
        return self._name

    def set_stream_chunks(self, chunks: List[StreamChunk]) -> None:
        """Set chunks to return from stream."""
        self._stream_chunks = chunks

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[list] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Mock stream implementation."""
        for chunk in self._stream_chunks:
            yield chunk

    async def chat(self, messages, **kwargs):
        """Mock chat implementation."""
        pass

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    async def close(self) -> None:
        """Mock close implementation."""
        pass


# =============================================================================
# STREAM METRICS TESTS
# =============================================================================


class TestStreamMetrics:
    """Tests for StreamMetrics dataclass."""

    def test_default_values(self):
        """Test default values are correct."""
        metrics = StreamMetrics()
        assert metrics.start_time == 0.0
        assert metrics.first_token_time == 0.0
        assert metrics.end_time == 0.0
        assert metrics.total_chunks == 0
        assert metrics.total_content_length == 0
        assert metrics.total_tool_calls == 0
        assert metrics.error_count == 0
        assert metrics.retries == 0

    def test_time_to_first_token(self):
        """Test TTFT calculation."""
        metrics = StreamMetrics(start_time=100.0, first_token_time=100.5)
        assert metrics.time_to_first_token == 0.5

    def test_time_to_first_token_zero_when_not_set(self):
        """Test TTFT is zero when times not set."""
        metrics = StreamMetrics()
        assert metrics.time_to_first_token == 0.0

    def test_total_duration(self):
        """Test total duration calculation."""
        metrics = StreamMetrics(start_time=100.0, end_time=102.5)
        assert metrics.total_duration == 2.5

    def test_total_duration_zero_when_not_set(self):
        """Test duration is zero when times not set."""
        metrics = StreamMetrics()
        assert metrics.total_duration == 0.0

    def test_tokens_per_second(self):
        """Test tokens per second calculation."""
        metrics = StreamMetrics(
            start_time=100.0,
            end_time=101.0,
            total_content_length=400,  # ~100 tokens
        )
        # 400 chars / 4 chars per token = 100 tokens / 1 second = 100 tps
        assert metrics.tokens_per_second == 100.0

    def test_tokens_per_second_zero_when_no_duration(self):
        """Test TPS is zero when no duration."""
        metrics = StreamMetrics(total_content_length=400)
        assert metrics.tokens_per_second == 0.0


# =============================================================================
# SDK TYPE TESTS
# =============================================================================


class TestSDKType:
    """Tests for SDKType enum."""

    def test_all_types_defined(self):
        """Test all expected SDK types exist."""
        assert SDKType.OPENAI.value == "openai"
        assert SDKType.ANTHROPIC.value == "anthropic"
        assert SDKType.GOOGLE.value == "google"
        assert SDKType.OLLAMA.value == "ollama"
        assert SDKType.LMSTUDIO.value == "lmstudio"
        assert SDKType.VLLM.value == "vllm"
        assert SDKType.UNKNOWN.value == "unknown"


# =============================================================================
# OPENAI ADAPTER TESTS
# =============================================================================


class TestOpenAIAdapter:
    """Tests for OpenAI SDK adapter."""

    def test_sdk_type(self):
        """Test SDK type is correct."""
        adapter = OpenAIAdapter()
        assert adapter.sdk_type == SDKType.OPENAI

    @pytest.mark.asyncio
    async def test_adapt_stream_with_content(self):
        """Test adapting OpenAI stream with content."""
        adapter = OpenAIAdapter()

        # Create mock OpenAI chunks
        chunk1 = MockOpenAIChunk(
            choices=[MockOpenAIChoice(delta=MockOpenAIDelta(content="Hello"), finish_reason=None)]
        )
        chunk2 = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(delta=MockOpenAIDelta(content=" World"), finish_reason="stop")
            ]
        )

        async def mock_stream():
            yield chunk1
            yield chunk2

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 2
        assert results[0].content == "Hello"
        assert not results[0].is_final
        assert results[1].content == " World"
        assert results[1].is_final
        assert results[1].stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_adapt_stream_passthrough_streamchunk(self):
        """Test StreamChunk objects pass through unchanged."""
        adapter = OpenAIAdapter()

        original = StreamChunk(content="test", is_final=True)

        async def mock_stream():
            yield original

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0] is original

    @pytest.mark.asyncio
    async def test_adapt_stream_with_tool_calls(self):
        """Test adapting OpenAI stream with tool calls."""
        adapter = OpenAIAdapter()

        # Create mock tool call
        mock_func = MagicMock()
        mock_func.name = "read_file"
        mock_func.arguments = '{"path": "/test.py"}'

        mock_tc = MagicMock()
        mock_tc.function = mock_func

        chunk = MockOpenAIChunk(
            choices=[
                MockOpenAIChoice(
                    delta=MagicMock(content="", tool_calls=[mock_tc]), finish_reason=None
                )
            ]
        )

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].tool_calls is not None
        assert results[0].tool_calls[0]["name"] == "read_file"

    def test_detect_stream_method_stream(self):
        """Test detecting 'stream' method."""
        adapter = OpenAIAdapter()
        mock_provider = MagicMock()
        mock_provider.stream = MagicMock()

        assert adapter.detect_stream_method(mock_provider) == "stream"

    def test_detect_stream_method_stream_chat(self):
        """Test detecting 'stream_chat' method."""
        adapter = OpenAIAdapter()
        mock_provider = MagicMock(spec=[])
        mock_provider.stream_chat = MagicMock()

        assert adapter.detect_stream_method(mock_provider) == "stream_chat"

    def test_detect_stream_method_raises(self):
        """Test raises when no stream method."""
        adapter = OpenAIAdapter()
        mock_provider = MagicMock(spec=[])  # No methods

        with pytest.raises(AttributeError, match="has no stream method"):
            adapter.detect_stream_method(mock_provider)


# =============================================================================
# ANTHROPIC ADAPTER TESTS
# =============================================================================


class TestAnthropicAdapter:
    """Tests for Anthropic SDK adapter."""

    def test_sdk_type(self):
        """Test SDK type is correct."""
        adapter = AnthropicAdapter()
        assert adapter.sdk_type == SDKType.ANTHROPIC

    @pytest.mark.asyncio
    async def test_adapt_stream_content_delta(self):
        """Test adapting content block delta."""
        adapter = AnthropicAdapter()

        event = MockAnthropicEvent(
            type="content_block_delta", delta=MockAnthropicDelta(text="Hello")
        )

        async def mock_stream():
            yield event

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].content == "Hello"
        assert not results[0].is_final

    @pytest.mark.asyncio
    async def test_adapt_stream_message_stop(self):
        """Test adapting message stop event."""
        adapter = AnthropicAdapter()

        event = MockAnthropicEvent(type="message_stop")

        async def mock_stream():
            yield event

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].is_final
        assert results[0].stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_adapt_stream_tool_use(self):
        """Test adapting tool use event."""
        adapter = AnthropicAdapter()

        event = MockAnthropicEvent(type="tool_use", name="read_file", input={"path": "/test.py"})

        async def mock_stream():
            yield event

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].tool_calls is not None
        assert results[0].tool_calls[0]["name"] == "read_file"


# =============================================================================
# GOOGLE ADAPTER TESTS
# =============================================================================


class TestGoogleAdapter:
    """Tests for Google/Gemini SDK adapter."""

    def test_sdk_type(self):
        """Test SDK type is correct."""
        adapter = GoogleAdapter()
        assert adapter.sdk_type == SDKType.GOOGLE

    @pytest.mark.asyncio
    async def test_adapt_stream_text(self):
        """Test adapting text content."""
        adapter = GoogleAdapter()

        chunk = MockGeminiChunk(text="Hello World")

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "Hello World"

    @pytest.mark.asyncio
    async def test_adapt_stream_with_finish_reason(self):
        """Test adapting with finish reason."""
        adapter = GoogleAdapter()

        chunk = MockGeminiChunk(text="Done", candidates=[MockGeminiCandidate(finish_reason="STOP")])

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].is_final
        assert results[0].stop_reason == "STOP"


# =============================================================================
# OLLAMA ADAPTER TESTS
# =============================================================================


class TestOllamaAdapter:
    """Tests for Ollama SDK adapter."""

    def test_sdk_type(self):
        """Test SDK type is correct."""
        adapter = OllamaAdapter()
        assert adapter.sdk_type == SDKType.OLLAMA

    @pytest.mark.asyncio
    async def test_adapt_stream_dict_format(self):
        """Test adapting dict format stream."""
        adapter = OllamaAdapter()

        chunk1 = {"message": {"content": "Hello"}, "done": False}
        chunk2 = {"message": {"content": " World"}, "done": True}

        async def mock_stream():
            yield chunk1
            yield chunk2

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 2
        assert results[0].content == "Hello"
        assert not results[0].is_final
        assert results[1].content == " World"
        assert results[1].is_final

    @pytest.mark.asyncio
    async def test_adapt_stream_with_tool_calls(self):
        """Test adapting Ollama stream with tool calls."""
        adapter = OllamaAdapter()

        chunk = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "/test"}}}],
            },
            "done": False,
        }

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].tool_calls is not None
        assert results[0].tool_calls[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_adapt_stream_object_format(self):
        """Test adapting object format stream."""
        adapter = OllamaAdapter()

        chunk = MagicMock()
        chunk.message = MagicMock()
        chunk.message.content = "Test content"
        chunk.done = True

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "Test content"
        assert results[0].is_final


# =============================================================================
# LMSTUDIO ADAPTER TESTS
# =============================================================================


class TestLMStudioAdapter:
    """Tests for LMStudio SDK adapter."""

    def test_sdk_type(self):
        """Test SDK type is correct."""
        adapter = LMStudioAdapter()
        assert adapter.sdk_type == SDKType.LMSTUDIO

    @pytest.mark.asyncio
    async def test_adapt_stream_passthrough(self):
        """Test StreamChunk objects pass through."""
        adapter = LMStudioAdapter()

        original = StreamChunk(content="test", is_final=True)

        async def mock_stream():
            yield original

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0] is original


# =============================================================================
# VLLM ADAPTER TESTS
# =============================================================================


class TestVLLMAdapter:
    """Tests for vLLM SDK adapter."""

    def test_sdk_type(self):
        """Test SDK type is correct."""
        adapter = VLLMAdapter()
        assert adapter.sdk_type == SDKType.VLLM

    @pytest.mark.asyncio
    async def test_adapt_stream_dict_format(self):
        """Test adapting vLLM dict format."""
        adapter = VLLMAdapter()

        chunk = {"text": "Generated text", "finished": True}

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "Generated text"
        assert results[0].is_final

    @pytest.mark.asyncio
    async def test_adapt_stream_passthrough(self):
        """Test StreamChunk objects pass through."""
        adapter = VLLMAdapter()

        original = StreamChunk(content="test", is_final=True)

        async def mock_stream():
            yield original

        results = []
        async for chunk in adapter.adapt_stream(mock_stream()):
            results.append(chunk)

        assert len(results) == 1
        assert results[0] is original


# =============================================================================
# SDK ADAPTER REGISTRY TESTS
# =============================================================================


class TestSDKAdapterRegistry:
    """Tests for SDKAdapterRegistry."""

    def test_get_adapter_openai(self):
        """Test getting OpenAI adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.OPENAI)
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_adapter_anthropic(self):
        """Test getting Anthropic adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.ANTHROPIC)
        assert isinstance(adapter, AnthropicAdapter)

    def test_get_adapter_google(self):
        """Test getting Google adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.GOOGLE)
        assert isinstance(adapter, GoogleAdapter)

    def test_get_adapter_ollama(self):
        """Test getting Ollama adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.OLLAMA)
        assert isinstance(adapter, OllamaAdapter)

    def test_get_adapter_lmstudio(self):
        """Test getting LMStudio adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.LMSTUDIO)
        assert isinstance(adapter, LMStudioAdapter)

    def test_get_adapter_vllm(self):
        """Test getting vLLM adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.VLLM)
        assert isinstance(adapter, VLLMAdapter)

    def test_get_adapter_unknown_returns_openai(self):
        """Test unknown SDK type defaults to OpenAI."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.UNKNOWN)
        assert isinstance(adapter, OpenAIAdapter)

    def test_detect_sdk_type_anthropic(self):
        """Test detecting Anthropic provider."""
        provider = MockProvider(name="anthropic")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.ANTHROPIC

    def test_detect_sdk_type_google(self):
        """Test detecting Google provider."""
        provider = MockProvider(name="google")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.GOOGLE

    def test_detect_sdk_type_ollama(self):
        """Test detecting Ollama provider."""
        provider = MockProvider(name="ollama")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.OLLAMA

    def test_detect_sdk_type_lmstudio(self):
        """Test detecting LMStudio provider."""
        provider = MockProvider(name="lmstudio")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.LMSTUDIO

    def test_detect_sdk_type_vllm(self):
        """Test detecting vLLM provider."""
        provider = MockProvider(name="vllm")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.VLLM

    def test_detect_sdk_type_openai(self):
        """Test detecting OpenAI provider."""
        provider = MockProvider(name="openai")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.OPENAI

    def test_detect_sdk_type_unknown(self):
        """Test unknown provider returns UNKNOWN."""
        provider = MockProvider(name="custom_provider")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.UNKNOWN

    def test_detect_sdk_type_by_class_name(self):
        """Test detection by class name."""

        # Create a provider with Anthropic in class name
        class AnthropicTestProvider(MockProvider):
            pass

        provider = AnthropicTestProvider(name="test")
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.ANTHROPIC

    def test_register_adapter(self):
        """Test registering custom adapter."""
        custom_adapter = OpenAIAdapter()
        SDKAdapterRegistry.register_adapter(SDKType.UNKNOWN, custom_adapter)

        retrieved = SDKAdapterRegistry.get_adapter(SDKType.UNKNOWN)
        assert retrieved is custom_adapter


# =============================================================================
# UNIFIED STREAM ADAPTER TESTS
# =============================================================================


class TestUnifiedStreamAdapter:
    """Tests for UnifiedStreamAdapter."""

    def test_init_with_provider(self):
        """Test initialization with provider."""
        provider = MockProvider(name="ollama")
        adapter = UnifiedStreamAdapter(provider)

        assert adapter.provider is provider
        assert adapter.sdk_type == SDKType.OLLAMA

    def test_init_with_sdk_type_override(self):
        """Test initialization with SDK type override."""
        provider = MockProvider(name="custom")
        adapter = UnifiedStreamAdapter(provider, sdk_type=SDKType.OPENAI)

        assert adapter.sdk_type == SDKType.OPENAI

    def test_init_metrics_disabled(self):
        """Test initialization with metrics disabled."""
        provider = MockProvider()
        adapter = UnifiedStreamAdapter(provider, collect_metrics=False)

        assert adapter.last_metrics is None

    @pytest.mark.asyncio
    async def test_stream_basic(self):
        """Test basic streaming."""
        provider = MockProvider()
        provider.set_stream_chunks(
            [
                StreamChunk(content="Hello", is_final=False),
                StreamChunk(content=" World", is_final=True),
            ]
        )

        adapter = UnifiedStreamAdapter(provider)

        results = []
        async for chunk in adapter.stream(
            messages=[Message(role="user", content="Hi")], model="test"
        ):
            results.append(chunk)

        assert len(results) == 2
        assert results[0].content == "Hello"
        assert results[1].content == " World"

    @pytest.mark.asyncio
    async def test_stream_collects_metrics(self):
        """Test that streaming collects metrics."""
        provider = MockProvider()
        provider.set_stream_chunks(
            [
                StreamChunk(content="Test", is_final=True),
            ]
        )

        adapter = UnifiedStreamAdapter(provider, collect_metrics=True)

        async for _ in adapter.stream(messages=[Message(role="user", content="Hi")], model="test"):
            pass

        metrics = adapter.last_metrics
        assert metrics is not None
        assert metrics.total_chunks == 1
        assert metrics.total_content_length == 4  # "Test"

    @pytest.mark.asyncio
    async def test_stream_chat_alias(self):
        """Test stream_chat is alias for stream."""
        provider = MockProvider()
        provider.set_stream_chunks(
            [
                StreamChunk(content="Response", is_final=True),
            ]
        )

        adapter = UnifiedStreamAdapter(provider)

        results = []
        async for chunk in adapter.stream_chat(
            messages=[Message(role="user", content="Hi")], model="test"
        ):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].content == "Response"

    @pytest.mark.asyncio
    async def test_stream_completion_alias(self):
        """Test stream_completion is alias for stream."""
        provider = MockProvider()
        provider.set_stream_chunks(
            [
                StreamChunk(content="Response", is_final=True),
            ]
        )

        adapter = UnifiedStreamAdapter(provider)

        results = []
        async for chunk in adapter.stream_completion(
            messages=[Message(role="user", content="Hi")], model="test"
        ):
            results.append(chunk)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_collect_to_response(self):
        """Test collecting stream to single response."""
        provider = MockProvider()
        provider.set_stream_chunks(
            [
                StreamChunk(content="Hello", is_final=False),
                StreamChunk(content=" World", is_final=False),
                StreamChunk(content="!", stop_reason="stop", is_final=True),
            ]
        )

        adapter = UnifiedStreamAdapter(provider)

        response = await adapter.collect_to_response(
            messages=[Message(role="user", content="Hi")], model="test"
        )

        assert response.content == "Hello World!"
        assert response.stop_reason == "stop"
        assert response.model == "test"

    @pytest.mark.asyncio
    async def test_collect_to_response_with_tool_calls(self):
        """Test collecting response with tool calls."""
        provider = MockProvider()
        provider.set_stream_chunks(
            [
                StreamChunk(
                    content="",
                    tool_calls=[{"name": "read_file", "arguments": {"path": "/test"}}],
                    is_final=True,
                ),
            ]
        )

        adapter = UnifiedStreamAdapter(provider)

        response = await adapter.collect_to_response(
            messages=[Message(role="user", content="Read file")], model="test"
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_stream_error_records_metrics(self):
        """Test that errors are recorded in metrics."""
        provider = MockProvider()

        # Make stream raise an error after yielding
        async def error_stream(*args, **kwargs):
            yield StreamChunk(content="Start", is_final=False)
            raise ValueError("Stream error")

        provider.stream = error_stream

        adapter = UnifiedStreamAdapter(provider, collect_metrics=True)

        with pytest.raises(ValueError, match="Stream error"):
            async for _ in adapter.stream(
                messages=[Message(role="user", content="Hi")], model="test"
            ):
                pass

        metrics = adapter.last_metrics
        assert metrics is not None
        assert metrics.error_count == 1


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_wrap_provider(self):
        """Test wrap_provider creates adapter."""
        provider = MockProvider()
        adapter = wrap_provider(provider)

        assert isinstance(adapter, UnifiedStreamAdapter)
        assert adapter.provider is provider

    def test_detect_provider_sdk(self):
        """Test detect_provider_sdk function."""
        provider = MockProvider(name="anthropic")
        sdk_type = detect_provider_sdk(provider)

        assert sdk_type == SDKType.ANTHROPIC

    def test_detect_provider_sdk_unknown(self):
        """Test detect_provider_sdk with unknown provider."""
        provider = MockProvider(name="custom")
        sdk_type = detect_provider_sdk(provider)

        assert sdk_type == SDKType.UNKNOWN
