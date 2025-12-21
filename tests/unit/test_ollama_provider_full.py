import pytest
import httpx
import os
import json
import re
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any, AsyncIterator

from victor.providers.ollama_provider import OllamaProvider
from victor.providers.base import Message, ProviderError, ProviderTimeoutError, ToolDefinition


# Helper for async generator
async def async_line_generator(lines):
    for line in lines:
        yield line


# Mock get_provider_limits to prevent actual config file access
@pytest.fixture(autouse=True)
def mock_get_provider_limits_fixture():
    with patch('victor.config.config_loaders.get_provider_limits') as mock_limits:
        mock_limits.return_value = MagicMock(context_window=4096)
        yield mock_limits

# Temporarily disabled autouse to isolate issues with function signatures
@pytest.fixture() 
def mock_ollama_capability_detector_patterns_fixture():
    # Patch where it's imported in ollama_provider
    with patch('victor.providers.ollama_provider.TOOL_SUPPORT_PATTERNS', ['tool_pattern']) as mock_patterns:
        yield mock_patterns

@pytest.fixture
def mock_async_client():
    """Mocks httpx.AsyncClient and its async methods directly."""
    with patch('victor.providers.ollama_provider.httpx.AsyncClient') as MockAsyncClient:
        mock_instance = AsyncMock() 

        # Default mock response object for get/post, with non-awaitable json and awaitable aread
        mock_response_obj = MagicMock() # Changed to MagicMock for json() to be non-awaitable
        mock_response_obj.json.return_value = {} # json() is synchronous in httpx
        mock_response_obj.raise_for_status = MagicMock()
        mock_response_obj.status_code = 200
        mock_response_obj.aread = AsyncMock(return_value=b"") # aread() is asynchronous

        mock_instance.get.return_value = mock_response_obj
        mock_instance.post.return_value = mock_response_obj

        # Mock the stream method return value (an async context manager)
        mock_stream_response_obj = MagicMock() # This is the 'response' object inside async with
        mock_stream_response_obj.aiter_lines.return_value = async_line_generator([]) # Default empty async iterator
        mock_stream_response_obj.raise_for_status = MagicMock()
        mock_stream_response_obj.status_code = 200
        mock_stream_response_obj.aread = AsyncMock(return_value=b"")

        mock_stream_context_manager = AsyncMock() # This acts as the async context manager itself
        mock_stream_context_manager.__aenter__.return_value = mock_stream_response_obj
        mock_instance.stream = MagicMock(return_value=mock_stream_context_manager) # stream() method is synchronous, returns an async context manager

        mock_instance.aclose = AsyncMock()

        MockAsyncClient.return_value = mock_instance
        yield MockAsyncClient


@pytest.fixture
def mock_httpx_client():
    with patch('httpx.Client') as mock_client:
        yield mock_client


class TestOllamaProviderInit:
    def test_init_default_base_url(self, mock_async_client):
        OllamaProvider()
        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://localhost:11434"

    def test_init_single_base_url(self, mock_async_client):
        OllamaProvider(base_url="http://192.168.1.1:11434")
        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://192.168.1.1:11434"

    def test_init_list_base_urls(self, mock_async_client):
        OllamaProvider(base_url=["http://host1:11434", "http://host2:11434"])
        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://host1:11434"

    @patch.dict(os.environ, {"OLLAMA_ENDPOINTS": "http://env_host:11434,http://env_host2:11434"})
    def test_init_env_var_precedence(self, mock_async_client):
        OllamaProvider(base_url="http://explicit:11434")
        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://env_host:11434"

    def test_init_with_skip_discovery(self, mock_async_client):
        OllamaProvider(base_url="http://skip:11434", _skip_discovery=True)
        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://skip:11434"

    def test_name_property(self):
        provider = OllamaProvider()
        assert provider.name == "ollama"

    def test_supports_tools(self):
        provider = OllamaProvider()
        assert provider.supports_tools() is True

    def test_supports_streaming(self):
        provider = OllamaProvider()
        assert provider.supports_streaming() is True

    def test_get_context_window_cached(self, mock_get_provider_limits_fixture):
        provider = OllamaProvider()
        provider._context_window_cache["http://localhost:11434:test-model"] = 8192
        window = provider.get_context_window("test-model")
        assert window == 8192
        mock_get_provider_limits_fixture.assert_not_called()

    def test_get_context_window_from_config(self, mock_get_provider_limits_fixture):
        provider = OllamaProvider()
        mock_get_provider_limits_fixture.return_value.context_window = 16384
        window = provider.get_context_window("test-model")
        assert window == 16384
        mock_get_provider_limits_fixture.assert_called_once_with("ollama", "test-model")
        assert provider._context_window_cache["http://localhost:11434:test-model"] == 16384


class TestOllamaProviderAsyncFactory:
    @pytest.mark.asyncio
    async def test_create_async_discovery_success(self, mock_async_client):
        with patch.object(OllamaProvider, '_select_base_url_async', new=AsyncMock(return_value="http://async:11434")):
            await OllamaProvider.create(base_url="http://initial:11434")

        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://async:11434"

    @pytest.mark.asyncio
    async def test_create_async_discovery_fallback(self, mock_async_client):
        with patch.object(OllamaProvider, '_select_base_url_async', new=AsyncMock(return_value="http://localhost:11434")):
            await OllamaProvider.create(base_url="http://nonexistent:11434")
        
        mock_async_client.assert_called_once()
        assert mock_async_client.call_args[1]['base_url'] == "http://localhost:11434"


class TestOllamaProviderCapabilities:
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_ollama_capability_detector_patterns_fixture") # Apply fixture here
    async def test_discover_capabilities_success(self, mock_async_client):
        provider = OllamaProvider()
        # Configure the post return value for this specific test
        mock_async_client.return_value.post.return_value = MagicMock(
            json=MagicMock(return_value={
                "parameters": "some_param val\nnum_ctx 8192\nother_param",
                "template": "tool_pattern",
                "model_info": {} 
            }),
            raise_for_status=MagicMock(),
            status_code=200
        )

        caps = await provider.discover_capabilities("test-model")
        assert caps.context_window == 8192
        assert caps.supports_tools is True
        assert caps.source == "discovered"
        assert provider._context_window_cache["http://localhost:11434:test-model"] == 8192

    @pytest.mark.asyncio
    async def test_discover_capabilities_api_failure_fallback_to_config(self, mock_async_client, mock_get_provider_limits_fixture):
        provider = OllamaProvider()
        mock_async_client.return_value.post.side_effect = Exception("API error")
        mock_get_provider_limits_fixture.return_value.context_window = 4096

        caps = await provider.discover_capabilities("test-model")
        assert caps.context_window == 4096
        assert caps.supports_tools is True # Default optimistic for OllamaProvider itself
        assert caps.source == "config"

    def test_parse_context_window_num_ctx(self):
        provider = OllamaProvider()
        response_data = {"parameters": "some_param val\nnum_ctx 16384\nother_param"}
        window = provider._parse_context_window(response_data)
        assert window == 16384

    def test_parse_context_window_model_info_context_length(self):
        provider = OllamaProvider()
        response_data = {"model_info": {"context_length": 32768}}
        window = provider._parse_context_window(response_data)
        assert window == 32768

    def test_parse_context_window_model_info_context_length_string(self):
        provider = OllamaProvider()
        response_data = {"model_info": {"context_length": "32768"}}
        window = provider._parse_context_window(response_data)
        assert window == 32768

    def test_parse_context_window_no_context_info(self):
        provider = OllamaProvider()
        response_data = {"parameters": "no context here", "model_info": {}}
        window = provider._parse_context_window(response_data)
        assert window is None

    @pytest.mark.usefixtures("mock_ollama_capability_detector_patterns_fixture") # Apply fixture here
    def test_detect_tool_support_true(self):
        provider = OllamaProvider()
        template = "This model supports tool_pattern calls."
        assert provider._detect_tool_support(template) is True

    def test_detect_tool_support_false(self):
        provider = OllamaProvider()
        template = "This model does not support tools."
        # Temporarily patch TOOL_SUPPORT_PATTERNS to something that won't match
        with patch('victor.providers.ollama_provider.TOOL_SUPPORT_PATTERNS', ['non_matching_pattern']):
            assert provider._detect_tool_support(template) is False

    def test_detect_tool_support_empty_template(self):
        provider = OllamaProvider()
        template = ""
        assert provider._detect_tool_support(template) is True # Optimistic default

    def test_detect_tool_support_pattern_exception(self):
        provider = OllamaProvider()
        with patch('re.search', side_effect=Exception("Regex error")):
            template = "tool_pattern"
            assert provider._detect_tool_support(template) is False


class TestOllamaProviderChat:
    @pytest.mark.asyncio
    async def test_chat_success(self, mock_async_client):
        provider = OllamaProvider()
        mock_async_client.return_value.post.return_value = MagicMock(
            json=MagicMock(return_value={
                "message": {"content": "Hello", "tool_calls": []},
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            }),
            raise_for_status=MagicMock(),
            status_code=200
        )

        messages = [Message(role="user", content="Hi")]
        response = await provider.chat(messages, model="test-model")
        assert response.content == "Hello"
        mock_async_client.return_value.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_timeout(self, mock_async_client):
        provider = OllamaProvider()
        # Mock _execute_with_circuit_breaker to raise the specific error
        with patch.object(provider, '_execute_with_circuit_breaker', side_effect=httpx.TimeoutException("Timeout")) as mock_breaker:
            with pytest.raises(ProviderTimeoutError):
                await provider.chat([Message(role="user", content="Hi")], model="test-model")
            mock_breaker.assert_called_once()


    @pytest.mark.asyncio
    async def test_chat_http_error(self, mock_async_client):
        provider = OllamaProvider()
        mock_response = MagicMock(status_code=500)
        mock_response.text = "Internal Server Error"
        # Mock _execute_with_circuit_breaker to raise the specific error
        with patch.object(provider, '_execute_with_circuit_breaker', side_effect=httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_response)) as mock_breaker:
            with pytest.raises(ProviderError) as exc_info:
                await provider.chat([Message(role="user", content="Hi")], model="test-model")
            assert exc_info.value.status_code == 500
            mock_breaker.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_tools_not_supported_retry(self, mock_async_client):
        provider = OllamaProvider(base_url="http://test:11434")
        messages = [Message(role="user", content="Test")]
        tools = [ToolDefinition(name="test_tool", description="desc", parameters={})]

        # First call fails due to tools not supported
        mock_response_400 = MagicMock(status_code=400, text="model does not support tools")
        mock_error_400 = httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_response_400)

        # Second call succeeds without tools
        mock_response_200 = MagicMock(
            raise_for_status=MagicMock(),
            status_code=200
        )
        mock_response_200.json.return_value = { # Corrected json mocking
            "message": {"content": "Fallback response", "tool_calls": []},
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

        # Patch _execute_with_circuit_breaker to return the responses
        with patch.object(provider, '_execute_with_circuit_breaker', side_effect=[
            mock_error_400, # First call raises error directly
            AsyncMock(return_value=mock_response_200)                      # Second call returns success
        ]) as mock_breaker:
            response = await provider.chat(messages, model="test-model", tools=tools)
            assert response.content == "Fallback response"
            assert "test-model" in provider._models_without_tools  # Model should be cached
            assert mock_breaker.call_count == 2
            # Check that the second call had tools=None
            second_call_kwargs_json = mock_breaker.call_args_list[1].args[2]['json']
            assert second_call_kwargs_json['tools'] is None

    @pytest.mark.asyncio
    async def test_chat_tool_call_from_content_fallback(self, mock_async_client):
        provider = OllamaProvider()
        mock_async_client.return_value.post.return_value = MagicMock(
            json=MagicMock(return_value={
                "message": {"content": '{"name": "test_tool", "arguments": {"a": 1}}'},
                "done_reason": "tool_code",
                "prompt_eval_count": 1,
                "eval_count": 1,
            }),
            raise_for_status=MagicMock(),
            status_code=200
        )

        messages = [Message(role="user", content="Hi")]
        response = await provider.chat(messages, model="test-model")
        assert response.content == "" # Content should be cleared
        assert response.tool_calls[0]['name'] == 'test_tool'


class TestOllamaProviderStream:
    @pytest.mark.asyncio
    async def test_stream_success(self, mock_async_client):
        provider = OllamaProvider()
        # Configure the mock_stream_response directly
        mock_stream_response_obj = mock_async_client.return_value.stream.return_value.__aenter__.return_value
        mock_stream_response_obj.aiter_lines.return_value = async_line_generator([ # Using the helper for async generator
            '{"message": {"content": "Hello "}}',
            '{"message": {"content": "world"}, "done": true, "done_reason": "stop", "model": "test-model"}'
        ])

        messages = [Message(role="user", content="Hi")]
        chunks = [chunk async for chunk in provider.stream(messages, model="test-model")]
        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "world"
        assert chunks[1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_timeout(self, mock_async_client):
        provider = OllamaProvider()
        mock_async_client.return_value.stream.side_effect = httpx.TimeoutException("Stream Timeout")

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(ProviderTimeoutError):
            async for _ in provider.stream(messages, model="test-model"):
                pass

    @pytest.mark.asyncio
    async def test_stream_http_error(self, mock_async_client):
        provider = OllamaProvider()
        mock_stream_response_obj = mock_async_client.return_value.stream.return_value.__aenter__.return_value
        mock_stream_response_obj.status_code = 500
        mock_stream_response_obj.aread.return_value = b"Internal Server Error"
        mock_stream_response_obj.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=mock_stream_response_obj)

        messages = [Message(role="user", content="Hi")]
        with pytest.raises(ProviderError):
            async for _ in provider.stream(messages, model="test-model"):
                pass

    @pytest.mark.asyncio
    async def test_stream_tools_not_supported_retry(self, mock_async_client):
        provider = OllamaProvider(base_url="http://test:11434")
        messages = [Message(role="user", content="Test")]
        tools = [ToolDefinition(name="test_tool", description="desc", parameters={})]

        # Mock responses for the side_effect sequence
        mock_error_response_400 = MagicMock(status_code=400)
        mock_error_response_400.aread = AsyncMock(return_value=b"model does not support tools") 
        mock_error_response_400.raise_for_status.side_effect = httpx.HTTPStatusError("Tools unsupported", request=MagicMock(), response=mock_error_response_400)

        mock_success_response_200 = MagicMock(status_code=200) 
        mock_success_response_200.aiter_lines.return_value = async_line_generator([ # Set return_value here
                '{"message": {"content": "Retry success"}, "done": true, "model": "test-model"}'
            ])
        mock_success_response_200.raise_for_status.return_value = None
        mock_success_response_200.aread = AsyncMock(return_value=b"") 

        mock_async_client.return_value.stream.side_effect = [
            AsyncMock(__aenter__=AsyncMock(return_value=mock_error_response_400)),
            AsyncMock(__aenter__=AsyncMock(return_value=mock_success_response_200))
        ]

        chunks = [chunk async for chunk in provider.stream(messages, model="test-model", tools=tools)]
        assert chunks[0].content == "Retry success"
        assert "test-model" in provider._models_without_tools

        assert mock_async_client.return_value.stream.call_count == 2
        second_call_kwargs = mock_async_client.return_value.stream.call_args_list[1].kwargs
        assert second_call_kwargs['json']['tools'] is None

    @pytest.mark.asyncio
    async def test_stream_json_decode_error(self, mock_async_client):
        provider = OllamaProvider()
        mock_stream_response = mock_async_client.return_value.stream.return_value.__aenter__.return_value
        mock_stream_response.aiter_lines.return_value = async_line_generator([ # Set return_value here
            '{"message": {"content": "Hello "}}',
            'INVALID JSON LINE',  # This should cause JSONDecodeError
            '{"message": {"content": "world"}, "done": true, "done_reason": "stop", "model": "test-model"}'
        ])

        messages = [Message(role="user", content="Hi")]
        chunks = [chunk async for chunk in provider.stream(messages, model="test-model")]
        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "world"

    @pytest.mark.asyncio
    async def test_stream_tool_call_from_content_fallback(self, mock_async_client):
        provider = OllamaProvider()
        mock_stream_response = mock_async_client.return_value.stream.return_value.__aenter__.return_value
        mock_stream_response.aiter_lines.return_value = async_line_generator([ # Set return_value here
            '{"message": {"content": ""}}',
            '{"message": {"content": "{\"name\": \"test_tool\", \"arguments\": {\"a\": 1}}"}, "done": true, "model": "test-model"}'
        ])
        messages = [Message(role="user", content="Hi")]
        chunks = [chunk async for chunk in provider.stream(messages, model="test-model")]
        assert len(chunks) == 2 # First is empty, second is tool call
        assert chunks[1].content == "" # Content cleared
        assert chunks[1].tool_calls[0]['name'] == 'test_tool'


class TestOllamaProviderUtilityMethods:
    def test_build_request_payload_no_tools(self):
        provider = OllamaProvider()
        messages = [Message(role="user", content="Test message")]
        payload = provider._build_request_payload(
            messages=messages, model="test-model", temperature=0.7, max_tokens=100, tools=None, stream=False
        )
        assert "tools" not in payload
        assert payload["messages"][0]["content"] == "Test message"

    def test_build_request_payload_with_tools(self):
        provider = OllamaProvider()
        messages = [Message(role="user", content="Test message")]
        tools = [ToolDefinition(name="get_weather", description="Weather tool", parameters={})]
        payload = provider._build_request_payload(
            messages=messages, model="test-model", temperature=0.7, max_tokens=100, tools=tools, stream=True
        )
        assert "tools" in payload
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["stream"] is True

    def test_normalize_tool_calls_openai_format(self):
        provider = OllamaProvider()
        raw_tool_calls = [
            {"id": "call_1", "function": {"name": "func1", "arguments": {"a": 1}}},
            {"id": "call_2", "function": {"name": "func2", "arguments": {"b": 2}}},
        ]
        normalized = provider._normalize_tool_calls(raw_tool_calls)
        assert len(normalized) == 2
        assert normalized[0]["name"] == "func1"
        assert normalized[1]["arguments"]["b"] == 2

    def test_normalize_tool_calls_already_normalized(self):
        provider = OllamaProvider()
        raw_tool_calls = [
            {"name": "func1", "arguments": {"a": 1}},
        ]
        normalized = provider._normalize_tool_calls(raw_tool_calls)
        assert len(normalized) == 1
        assert normalized[0]["name"] == "func1"

    def test_normalize_tool_calls_empty_or_none(self):
        provider = OllamaProvider()
        assert provider._normalize_tool_calls(None) is None
        assert provider._normalize_tool_calls([]) is None

    def test_parse_response_with_usage(self):
        provider = OllamaProvider()
        result = {
            "message": {"content": "Response", "tool_calls": []},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        response = provider._parse_response(result, "test-model")
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30

    def test_parse_response_no_usage(self):
        provider = OllamaProvider()
        result = {
            "message": {"content": "Response"},
            "done_reason": "stop",
        }
        response = provider._parse_response(result, "test-model")
        assert response.usage is None

    def test_parse_stream_chunk_tool_call_from_content_fallback(self):
        provider = OllamaProvider()
        chunk_data = {
            "message": {"content": '{"name": "tool", "parameters": {"p": 1}}'},
            "done": True,
            "model": "test-model"
        }
        chunk = provider._parse_stream_chunk(chunk_data)
        assert chunk.content == ""
        assert chunk.tool_calls[0]['name'] == 'tool'
        assert chunk.is_final is True

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_async_client):
        provider = OllamaProvider()
        mock_async_client.return_value.get.return_value = MagicMock(
            json=MagicMock(return_value={"models": [{"name": "llama2"}]}),
            raise_for_status=MagicMock(),
            status_code=200
        )

        models = await provider.list_models()
        assert models[0]["name"] == "llama2"
        mock_async_client.return_value.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_models_error(self, mock_async_client):
        provider = OllamaProvider()
        mock_async_client.return_value.get.side_effect = Exception("List error")

        with pytest.raises(ProviderError):
            await provider.list_models()

    @pytest.mark.asyncio
    async def test_pull_model_success(self, mock_async_client):
        provider = OllamaProvider()
        # Get the mock_stream_response from the fixture setup
        mock_stream_response = mock_async_client.return_value.stream.return_value.__aenter__.return_value
        mock_stream_response.aiter_lines.return_value = async_line_generator([
            '{"status": "pulling"}',
            '{"status": "success"}'
        ])
        progress_updates = [update async for update in provider.pull_model("llama2")]
        assert len(progress_updates) == 2
        assert progress_updates[0]["status"] == "pulling"

    @pytest.mark.asyncio
    async def test_pull_model_error(self, mock_async_client):
        provider = OllamaProvider()
        mock_async_client.return_value.stream.side_effect = Exception("Pull error")

        with pytest.raises(ProviderError):
            async for _ in provider.pull_model("llama2"):
                pass

    @pytest.mark.asyncio
    async def test_close_client(self, mock_async_client):
        provider = OllamaProvider()
        await provider.close()
        mock_async_client.return_value.aclose.assert_called_once()