# Tutorial: Integrating a New LLM Provider - Part 4

**Part 4 of 5:** Model Capabilities Configuration, Provider Registration, and Testing

---

## Navigation

- [Part 1: Architecture & Steps 1-2](part-1-provider-architecture.md)
- [Part 2: Streaming & Error Handling](part-2-streaming-error-handling.md)
- [Part 3: Tool Calling Adapter](part-3-tool-calling-adapter.md)
- **[Part 4: Registration & Testing](#)** (Current)
- [Part 5: Best Practices & Examples](part-5-best-practices-examples.md)
- [**Complete Guide**](integrate-provider.md)

---

## 4. Model Capabilities Configuration

Add your provider and models to `victor/config/model_capabilities.yaml`:

```yaml
# Add to provider_defaults section
provider_defaults:
  custom_llm:
    native_tool_calls: true
    streaming_tool_calls: true
    parallel_tool_calls: true
    tool_choice_param: true
    requires_strict_prompting: false
    recommended_max_tools: 30
    recommended_tool_budget: 15

# Add to models section
models:
  "custom-llm-large*":
    training:
      tool_calling: true
      code_generation: true

    providers:
      custom_llm:
        native_tool_calls: true
        streaming_tool_calls: true
        parallel_tool_calls: true

    settings:
      recommended_max_tools: 40
      recommended_tool_budget: 20
      argument_format: json

  "custom-llm-small*":
    training:
      tool_calling: true
      code_generation: false

    providers:
      custom_llm:
        native_tool_calls: true
        parallel_tool_calls: false  # Smaller model limitation

    settings:
      recommended_max_tools: 20
      recommended_tool_budget: 10
```text

---

## 5. Provider Registration

### Method 1: Internal Registration

Add your provider to `victor/providers/registry.py`:

```python
def _register_default_providers() -> None:
    """Register all default providers."""
    # ... existing providers ...

    # Add your provider
    from victor.providers.custom_provider import CustomLLMProvider
    ProviderRegistry.register("custom_llm", CustomLLMProvider)
    ProviderRegistry.register("customllm", CustomLLMProvider)  # Alias
```

### Method 2: Plugin Registration (External)

For external packages, use entry points in `pyproject.toml`:

```toml
[project.entry-points."victor.providers"]
custom_llm = "victor_custom:CustomLLMProvider"
```text

---

## 6. Testing Your Provider

### Unit Tests

Create `tests/unit/providers/test_custom_provider.py`:

```python
"""Unit tests for CustomLLM provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.custom_provider import CustomLLMProvider
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
)


@pytest.fixture
def custom_provider():
    """Create CustomLLMProvider instance for testing."""
    return CustomLLMProvider(
        api_key="test-api-key",
        base_url="https://api.customllm.example.com/v1",
        timeout=30,
    )


class TestCustomLLMProvider:
    """Tests for CustomLLMProvider."""

    def test_initialization(self, custom_provider):
        """Test provider initializes correctly."""
        assert custom_provider.name == "custom_llm"
        assert custom_provider.supports_tools() is True
        assert custom_provider.supports_streaming() is True

    @pytest.mark.asyncio
    async def test_chat_success(self, custom_provider):
        """Test successful chat completion."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello! How can I help?",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            messages = [Message(role="user", content="Hello")]
            response = await custom_provider.chat(
                messages=messages,
                model="custom-llm-large",
            )

            assert response.content == "Hello! How can I help?"
            assert response.role == "assistant"
            assert response.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, custom_provider):
        """Test chat with tool calling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            tools = [
                ToolDefinition(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                )
            ]

            messages = [Message(role="user", content="Weather in London?")]
            response = await custom_provider.chat(
                messages=messages,
                model="custom-llm-large",
                tools=tools,
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["name"] == "get_weather"
            assert response.tool_calls[0]["arguments"] == {"location": "London"}

    @pytest.mark.asyncio
    async def test_chat_auth_error(self, custom_provider):
        """Test authentication error handling."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"

        error = httpx.HTTPStatusError(
            "Auth failed",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value.raise_for_status.side_effect = error

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderAuthError):
                await custom_provider.chat(
                    messages=messages,
                    model="custom-llm-large",
                )

    @pytest.mark.asyncio
    async def test_chat_rate_limit_error(self, custom_provider):
        """Test rate limit error handling."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        error = httpx.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value.raise_for_status.side_effect = error

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderRateLimitError):
                await custom_provider.chat(
                    messages=messages,
                    model="custom-llm-large",
                )

    @pytest.mark.asyncio
    async def test_close(self, custom_provider):
        """Test closing the provider."""
        with patch.object(
            custom_provider.client, "aclose", new_callable=AsyncMock
        ) as mock_close:
            await custom_provider.close()
            mock_close.assert_called_once()


class TestCustomLLMProviderStreaming:
    """Tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_success(self, custom_provider):
        """Test successful streaming."""
        # Mock streaming response
        async def mock_aiter_lines():
            lines = [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                'data: [DONE]',
            ]
            for line in lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock()

        with patch.object(
            custom_provider.client, "stream"
        ) as mock_stream:
            mock_stream.return_value = mock_context

            messages = [Message(role="user", content="Hello")]
            chunks = []

            async for chunk in custom_provider.stream(
                messages=messages,
                model="custom-llm-large",
            ):
                chunks.append(chunk)

            # Verify chunks received
            assert len(chunks) >= 2
            content = "".join(c.content for c in chunks)
            assert "Hello" in content
```

### Running Tests

```bash
# Run provider tests
pytest tests/unit/providers/test_custom_provider.py -v

# Run with coverage
pytest tests/unit/providers/test_custom_provider.py --cov=victor/providers/custom_provider

# Run all provider tests
pytest tests/unit/providers/ -v
```text

### Integration Testing

For integration testing with a real API:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_api_chat():
    """Integration test with real API (requires API key)."""
    import os

    api_key = os.environ.get("CUSTOM_LLM_API_KEY")
    if not api_key:
        pytest.skip("CUSTOM_LLM_API_KEY not set")

    provider = CustomLLMProvider(api_key=api_key)

    try:
        messages = [Message(role="user", content="Say 'Hello, Victor!'")]
        response = await provider.chat(
            messages=messages,
            model="custom-llm-large",
            max_tokens=50,
        )

        assert response.content
        assert len(response.content) > 0
    finally:
        await provider.close()
```

---

**Continue to [Part 5: Best Practices & Examples](part-5-best-practices-examples.md)**

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 4 min
**Last Updated:** February 01, 2026
**Part 4 of 5**
