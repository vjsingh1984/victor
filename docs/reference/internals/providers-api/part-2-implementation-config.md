# Providers API Reference - Part 2

**Part 2 of 3:** Provider Implementation and Configuration

---

## Navigation

- [Part 1: Interface, Protocols, Classes](part-1-interface-protocols-classes.md)
- **[Part 2: Implementation & Config](#)** (Current)
- [Part 3: Error Handling & Reference](part-3-error-handling-reference.md)
- [**Complete Reference**](../providers-api.md)

---

To implement a custom provider, follow this pattern:

```python
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)


class CustomProvider(BaseProvider):
    """Custom LLM provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize custom provider.

        Args:
            api_key: API key for authentication
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        # Initialize your client here
        self.client = YourAPIClient(api_key=api_key, ...)

    @property
    def name(self) -> str:
        """Provider name."""
        return "custom"

    def supports_tools(self) -> bool:
        """Return True if your provider supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Return True if your provider supports streaming."""
        return True

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with generated content
        """
        try:
            # Convert messages to your provider's format
            formatted_messages = self._format_messages(messages)

            # Make API call with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.chat,
                messages=formatted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=self._convert_tools(tools) if tools else None,
                **kwargs,
            )

            # Parse and return normalized response
            return self._parse_response(response, model)

        except Exception as e:
            raise self._handle_error(e)

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion response.

        Yields:
            StreamChunk objects with incremental content
        """
        try:
            formatted_messages = self._format_messages(messages)

            async for chunk in self.client.stream(
                messages=formatted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield StreamChunk(
                    content=chunk.text or "",
                    is_final=chunk.is_done,
                    tool_calls=chunk.tool_calls if chunk.is_done else None,
                    stop_reason="stop" if chunk.is_done else None,
                )

        except Exception as e:
            raise self._handle_error(e)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()

    def _format_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert standard messages to provider format."""
        return [{"role": m.role, "content": m.content} for m in messages]

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict]:
        """Convert standard tools to provider format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    def _parse_response(self, response: Any, model: str) -> CompletionResponse:
        """Parse API response to standard format."""
        return CompletionResponse(
            content=response.text,
            role="assistant",
            tool_calls=response.tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input,
                "completion_tokens": response.usage.output,
                "total_tokens": response.usage.total,
            },
            model=model,
        )

    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert API errors to standard ProviderError."""
        raise ProviderError(
            message=f"Custom provider error: {error}",
            provider=self.name,
            raw_error=error,
        )
```text

### Registering Your Provider

```python
from victor.providers.registry import ProviderRegistry

# Register your provider
ProviderRegistry.register("custom", CustomProvider)

# Create an instance
provider = ProviderRegistry.create(
    "custom",
    api_key="your-api-key",
    base_url="https://api.example.com",
)
```

---

## Configuration

### Model Capabilities

Model capabilities are configured in `victor/config/model_capabilities.yaml`. This file defines:

1. **Global Defaults** - Default capability assumptions for unknown models
2. **Provider Defaults** - Infrastructure-level capabilities for each provider
3. **Model Configurations** - Specific settings for model patterns

#### Schema Structure

```yaml
schema_version: "0.2.0"

# Global defaults for unknown models
defaults:
  training:
    tool_calling: false
    thinking_mode: false
    code_generation: false
    vision: false
    reasoning: false

  provider_support:
    native_tool_calls: false
    streaming_tool_calls: false
    parallel_tool_calls: false
    tool_choice_param: false

  settings:
    recommended_max_tools: 15
    recommended_tool_budget: 10
    requires_strict_prompting: true
    argument_format: json
    exploration_multiplier: 1.0
    continuation_patience: 10
    timeout_multiplier: 1.0
    tool_reliability: medium

# Provider-level defaults
provider_defaults:
  anthropic:
    native_tool_calls: true
    streaming_tool_calls: true
    parallel_tool_calls: true
    tool_choice_param: true
    requires_strict_prompting: false
    recommended_max_tools: 50
    recommended_tool_budget: 20

  openai:
    native_tool_calls: true
    streaming_tool_calls: true
    parallel_tool_calls: true
    # ...

  ollama:
    native_tool_calls: false  # Must be enabled per-model
    timeout_multiplier: 2.0   # Local inference needs longer timeouts
    # ...

# Model-specific configurations
models:
  "claude-*":
    training:
      tool_calling: true
      code_generation: true

    providers:
      anthropic:
        native_tool_calls: true
        streaming_tool_calls: true
        parallel_tool_calls: true

    settings:
      recommended_max_tools: 50
      recommended_tool_budget: 25
      argument_format: json
```text

#### Resolution Order

Capabilities are resolved in this order (later overrides earlier):

1. `defaults`
2. `provider_defaults.<provider>`
3. `models.<pattern>.training` (inherent capabilities)
4. `models.<pattern>.providers.<provider>` (provider-specific overrides)
5. `models.<pattern>.settings` (tuning)

### Supported Providers

Victor supports 21 LLM providers out of the box:

| Provider | Name | Type | Tool Calling |
|----------|------|------|--------------|
| Anthropic | `anthropic` | Cloud | Native |
| OpenAI | `openai` | Cloud | Native |
| Google AI | `google` | Cloud | Native |
| xAI (Grok) | `xai`, `grok` | Cloud | Native |
| Azure OpenAI | `azure`, `azure-openai` | Enterprise | Native |
| AWS Bedrock | `bedrock`, `aws` | Enterprise | Native |
| Vertex AI | `vertex`, `vertexai` | Enterprise | Native |
| DeepSeek | `deepseek` | Cloud | Native |
| Mistral | `mistral` | Cloud | Native |
| Groq | `groqcloud` | Cloud | Native |
| Together AI | `together` | Cloud | Native |
| Fireworks AI | `fireworks` | Cloud | Native |
| OpenRouter | `openrouter` | Gateway | Native |
| Cerebras | `cerebras` | Cloud | Native |
| Moonshot (Kimi) | `moonshot`, `kimi` | Cloud | Native |
| ZhipuAI | `zai`, `zhipuai`, `zhipu` | Cloud | Native |
| HuggingFace | `huggingface`, `hf` | Cloud | Varies |
| Replicate | `replicate` | Cloud | Limited |
| Ollama | `ollama` | Local | Per-model |
| LMStudio | `lmstudio` | Local | Native |
| vLLM | `vllm` | Local | Native |
| llama.cpp | `llamacpp`, `llama-cpp`, `llama.cpp` | Local | Native |


**Reading Time:** 4 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Error Handling
