# Test Mocks Infrastructure

This directory provides production-quality mock implementations for testing Victor components without external dependencies.

## Overview

The mock infrastructure provides:

- **MockBaseProvider** - Configurable mock with controllable responses
- **FailingProvider** - Simulates various failure modes (timeouts, rate limits, etc.)
- **StreamingTestProvider** - Specialized for streaming response testing
- **ToolCallMockProvider** - Provider with predefined tool call responses
- **ProviderTestHelpers** - Utility functions for common test scenarios

## Usage

### Basic Mock Provider

```python
from tests.mocks import MockBaseProvider
from victor.providers.base import Message

# Create simple mock
provider = MockBaseProvider(response_text="Hello, world!")
messages = [Message(role="user", content="Test")]

# Use in tests
response = await provider.chat(messages, model="test-model")
assert response.content == "Hello, world!"
```

### Failing Provider

```python
from tests.mocks import FailingProvider
import pytest
from victor.core.errors import ProviderTimeoutError

# Provider that fails immediately
provider = FailingProvider(error_type="timeout")

# Test error handling
with pytest.raises(ProviderTimeoutError):
    await provider.chat(messages, model="test")
```

### Streaming Provider

```python
from tests.mocks import StreamingTestProvider

# Control exact chunks
provider = StreamingTestProvider(
    chunks=["Hello", " world", "!"],
    chunk_delay=0.05,
)

chunks = []
async for chunk in provider.stream(messages, model="test"):
    chunks.append(chunk.content)

assert chunks == ["Hello", " world", "!"]
```

### Tool Call Provider

```python
from tests.mocks import ToolCallMockProvider, ProviderTestHelpers

# Create tool call
tool_call = ProviderTestHelpers.create_test_tool_call(
    name="search",
    arguments={"query": "test"}
)

# Provider returns tool calls
provider = ToolCallMockProvider(
    response_text="Searching...",
    tool_calls=[tool_call],
)

response = await provider.chat(messages, model="test")
assert response.tool_calls is not None
```

### Helper Functions

```python
from tests.mocks import ProviderTestHelpers

# Create test data
messages = ProviderTestHelpers.create_test_messages("Hello")
tool_call = ProviderTestHelpers.create_test_tool_call("search", {"q": "test"})

# Collect streaming chunks
chunks = await ProviderTestHelpers.collect_stream_chunks(
    provider, messages, model="test"
)

# Validate responses
ProviderTestHelpers.assert_valid_response(response)
ProviderTestHelpers.assert_valid_stream_chunks(chunks)
```

## Provider Features

### MockBaseProvider

- **Configurable response text** - Set exact content to return
- **Network delay simulation** - Add artificial latency
- **Token usage tracking** - Auto-calculate or provide custom usage
- **Call tracking** - Monitor invocation count and parameters
- **Tool support toggle** - Enable/disable tool calling
- **Streaming support** - Split response into realistic chunks

### FailingProvider

Supports all error types from `victor.core.errors`:
- `timeout` - ProviderTimeoutError
- `rate_limit` - ProviderRateLimitError
- `auth` - ProviderAuthError
- `connection` - ProviderConnectionError
- `invalid_response` - ProviderInvalidResponseError
- `generic` - ProviderError

Features:
- **Fail after N successes** - Test retry logic
- **Custom error messages** - Specific error scenarios
- **Call counting** - Verify retry attempts

### StreamingTestProvider

- **Exact chunk control** - Define chunk sequence
- **Inter-chunk delays** - Simulate network latency
- **Chunk tracking** - Inspect streamed chunks after test
- **Final metadata** - Automatic final chunk with usage info

### ToolCallMockProvider

- **Predefined tool calls** - Return specific tool calls
- **Multi-turn sequences** - Simulate conversation flows
- **Call history** - Track all invocations
- **Mixed responses** - Text + tool calls in same response

## Examples

### Testing Retry Logic

```python
async def test_retry_on_timeout():
    """Test that retries work on timeout."""
    provider = FailingProvider(
        error_type="timeout",
        fail_after=3  # First 3 succeed, then fail
    )

    messages = [Message(role="user", content="Test")]

    # First 3 calls should succeed
    for _ in range(3):
        response = await provider.chat(messages, model="test")
        assert "Success" in response.content

    # Fourth call fails
    with pytest.raises(ProviderTimeoutError):
        await provider.chat(messages, model="test")
```

### Testing Streaming

```python
async def test_streaming_aggregation():
    """Test aggregating streaming chunks."""
    provider = MockBaseProvider(
        response_text="Hello world!",
        response_delay=0.01,
    )

    messages = [Message(role="user", content="Test")]

    chunks = []
    async for chunk in provider.stream(messages, model="test"):
        chunks.append(chunk.content)

    full_content = "".join(chunks)
    assert full_content == "Hello world!"
```

### Testing Tool Calling

```python
async def test_tool_call_execution():
    """Test tool call in response."""
    from tests.mocks import ProviderTestHelpers

    tool_call = ProviderTestHelpers.create_test_tool_call(
        name="get_weather",
        arguments={"location": "SF"}
    )

    provider = ToolCallMockProvider(
        response_text="I'll check the weather.",
        tool_calls=[tool_call],
    )

    response = await provider.chat(messages, model="test")

    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
```

## Integration with Existing Tests

The mocks are fully compatible with Victor's BaseProvider interface and can be used anywhere a real provider would be used:

```python
# In orchestrator tests
from tests.mocks import MockBaseProvider

async def test_orchestrator_with_mock():
    mock_provider = MockBaseProvider(response_text="Test response")
    orchestrator = AgentOrchestrator(provider=mock_provider)

    result = await orchestrator.process("Test query")
    assert "Test response" in result
```

## Running Tests

```bash
# Run mock tests
pytest tests/mocks/test_provider_mocks.py -v

# Run specific test class
pytest tests/mocks/test_provider_mocks.py::TestMockBaseProvider -v

# Run with coverage
pytest tests/mocks/ --cov=tests.mocks --cov-report=html
```

## Design Principles

1. **Interface Compliance** - All mocks implement BaseProvider exactly
2. **Test Isolation** - No shared state between instances
3. **Configurability** - Rich configuration options for various scenarios
4. **Observability** - Track calls, requests, and internal state
5. **Documentation** - Comprehensive docstrings with examples
6. **Type Safety** - Full type hints for IDE support

## Contributing

When adding new mock providers:

1. Inherit from `BaseProvider`
2. Implement all abstract methods (`chat`, `stream`, `close`, `name`)
3. Add comprehensive docstrings
4. Include usage examples
5. Add tests in `test_provider_mocks.py`
6. Export from `__init__.py`
