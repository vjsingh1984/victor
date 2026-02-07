# Tutorial: Integrating a New LLM Provider

This tutorial has been split into 5 parts for easier reading and navigation.

---

## Quick Start

**Time estimate:** 45-60 minutes

**What you will build:**
A complete LLM provider integration that connects to any LLM API (cloud or local), supports synchronous and streaming chat completions, handles tool/function calling, integrates with Victor's circuit breaker, and is properly registered and testable.

---

## Tutorial Parts

### [Part 1: Provider Architecture & Steps 1-2](integrate-provider/part-1-provider-architecture.md)
- The BaseProvider class
- Required and optional methods
- Core data types
- **Step 1:** Create the provider file
- **Step 2:** Implement the `chat()` method

### [Part 2: Streaming & Error Handling](integrate-provider/part-2-streaming-error-handling.md)
- **Step 3:** Implement the `stream()` method
- **Step 4:** Add error handling and `close()` method
- HTTP error handling
- Resource cleanup

### [Part 3: Tool Calling Adapter](integrate-provider/part-3-tool-calling-adapter.md)
- When to create an adapter
- Creating a custom tool calling adapter
- Converting tools to provider format
- Parsing tool calls from responses

### [Part 4: Registration & Testing](integrate-provider/part-4-registration-testing.md)
- Model capabilities configuration
- Provider registration (internal and external)
- Unit tests
- Integration tests

### [Part 5: Best Practices & Examples](integrate-provider/part-5-best-practices-examples.md)
- Error handling best practices
- Circuit breaker usage
- Retry strategies
- Complete provider example
- Quick reference checklist

---

## Prerequisites

- Python 3.11+
- Victor development environment set up (`pip install -e ".[dev]"`)
- API access to the LLM provider you want to integrate
- Basic understanding of async/await in Python

---

## Quick Reference

### Required Methods

| Method | Description |
|--------|-------------|
| `name` (property) | Returns the provider identifier |
| `chat()` | Sends a chat completion request |
| `stream()` | Streams a chat completion response |
| `close()` | Closes connections and resources |

### Error Types

| Error Type | When to Use |
|------------|-------------|
| `ProviderError` | Base class for all provider errors |
| `ProviderAuthError` | Authentication/authorization failures (401, 403) |
| `ProviderRateLimitError` | Rate limiting (429) |
| `ProviderTimeoutError` | Request timeouts |

### Import Locations

```python
# Provider base classes and types
from victor.providers.base import (
    BaseProvider,
    Message,
    CompletionResponse,
    StreamChunk,
    ToolDefinition,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Provider registry
from victor.providers.registry import ProviderRegistry

# Tool calling adapter base
from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    FallbackParsingMixin,
    ToolCallingCapabilities,
    ToolCallFormat,
)
```

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes (complete tutorial)
