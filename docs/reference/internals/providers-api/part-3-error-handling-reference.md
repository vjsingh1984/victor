# Providers API Reference - Part 3

**Part 3 of 3:** Error Handling and References

---

## Navigation

- [Part 1: Interface, Protocols, Classes](part-1-interface-protocols-classes.md)
- [Part 2: Implementation & Config](part-2-implementation-config.md)
- **[Part 3: Error Handling & Reference](#)** (Current)
- [**Complete Reference**](../providers-api.md)

---

Victor provides a hierarchy of provider errors for granular error handling.

**Import:**
```python
from victor.providers.base import (
    ProviderError,
    ProviderNotFoundError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderInvalidResponseError,
)
```

### Error Classes

```python
class ProviderError(Exception):
    """Base exception for all provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        raw_error: Optional[Exception] = None,
    ):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.raw_error = raw_error


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered."""


class ProviderAuthError(ProviderError):
    """Raised when authentication fails (invalid API key, etc.)."""


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded (HTTP 429)."""


class ProviderTimeoutError(ProviderError):
    """Raised when a request times out."""


class ProviderConnectionError(ProviderError):
    """Raised when connection to provider fails."""


class ProviderInvalidResponseError(ProviderError):
    """Raised when provider returns invalid/unparseable response."""
```

### Circuit Breaker

The circuit breaker pattern prevents cascading failures when providers are unavailable.

```python
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)

# Circuit states
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

# Get circuit breaker stats
stats = provider.get_circuit_breaker_stats()
# {
#     "name": "provider_AnthropicProvider",
#     "state": "closed",
#     "total_calls": 100,
#     "total_failures": 2,
#     "total_rejected": 0,
#     "failure_count": 0,
#     "success_count": 0,
# }

# Reset circuit breaker manually
provider.reset_circuit_breaker()
```

### Error Handling Example

```python
from victor.providers import (
    ProviderRegistry,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

async def call_llm(messages, model):
    provider = ProviderRegistry.create("anthropic", api_key="...")

    try:
        response = await provider.chat(messages, model=model)
        return response.content

    except ProviderAuthError as e:
        logger.error(f"Authentication failed: {e.message}")
        raise

    except ProviderRateLimitError as e:
        logger.warning(f"Rate limited, retry after delay")
        await asyncio.sleep(60)
        return await call_llm(messages, model)  # Retry

    except ProviderTimeoutError as e:
        logger.warning(f"Request timed out: {e.message}")
        raise

    except ProviderError as e:
        logger.error(f"Provider error: {e.message}")
        raise

    finally:
        await provider.close()
```

---

## See Also

- [Tool System API Reference](./tools-api.md)
- [Workflow Engine API Reference](./workflows-api.md)
- [Configuration Guide](../../getting-started/configuration.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
